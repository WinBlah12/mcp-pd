import json
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
load_dotenv()  # Load OPENAI_API_KEY from .env file

# File Paths
INPUT_JSON = "mcp_tools_with_embedding.json"
OUTPUT_CSV = "mcp_zero_openai_queries_v2.csv"
OUTPUT_XLSX = "mcp_zero_openai_queries_v2.xlsx"
LOG_FILE = "generation.log"

# Models & API Settings
GENERATOR_MODEL = "gpt-4o-mini"
CRITIC_MODEL = "gpt-4o"
MAX_RETRIES = 2
NUM_WORKERS = 8  # Number of concurrent threads to run

# Processing Limits (for testing)
# SET THIS TO A NUMBER (e.g., 5) for a small test run.
# SET TO None to process the entire file.
TEST_LIMIT = 500

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    filename=LOG_FILE,
    filemode='a' # Append to the log file
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING) # Only show warnings and errors in the console
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


# --- Thread-Safe Utilities ---
# We use locks to prevent threads from writing to the console or CSV file at the same time
print_lock = threading.Lock()
csv_lock = threading.Lock()
worker_id_map = {}
next_worker_id = 1

def get_worker_id():
    """Assigns a simple, readable ID to each thread for logging."""
    global next_worker_id
    thread_id = threading.get_ident()
    with print_lock:
        if thread_id not in worker_id_map:
            worker_id_map[thread_id] = next_worker_id
            next_worker_id += 1
        return worker_id_map[thread_id]

def thread_safe_print(message):
    """A print function that prevents jumbled output from multiple threads."""
    with print_lock:
        print(message)

# Initialize OpenAI client (it's thread-safe by default)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- PERSONA DEFINITIONS ---
PERSONAS = {
    "problem_oriented": "User is a complete novice. They describe a problem or ask a question in simple, everyday language, with no awareness of any specific tools or technical concepts.",
    "goal_oriented": "User knows the final outcome they want to achieve but has no idea how to get there. The query is focused on the end state, not the process or the tool.",
    "category_aware": "User understands the general category of the tool they need but doesn't know the specific tool name. They might use some light technical jargon related to the task domain.",
    "function_specific": "User is technically skilled. They describe the exact function and parameters they need with precision, but they do not know the specific name of the tool to call.",
    "tool_explicit": "User is an expert who knows the exact tool they want to use. They mention the tool name directly and may also specify its parameters in their request."
}

# --- PROMPTS ---
GENERATOR_SYSTEM_PROMPT = """You are an expert in generating synthetic evaluation data for AI agents. 
Your task is to create realistic user queries that would trigger a specific software tool (MCP Tool).
You must adhere strictly to the requested USER PERSONA."""

CRITIC_SYSTEM_PROMPT = """You are a strict QA auditor for a synthetic dataset. 
Your job is to evaluate if a generated user query accurately reflects a specific USER PERSONA 
and if it legitimately applies to the provided TOOL definition.
Output JSON only: {"status": "PASS" or "FAIL", "reason": "short explanation"}."""

def call_llm(model, messages, is_json=False):
    """Generic function to call OpenAI API with error handling and backoff."""
    try:
        response_format = {"type": "json_object"} if is_json else {"type": "text"}
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0 if not is_json else 0.0,
            response_format=response_format
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"API Error with model {model}: {e}")
        time.sleep(5)  # Wait before retrying
        return None

def call_generator(server, tool_name, tool_desc, persona_key, feedback=None):
    """Calls the generator LLM."""
    persona_desc = PERSONAS[persona_key]
    prompt = f"TOOL CONTEXT\nServer: {server}\nTool Name: {tool_name}\nDescription: {tool_desc}\n\n" \
             f"TARGET PERSONA ({persona_key.upper()})\n{persona_desc}\n\n" \
             "TASK\nGenerate ONE natural language user query based on the context and persona. " \
             "Do not include any introductory text, just the query."
    if feedback:
        prompt += f"\n\nNOTE: A previous attempt failed. Feedback: '{feedback}'. Please generate a new, improved query."

    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    query = call_llm(GENERATOR_MODEL, messages)
    return query.strip().replace('"', '') if query else None

def call_critic(server, tool_name, tool_desc, persona_key, generated_query):
    """Calls the critic LLM to verify the query."""
    persona_desc = PERSONAS[persona_key]
    prompt = f"TOOL CONTEXT\nServer: {server}\nTool Name: {tool_name}\nDescription: {tool_desc}\n\n" \
             f"TARGET PERSONA ({persona_key.upper()})\nDefinition: {persona_desc}\n\n" \
             f'GENERATED QUERY TO AUDIT\n"{generated_query}"\n\n' \
             "AUDIT CRITERIA\n1. Does the query fit the Target Persona definition perfectly?\n" \
             "2. Would a human reasonably ask this to use the functionality described in the Tool Context?\n" \
             "3. Is it distinct from other personas? (e.g., a 'vague' query should NOT mention the tool name).\n\n" \
             "Evaluate and respond ONLY in JSON format."

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    response_str = call_llm(CRITIC_MODEL, messages, is_json=True)
    if response_str:
        try:
            result = json.loads(response_str)
            return result.get("status") == "PASS", result.get("reason", "No reason provided.")
        except json.JSONDecodeError:
            logging.error(f"Critic returned invalid JSON: {response_str}")
            return False, f"Critic returned invalid JSON: {response_str}"
    return False, "Critic API call failed."

def process_tool_persona(job):
    """
    The main worker function for each thread. It takes a 'job' dictionary,
    generates a query, gets it validated, and saves it.
    """
    server_name, tool_name, tool_desc, persona_key = job['server'], job['name'], job['description'], job['persona']
    worker_id = get_worker_id()
    
    thread_safe_print(f"[WORKER {worker_id}] Processing: {tool_name} | Persona: {persona_key}")

    current_feedback = None
    final_query = None
    for attempt in range(MAX_RETRIES + 1):
        query = call_generator(server_name, tool_name, tool_desc, persona_key, current_feedback)
        if not query:
            final_query = "GENERATION_FAILED"
            logging.error(f"FAILED to generate query for Tool='{tool_name}', Persona='{persona_key}' after API error.")
            break
        
        passed, reason = call_critic(server_name, tool_name, tool_desc, persona_key, query)
        
        if passed:
            final_query = query
            logging.info(f"SUCCESS: Generated query for Tool='{tool_name}', Persona='{persona_key}'")
            break
        else:
            current_feedback = reason
            logging.warning(f"CRITIC FAIL (Attempt {attempt+1}/{MAX_RETRIES+1}) for Tool='{tool_name}', Persona='{persona_key}'. Reason: {reason}")
            time.sleep(0.5) # Small delay before retry
            final_query = query # Keep the last failed attempt as a fallback

    # Create the result dictionary
    result = {
        "server_name": server_name,
        "tool_name": tool_name,
        "tool_description": tool_desc,
        "persona": persona_key,
        "generated_query": final_query
    }

    # Append the result to the CSV file immediately and safely
    with csv_lock:
        df_result = pd.DataFrame([result])
        df_result.to_csv(OUTPUT_CSV, mode='a', header=False, index=False, encoding='utf-8')
    
    return result

def main():
    if not os.getenv("OPENAI_API_KEY"):
        logging.critical("CRITICAL: OPENAI_API_KEY not found in .env file. Exiting.")
        return

    # --- 1. Load and Prepare Jobs ---
    try:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            mcp_servers = json.load(f)
    except Exception as e:
        logging.critical(f"CRITICAL: Error reading {INPUT_JSON}: {e}. Exiting.")
        return

    all_jobs = []
    for server in mcp_servers:
        if "tools" in server and isinstance(server["tools"], list):
            for tool in server["tools"]:
                for p_key in PERSONAS.keys():
                    all_jobs.append({
                        "server": server.get("name"),
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "persona": p_key
                    })

    # --- 2. Check for Existing Results to Resume ---
    completed_jobs = set()
    if os.path.exists(OUTPUT_CSV):
        print(f"Found existing output file '{OUTPUT_CSV}'. Reading to resume progress.")
        df_existing = pd.read_csv(OUTPUT_CSV)
        for _, row in df_existing.iterrows():
            completed_jobs.add((row['server_name'], row['tool_name'], row['persona']))
        print(f"Resuming. Found {len(completed_jobs)} completed jobs to skip.")
    else:
        # Create the file and write the header
        pd.DataFrame(columns=["server_name", "tool_name", "tool_description", "persona", "generated_query"]).to_csv(OUTPUT_CSV, index=False)

    
    # Filter out jobs that are already completed
    pending_jobs = [job for job in all_jobs if (job['server'], job['name'], job['persona']) not in completed_jobs]
    
    # Apply testing limit if specified
    if TEST_LIMIT is not None:
        pending_jobs = pending_jobs[:TEST_LIMIT * len(PERSONAS)]
        print(f"!!! TEST MODE: Will process up to {len(pending_jobs)} new queries. !!!")
    
    if not pending_jobs:
        print("No pending jobs to process. All tasks are complete.")
    else:
        print(f"Starting Generator/Critic Loop for {len(pending_jobs)} queries using {NUM_WORKERS} workers...")
        
        # --- 3. Execute Jobs with Multithreading ---
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Use as_completed to process results as they finish, with a tqdm progress bar
            futures = [executor.submit(process_tool_persona, job) for job in pending_jobs]
            for future in tqdm(as_completed(futures), total=len(pending_jobs)):
                try:
                    future.result() # We just need to check if it finished, result is already saved
                except Exception as e:
                    logging.error(f"A worker thread raised an exception: {e}")

    # --- 4. Final Sort and Save ---
    print("\nGeneration complete. Reading and sorting final dataset...")
    df_final = pd.read_csv(OUTPUT_CSV)
    
    # Define a custom sort order for personas
    persona_order = list(PERSONAS.keys())
    df_final['persona'] = pd.Categorical(df_final['persona'], categories=persona_order, ordered=True)
    
    # Sort the dataframe
    df_final_sorted = df_final.sort_values(by=['server_name', 'tool_name', 'persona']).reset_index(drop=True)
    
    # Overwrite the CSV and write the Excel file with the sorted data
    df_final_sorted.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    df_final_sorted.to_excel(OUTPUT_XLSX, index=False)

    print("-" * 30)
    print(f"Successfully processed and saved {len(df_final_sorted)} queries.")
    print(f"Sorted output saved to '{OUTPUT_CSV}' and '{OUTPUT_XLSX}'.")
    print(f"Check '{LOG_FILE}' for detailed logs.")

if __name__ == "__main__":
    main()