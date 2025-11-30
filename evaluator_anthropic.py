# evaluate_tool_selection.py
import json
import os
import pandas as pd
import anthropic # Using the Anthropic library
from dotenv import load_dotenv
from tqdm import tqdm
import time
import logging
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- User-Configurable Variables ---
NUM_CHOICES = 100
NUM_TESTS = 500
MODEL_TO_TEST = "claude-3-5-haiku-20241022"
NUM_WORKERS = 4 

# --- Configuration ---
load_dotenv()
INPUT_CSV = "mcp_zero_openai_queries_v2.csv"
OUTPUT_CSV = f"evaluation_results_claude_3.5_haiku_100_context.csv"
LOG_FILE = "evaluation_anthropic_tool_use.log"
MAX_API_RETRIES = 3

# Instantiate the Anthropic client
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if ANTHROPIC_API_KEY:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
else:
    client = None

# --- Logging and Thread-Safe Utilities (No changes here) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s', filename=LOG_FILE, filemode='w')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)
print_lock = threading.Lock()
csv_lock = threading.Lock()
worker_id_map = {}
next_worker_id = 1
def get_worker_id():
    global next_worker_id
    thread_id = threading.get_ident()
    with print_lock:
        if thread_id not in worker_id_map:
            worker_id_map[thread_id] = next_worker_id
            next_worker_id += 1
        return worker_id_map[thread_id]

# --- MODIFICATION START: Tool and Prompt Definition ---

# The system prompt is now simpler. We just tell it to use the provided tool.
EVALUATION_SYSTEM_PROMPT = """You are a highly intelligent AI agent. Your task is to select the single most appropriate tool from the provided list to handle the user's query."""

# Define the "tool" the model can use. This is a JSON schema.
# It tells the model it has one function available: `select_tool`, which takes one argument: `tool_name`.
TOOL_SPEC = {
    "name": "select_tool",
    "description": "Select the single most appropriate tool to handle the user's query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "The name of the chosen tool. This must be one of the tools from the provided list."
            }
        },
        "required": ["tool_name"]
    }
}

def call_evaluation_llm(user_prompt_content):
    """
    Calls the Anthropic model using the Tool Use feature.
    """
    for attempt in range(MAX_API_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL_TO_TEST,
                system=EVALUATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt_content}],
                tools=[TOOL_SPEC],          # Pass the tool specification here
                tool_choice={"type": "tool", "name": "select_tool"}, # Force the model to use our tool
                temperature=0.0,
                max_tokens=1024
            )
            return response
        except Exception as e:
            logging.error(f"API Error on attempt {attempt + 1}: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    logging.critical(f"API call failed after {MAX_API_RETRIES} retries.")
    return None

def run_single_test(job_data):
    worker_id = get_worker_id()
    test_case = job_data['test_case']
    all_tools_df = job_data['all_tools_df']
    correct_tool_name = test_case['tool_name']
    
    # 1. Prepare tool choices (same as before)
    correct_tool_details = all_tools_df[all_tools_df['tool_name'] == correct_tool_name].iloc[0]
    distractors_df = all_tools_df[all_tools_df['tool_name'] != correct_tool_name]
    if len(distractors_df) < NUM_CHOICES - 1:
        distractor_sample = distractors_df
    else:
        distractor_sample = distractors_df.sample(n=NUM_CHOICES - 1, random_state=int(time.time()) + worker_id)
    choices = pd.concat([pd.DataFrame([correct_tool_details]), distractor_sample]).sample(frac=1).reset_index(drop=True)

    # 2. Format the prompt (same as before)
    tool_list_str = ""
    for idx, row in choices.iterrows():
        tool_list_str += f"{idx + 1}. name: {row['tool_name']}\n   description: {row['tool_description']}\n\n"
    user_prompt = f"User Query: \"{test_case['generated_query']}\"\n\n### Available Tools:\n{tool_list_str}Select the single best tool from the list above to address the user query."

    # 3. Call the LLM and parse the response
    response = call_evaluation_llm(user_prompt)
    llm_choice = "API_FAILURE"

    if response and response.stop_reason == "tool_use":
        # Find the tool_use content block
        tool_use_block = next((block for block in response.content if block.type == "tool_use"), None)
        if tool_use_block:
            # The model's choice is in the 'input' dictionary of the tool call
            llm_choice = tool_use_block.input.get("tool_name", "MISSING_KEY")
        else:
            llm_choice = "TOOL_USE_BLOCK_NOT_FOUND"
    elif response:
        llm_choice = f"WRONG_STOP_REASON:_{response.stop_reason}"

    # --- MODIFICATION END ---

    # 4. Determine correctness and log result (same as before)
    is_correct = 1 if llm_choice == correct_tool_name else 0
    log_message = f"Tool: {correct_tool_name}, Persona: {test_case['persona']}, Correct: {bool(is_correct)}, LLM chose: {llm_choice}"
    if is_correct:
        logging.info(f"SUCCESS | {log_message}")
    else:
        logging.warning(f"FAILURE | {log_message}")

    # 5. Prepare and save result (same as before)
    result = {'tool_tested': correct_tool_name, 'persona': test_case['persona'], 'num_choices': NUM_CHOICES, 'llm_choice': llm_choice, 'correct_choice': correct_tool_name, 'is_correct': is_correct}
    with csv_lock:
        pd.DataFrame([result]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False, encoding='utf-8')

def main():
    # Main function remains largely the same, just checking for the correct API key
    if not ANTHROPIC_API_KEY or client is None:
        logging.critical("CRITICAL: ANTHROPIC_API_KEY not found in .env file or client failed to initialize. Exiting.")
        print("Please ensure your .env file contains: ANTHROPIC_API_KEY=<your_anthropic_api_key>")
        return
    # ... (rest of main function is identical to your last Anthropic script)
    try:
        test_cases_df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        logging.critical(f"CRITICAL: Input file '{INPUT_CSV}' not found. Exiting.")
        return
    all_tools_df = test_cases_df[['server_name', 'tool_name', 'tool_description']].drop_duplicates().reset_index(drop=True)
    completed_jobs = set()
    if os.path.exists(OUTPUT_CSV):
        print(f"Resuming from existing results file: '{OUTPUT_CSV}'")
        df_existing = pd.read_csv(OUTPUT_CSV)
        for _, row in df_existing.iterrows():
            completed_jobs.add((row['tool_tested'], row['persona']))
        print(f"Found {len(completed_jobs)} completed tests to skip.")
    else:
        pd.DataFrame(columns=['tool_tested', 'persona', 'num_choices', 'llm_choice', 'correct_choice', 'is_correct']).to_csv(OUTPUT_CSV, index=False)
    all_test_cases = [row for _, row in test_cases_df.iterrows()]
    pending_test_cases = [tc for tc in all_test_cases if (tc['tool_name'], tc['persona']) not in completed_jobs]
    if NUM_TESTS is not None:
        pending_test_cases = pending_test_cases[:NUM_TESTS]
    if not pending_test_cases:
        print("All specified tests are already complete.")
        return
    print(f"Starting evaluation of {len(pending_test_cases)} queries using {NUM_WORKERS} workers...")
    jobs_to_run = [{ 'test_case': tc, 'all_tools_df': all_tools_df } for tc in pending_test_cases]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(run_single_test, job) for job in jobs_to_run]
        for future in tqdm(as_completed(futures), total=len(jobs_to_run), desc="Evaluating Tools"):
            try:
                future.result()
            except Exception as exc:
                logging.error(f"A worker thread generated an exception: {exc}")
    print("\nEvaluation complete. Calculating final results...")
    final_df = pd.read_csv(OUTPUT_CSV)
    total_tested = len(final_df)
    if total_tested > 0:
        total_correct = final_df['is_correct'].sum()
        accuracy = (total_correct / total_tested) * 100
        print(f"\n--- OVERALL LLM PERFORMANCE ---\nModel Tested:      {MODEL_TO_TEST}\nTotal Tests Run:   {total_tested}\nTotal Correct:     {total_correct}\nAccuracy:          {accuracy:.2f}%\n-------------------------------")
    else:
        print("No results found to analyze.")

if __name__ == "__main__":
    main()