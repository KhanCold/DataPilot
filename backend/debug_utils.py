import os
from datetime import datetime

LOG_DIR = "logs"
PROMPT_LOG_FILE = os.path.join(LOG_DIR, "prompts.log")

def log_prompt_to_file(prompt_name: str, prompt_content: str):
    """
    Appends a formatted prompt to the prompt log file for debugging.

    Args:
        prompt_name (str): The name of the prompt (e.g., "planner_prompt").
        prompt_content (str): The full content of the prompt.
    """
    try:
        # Ensure the log directory exists
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        with open(PROMPT_LOG_FILE, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"""
================================================================================
Timestamp: {timestamp}
Prompt Name: {prompt_name}
================================================================================
{prompt_content}
================================ End of Prompt =================================
\n\n
"""
            f.write(log_entry)
    except IOError as e:
        print(f"Error writing to prompt log file: {e}")
