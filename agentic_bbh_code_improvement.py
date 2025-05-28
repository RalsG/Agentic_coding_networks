import os
import subprocess
import sys
import logging # Added
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # Added for safety settings
import time
import json
import shutil
import ast
import scri
import h5py


PROMPT_DIR = "Prompts" # Or an absolute path if needed

def load_prompt(filename: str) -> str:
    filepath = os.path.join(PROMPT_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"CRITICAL: Prompt file not found: {filepath}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"CRITICAL: Error loading prompt file {filepath}: {e}. Exiting.")
        sys.exit(1)
# --- Logging Configuration ---
# Configure logging (Set to DEBUG for detailed output, include filename and line number)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment and API Configuration ---
if not load_dotenv():
    logger.warning(".env file not found or failed to load. Environment variables might not be set.")

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    logger.error("CRITICAL: GOOGLE_API_KEY not found in environment variables. Please set it.")
    sys.exit(1)

try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Google Generative AI SDK configured successfully.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to configure Google Generative AI SDK: {e}")
    sys.exit(1)

MODEL_NAME = 'models/gemini-1.5-flash-latest' # Updated to a common, potentially more robust flash model
# MODEL_NAME = 'models/gemini-1.5-pro-latest' # Consider for more complex tasks if flash has issues
logger.info(f"Using LLM Model: {MODEL_NAME}")

try:
    model = genai.GenerativeModel(MODEL_NAME)
    logger.info(f"Successfully initialized GenerativeModel: {MODEL_NAME}")
except Exception as e:
    logger.error(f"CRITICAL: Failed to initialize GenerativeModel ({MODEL_NAME}): {e}")
    sys.exit(1)

# Safety settings for code generation (can be tuned)
# Keeping BLOCK_NONE as per original intent of Version 1, but this is RISKY.
# Consider HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE or BLOCK_LOW_AND_ABOVE for DANGEROUS_CONTENT.
CODE_GENERATION_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}
# Safety settings for non-code generation prompts (e.g., analysis, documentation)
# Can be more restrictive
DEFAULT_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


# --- Agent Prompts (Modified for Python) ---
physics_spec_agent_prompt = load_prompt("a1_bbh_visualization_specifier.txt")
python_developer_agent_prompt = load_prompt("a2_bbh_python_developer.txt")
manager_agent_prompt = load_prompt("a3_bbh_manager_review.txt")
python_qa_agent_prompt = load_prompt("a4_bbh_python_qa.txt")
error_interpreter_prompt = load_prompt("a5_error_interpreter.txt")
mathematician_agent_prompt = load_prompt("a6_bbh_computational_physicist_reviewer.txt") # Note: A-6 in your current code is selector
code_commenter_agent_prompt = load_prompt("a7_code_commenter.txt")
documentation_agent_prompt = load_prompt("a8_bbh_documentation.txt")
multi_version_selector_agent_prompt = load_prompt("a9_bbh_multi_version_selector.txt") # A-9 in your current code
A9_DEEP_REVIEWER_PROMPT = load_prompt("ax_bbh_deep_reviewer.txt")
#--- START: NEW PROMPT FOR A-X ---


ALL_PROMPTS = {
"physics_spec_agent_prompt": physics_spec_agent_prompt,
"python_developer_agent_prompt": python_developer_agent_prompt,
"manager_agent_prompt": manager_agent_prompt,
"python_qa_agent_prompt": python_qa_agent_prompt,
"error_interpreter_prompt": error_interpreter_prompt,
"multi_version_selector_agent_prompt": multi_version_selector_agent_prompt,
"mathematician_agent_prompt": mathematician_agent_prompt,
"code_commenter_agent_prompt": code_commenter_agent_prompt,
"documentation_agent_prompt": documentation_agent_prompt,
"AX_DEEP_REVIEWER_PROMPT": A9_DEEP_REVIEWER_PROMPT, # Added new prompt
}

def save_prompts_to_file(prompts_dict, model_name_for_file, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    # Ensure valid filename
    safe_model_name = model_name_for_file.replace('/', '').replace(':', '')
    filename = os.path.join(
        output_dir, f"prompts_used_{safe_model_name}.txt")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(
                f"--- Prompts used with Model: {model_name_for_file} ---\n\n")
            for name, prompt_text in prompts_dict.items():
                f.write(f"--- PROMPT NAME: {name} ---\n")
                f.write(prompt_text)
                f.write("\n\n--- END OF PROMPT ---\n\n")
        logger.info(f"All prompts saved to {filename}")
    except IOError as e:
        logger.error(f"Error saving prompts to {filename}: {e}")


def strip_code_block_markers(code_string: str | None) -> str:
    if code_string is None:
        return ""

    processed_string = code_string.strip()
    # Check for ```python, ```json, etc. and ```
    # More robustly remove them if they are on their own lines at start/end
    lines = processed_string.splitlines()

    if not lines:
        return ""

    # Remove leading marker if it's the first line (and possibly only content before code)
    first_line_stripped = lines[0].strip()
    if first_line_stripped.startswith("```"):  # Handles ```, ```python, ```json etc.
        # Check if there's anything else on this line other than the marker and optional language specifier
        # A simple heuristic: if the line is short and starts with ```, it's likely just the marker.
        if len(first_line_stripped) < 20:  # Arbitrary short length, covers ```python, ```json etc.
            lines.pop(0)
            if not lines:  # If that was the only line
                return ""
        # If it's longer, it might be actual content that happens to start with ```.
        # The current logic relies on the LLM mostly following "ONLY code" for code generation.

    # Remove trailing marker if it's the last line
    if lines and lines[-1].strip() == "```":
        lines.pop(-1)
        if not lines:  # If that was the only line left
            return ""

    return "\n".join(lines).strip()


def generate_llm_response(prompt_template: str, agent_safety_settings: dict | None = None, **kwargs) -> str | None:
    """Generates a response from the LLM based on a prompt template and arguments."""
    global model  # Assuming model is initialized globally
    # Assuming this is defined globally
    global DEFAULT_SAFETY_SETTINGS
    # Determine safety settings: use agent-specific if provided, else default
    current_safety_settings = agent_safety_settings if agent_safety_settings is not None else DEFAULT_SAFETY_SETTINGS

    try:
        prompt = prompt_template.format(**kwargs)
    except KeyError as e:
        logger.error(
            f"Error formatting prompt template: Missing key {e}. Check prompt template and kwargs.")
        logger.debug(f"Template (first 500 chars): {prompt_template[:500]}...")
        logger.debug(f"Available Kwargs: {list(kwargs.keys())}")
        return None

    prompt_log_snippet = prompt[:1000] + \
        ("..." if len(prompt) > 1000 else "")
    logger.debug(
        f"--- PROMPT SENT TO LLM (length: {len(prompt)}) ---\n{prompt_log_snippet}\n--- END PROMPT SNIPPET ---")

    max_retries = 3

    generation_config = genai.types.GenerationConfig(
        max_output_tokens=32768,
        temperature=0.3,
    )

    for attempt in range(max_retries):
        logger.info(
            f"Attempting LLM call (Attempt {attempt + 1}/{max_retries})...")
        try:
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=current_safety_settings
            )

            full_text_response = ""
            # `candidate.finish_reason` is expected to be an enum instance directly
            # (e.g., an instance of a FinishReason enum).
            finish_reason_enum_instance = None

            if response and response.candidates:
                candidate = response.candidates[0]
                finish_reason_enum_instance = candidate.finish_reason

                if candidate.content and candidate.content.parts:
                    full_text_response = "".join(
                        part.text for part in candidate.content.parts)

            if finish_reason_enum_instance is not None:
                # Log the name of the finish reason.
                logger.info(
                    f"LLM call attempt {attempt+1} finished with reason: {finish_reason_enum_instance.name}")

                # Get the specific Enum Type from the instance itself
                FinishReasonEnumType = type(finish_reason_enum_instance)

                # Compare the instance with members of its own type
                if finish_reason_enum_instance == FinishReasonEnumType.STOP or \
                   finish_reason_enum_instance == FinishReasonEnumType.MAX_TOKENS:
                    if full_text_response.strip():
                        logger.info(
                            f"LLM response received (length: {len(full_text_response)} chars).")
                        logger.debug(
                            f"LLM Response (first 200): {full_text_response[:200]}...")
                        if finish_reason_enum_instance == FinishReasonEnumType.MAX_TOKENS:
                            logger.warning(
                                "LLM response was truncated due to MAX_TOKENS. Content may be incomplete.")
                        return full_text_response
                    else:
                        logger.warning(
                            f"LLM finished with {finish_reason_enum_instance.name} but returned empty text content on attempt {attempt+1}.")

                elif finish_reason_enum_instance == FinishReasonEnumType.SAFETY or \
                        finish_reason_enum_instance == FinishReasonEnumType.RECITATION:
                    logger.error(
                        f"Generation definitively blocked or problematic due to {finish_reason_enum_instance.name}. Not using this response.")
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        # Access block_reason's name if it's an enum, or the message directly
                        block_reason_detail = response.prompt_feedback.block_reason
                        if hasattr(block_reason_detail, 'name'):  # It's an enum
                            block_reason_log = block_reason_detail.name
                        else:  # It might be a string or other type
                            block_reason_log = str(block_reason_detail)
                        logger.error(
                            f"Prompt Feedback Block Reason: {response.prompt_feedback.block_reason_message or block_reason_log}")
                    return None
                else:  # OTHER, UNSPECIFIED, UNKNOWN
                    logger.warning(
                        f"LLM generation finished with non-ideal reason: {finish_reason_enum_instance.name} on attempt {attempt + 1}.")
                    # Fall through to retry logic for these less common but non-fatal reasons
            else:
                logger.warning(
                    f"Could not determine finish reason from LLM response candidate on attempt {attempt+1}.")
                # Fall through to retry logic

            logger.warning(
                f"No valid content extracted, or non-ideal finish reason without usable content on attempt {attempt + 1}.")
            if response and response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_detail = response.prompt_feedback.block_reason
                block_reason_log = block_reason_detail.name if hasattr(
                    block_reason_detail, 'name') else str(block_reason_detail)
                logger.error(
                    f"Prompt was blocked! Reason: {response.prompt_feedback.block_reason_message or block_reason_log}")
                return None

            if attempt < max_retries - 1:
                logger.info("Retrying LLM call after a delay...")
                time.sleep(min(30, (2 ** attempt) * 5))
                continue
            else:
                logger.error(
                    "Max retries reached after issues with response structure or finish reason.")
                return None

        except AttributeError as e:
            # This catch might be triggered if finish_reason_enum_instance is None and .name is accessed,
            # or if FinishReasonEnumType.STOP (etc.) access fails for some reason (e.g. not a proper enum).
            logger.error(
                f"AttributeError during LLM response processing (attempt {attempt+1}): {e}", exc_info=True)
            # If the error is related to accessing STOP etc. on FinishReasonEnumType, it means finish_reason_enum_instance was not a standard enum.
            if "object has no attribute" in str(e) and "FinishReasonEnumType" in str(e):
                logger.error(
                    "The 'finish_reason' from the candidate does not appear to be a standard enum type with members like STOP, MAX_TOKENS.")
            # Fall through to retry or fail
        except Exception as e:
            logger.error(
                f"Exception during LLM call on attempt {attempt+1}: {e}", exc_info=True)
            error_message = str(e).lower()

            if "api key not valid" in error_message:
                logger.error(
                    "API Key error. Please check your GOOGLE_API_KEY.")
                return None

            is_rate_limit_error = (
                "429" in error_message or
                "resource_exhausted" in error_message or
                "quota" in error_message or
                "rate limit" in error_message or
                "userratelimitexceeded" in error_message
            )

            if is_rate_limit_error:
                if attempt < max_retries - 1:
                    wait_time = min(60, (2 ** (attempt + 1)) * 10)
                    logger.warning(
                        f"Rate limit/quota hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        "Max retries reached for rate limit/quota error.")
                    return None

            if attempt < max_retries - 1:
                logger.warning(
                    f"Retrying after an unexpected error on attempt {attempt + 1}...")
                time.sleep(min(30, (2 ** attempt) * 5))
                continue
            else:
                logger.error(
                    "Max retries reached due to other persistent errors.")
                return None

    logger.error("All retries failed for LLM generation.")
    return None


# --- Python Specific Utility Functions ---

def run_subprocess_command(command: list, timeout: int, description: str, working_dir: str | None = None, env_vars: dict | None = None):
    logger.info(
        f"Executing {description}: {' '.join(command)} {f'(in {working_dir})' if working_dir else ''}")
    try:
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            cwd=working_dir,
            env=process_env
        )
        stdout_log = process.stdout if len(
            process.stdout) < 2000 else process.stdout[:2000] + "\n... (stdout truncated)"
        stderr_log = process.stderr if len(
            process.stderr) < 2000 else process.stderr[:2000] + "\n... (stderr truncated)"
        logger.debug(f"{description} STDOUT:\n{stdout_log}")
        if process.stderr.strip():  # Log stderr more prominently if not empty
            logger.warning(f"{description} STDERR:\n{stderr_log}")
        logger.info(f"{description} Return Code: {process.returncode}")
        return process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired:
        logger.error(f"{description} timed out after {timeout} seconds.")
        return "", f"{description} timed out after {timeout} seconds.", -99
    except FileNotFoundError:
        cmd_exe = command[0]
        logger.error(
            f"Command '{cmd_exe}' not found for {description}. Is it in your PATH or correctly specified?")
        return "", f"Command '{cmd_exe}' not found.", -98
    except PermissionError:
        logger.error(
            f"Permission denied to execute {command[0]} for {description}.")
        return "", f"Permission denied to execute {command[0]}.", -97
    except Exception as e:
        logger.error(
            f"Unexpected error during {description}: {e}", exc_info=True)
        return "", f"Unexpected error during {description}: {e}", -1


def check_python_syntax(python_file_path: str):
    """Checks Python syntax using ast.parse(). Returns (stdout, stderr, returncode)."""
    logger.info(f"Checking Python syntax for {python_file_path}...")
    try:
        with open(python_file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        ast.parse(source_code, filename=python_file_path)
        logger.info(f"Syntax OK for {python_file_path}")
        return f"Syntax OK for {python_file_path}", "", 0
    except SyntaxError as e:
        logger.warning(f"SyntaxError in {python_file_path}: {e}")
        return "", f"SyntaxError in {python_file_path}: {e}", 1
    except Exception as e:
        logger.error(
            f"Error checking syntax for {python_file_path}: {e}", exc_info=True)
        return "", f"Error checking syntax for {python_file_path}: {e}", 2


def run_python_tests(test_py_file_path: str, source_py_file_path: str):
    """Runs Python tests using the unittest module via direct execution of the test file."""
    if not os.path.exists(test_py_file_path):
        logger.error(f"Test file not found: {test_py_file_path}")
        return "", f"Test file not found: {test_py_file_path}", -1

    source_dir = os.path.dirname(os.path.abspath(source_py_file_path))
    test_env = {
        "PYTHONPATH": f"{source_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}

    command = [sys.executable, test_py_file_path]
    return run_subprocess_command(
        command,
        timeout=300,
        description=f"Running Python tests ({os.path.basename(test_py_file_path)})",
        env_vars=test_env
    )


def run_python_script(script_path: str, args: list = None):
    """Runs a Python script. Returns (stdout, stderr, returncode, execution_time)."""
    if args is None:
        args = []
    if not os.path.exists(script_path):
        logger.error(f"Python script not found at {script_path}")
        return "", f"Error: Python script not found at {script_path}", -1, -1.0

    execution_command = [sys.executable, script_path] + args

    start_time = time.time()
    stdout, stderr, retcode = run_subprocess_command(
        execution_command,
        timeout=900,
        description=f"Running Python script ({os.path.basename(script_path)})"
    )
    end_time = time.time()
    execution_time = end_time - start_time

    # Log execution time along with return status
    if retcode == 0:
        logger.info(
            f"Python script {os.path.basename(script_path)} executed successfully in {execution_time:.2f}s.")
    else:
        logger.warning(
            f"Python script {os.path.basename(script_path)} failed with return code {retcode} after {execution_time:.2f}s.")

    return stdout, stderr, retcode, execution_time
#--- Main Orchestration ---

import os
import shutil
import sys
import json
import time

# Assume logger, ALL_PROMPTS, MODEL_NAME, DEFAULT_SAFETY_SETTINGS,
# CODE_GENERATION_SAFETY_SETTINGS, and utility functions like
# save_prompts_to_file, generate_llm_response, strip_code_block_markers,
# check_python_syntax, run_python_tests, run_python_script
# are defined or imported elsewhere.

def main():
    N_CODE_VERSIONS = 3
    MAX_FIX_ATTEMPTS_PER_VERSION = 5
    # Changed dir name
    OUTPUT_BASE_DIR = "bbh_visualization_python_project_v1" # Name a folder to be created for all the output

    path_to_your_existing_code = "<path>" # CHANGE THIS
    existing_code_content = ""
    try:
        with open(path_to_your_existing_code, "r", encoding="utf-8") as f:
            existing_code_content = f.read()
        logger.info(f"Successfully loaded existing code from {path_to_your_existing_code}")
    except FileNotFoundError:
        logger.warning(f"Existing code file not found at {path_to_your_existing_code}. Proceeding without it.")
    except Exception as e:
        logger.error(f"Error loading existing code: {e}")

    if os.path.exists(OUTPUT_BASE_DIR):
        logger.info(f"Output directory {OUTPUT_BASE_DIR} exists. Removing it.")
        try:
            shutil.rmtree(OUTPUT_BASE_DIR)
        except OSError as e:
            logger.error(
                f"Failed to remove existing output directory {OUTPUT_BASE_DIR}: {e}. Please remove manually and retry.")
            sys.exit(1)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    logger.info(f"Created output directory: {OUTPUT_BASE_DIR}")

    save_prompts_to_file(ALL_PROMPTS, MODEL_NAME.replace(
        '/', '_'), output_dir=OUTPUT_BASE_DIR)

    logger.info(
        f"Starting Agentic Python Software Development for Binary Black Hole Visualization (Targeting {N_CODE_VERSIONS} versions).")

    # --- A-1: Physics Specifier Agent ---
    logger.info(
        "--- Step A-1: Generating Technical Specifications (for Python) ---")
    specifications = generate_llm_response(
        ALL_PROMPTS["physics_spec_agent_prompt"],
        agent_safety_settings=DEFAULT_SAFETY_SETTINGS
    )
    if not specifications:
        logger.error("CRITICAL: Failed to generate specifications. Exiting.")
        sys.exit(1)

    specifications = strip_code_block_markers(specifications)
    if not specifications.strip():
        logger.error(
            "CRITICAL: Generated specifications are empty after stripping. Exiting.")
        sys.exit(1)

    specs_file_path = os.path.join(
        OUTPUT_BASE_DIR, "bbh_visualization_specifications_python.txt")
    try:
        with open(specs_file_path, "w", encoding="utf-8") as f:
            f.write(specifications)
        logger.info(
            f"Python-targeted specifications generated and saved to {specs_file_path}")
    except IOError as e:
        logger.error(
            f"Failed to save specifications to {specs_file_path}: {e}. Exiting.")
        sys.exit(1)

    all_versions_data = []

    for version_idx in range(N_CODE_VERSIONS):
        version_id = f"v{version_idx}"
        logger.info(
            f"--- Generating Python Code Version {version_id} ({version_idx + 1}/{N_CODE_VERSIONS}) ---")

        version_output_dir = os.path.join(
            OUTPUT_BASE_DIR, f"version_{version_id}")
        os.makedirs(version_output_dir, exist_ok=True)
        logger.info(
            f"Created directory for version {version_id}: {version_output_dir}")

        current_py_file_name = f"bbh_visualizer_{version_id}.py"
        current_py_file_path = os.path.join(
            version_output_dir, current_py_file_name)
        current_module_name = current_py_file_name.replace(".py", "")

        current_test_py_file_name = f"test_{current_module_name}.py"
        current_test_py_file_path = os.path.join(
            version_output_dir, current_test_py_file_name)

        version_data = {
            "version_id": version_id, "python_code_content": None, "file_size_bytes": -1,
            "syntax_check_status": "Not Attempted", "test_results": "Not Run",
            "execution_time_seconds": -1.0, "main_exec_status": "Not Run",
            "mathematician_review": "Not Reviewed", "manager_review": "Not Reviewed",
            "deep_reviewer_feedback_log": [],  # NEW: To store A-X feedback history
            "final_feedback_summary_for_developer": "", "log": [f"Initialized version {version_id} data structure."]
        }
        current_code_content = None

        for attempt in range(MAX_FIX_ATTEMPTS_PER_VERSION):
            logger.info(
                f"Version {version_id} - Iteration {attempt + 1}/{MAX_FIX_ATTEMPTS_PER_VERSION} of A2-A5 Loop")
            version_data["log"].append(f"Starting iteration {attempt + 1}")

            # --- A-2: Python Programmer Agent ---
            logger.info(
                f"A-2: Generating/Fixing Python Code ({current_py_file_name})...")
            developer_prompt_args = {
                "existing_code_snippet": existing_code_content,
                "specifications": specifications,
                "target_py_file_name": current_py_file_name,
                "feedback_from_previous_attempts": version_data["final_feedback_summary_for_developer"]
            }
            raw_py_code = generate_llm_response(
                ALL_PROMPTS["python_developer_agent_prompt"],
                agent_safety_settings=CODE_GENERATION_SAFETY_SETTINGS,
                **developer_prompt_args
            )

            if not raw_py_code:
                logger.error(
                    f"Failed to generate Python code for {version_id} on attempt {attempt + 1}.")
                version_data["log"].append(
                    "A-2: Failed to generate Python code.")
                version_data["final_feedback_summary_for_developer"] = "Developer agent failed to produce Python code. Please try again, ensuring the output is complete Python code."
                if attempt == MAX_FIX_ATTEMPTS_PER_VERSION - 1:
                    version_data["syntax_check_status"] = "Code Generation Failed"
                continue

            current_code_content = strip_code_block_markers(raw_py_code)
            if not current_code_content.strip():
                logger.error(
                    f"Generated Python code for {version_id} was empty after stripping markers on attempt {attempt + 1}.")
                version_data["log"].append(
                    "A-2: Generated Python code was empty after stripping markers.")
                version_data["final_feedback_summary_for_developer"] = (
                    "Developer agent produced code that was empty after stripping markers. "
                    "Please ensure your output is valid Python code and not just formatting. "
                    f"The raw output started with: {raw_py_code[:200]}..."
                )
                if attempt == MAX_FIX_ATTEMPTS_PER_VERSION - 1:
                    version_data["syntax_check_status"] = "Code Generation Failed (Empty)"
                current_code_content = None
                continue

            try:
                with open(current_py_file_path, "w", encoding="utf-8") as f:
                    f.write(current_code_content)
                logger.info(
                    f"A-2: Python code generated and saved to {current_py_file_path}")
                version_data["python_code_content"] = current_code_content
                version_data["file_size_bytes"] = os.path.getsize(
                    current_py_file_path)
            except IOError as e:
                logger.error(
                    f"Failed to save Python code to {current_py_file_path}: {e}")
                version_data["log"].append(
                    f"A-2: Failed to save Python code: {e}")
                version_data["final_feedback_summary_for_developer"] = f"Internal error saving code: {e}. Please regenerate."
                current_code_content = None
                continue

            # Prepare for feedback accumulation in this iteration
            current_attempt_feedback = f"\n--- Feedback for Version {version_id}, Iteration {attempt + 1} ---\n"

            # --- A-3: Manager Agent (Python) ---
            logger.info(
                f"A-3: Reviewing Python Code ({current_py_file_name}) for best practices...")
            manager_feedback_raw = generate_llm_response(
                ALL_PROMPTS["manager_agent_prompt"],
                agent_safety_settings=DEFAULT_SAFETY_SETTINGS,
                specifications=specifications,
                python_code_content=current_code_content
            )
            manager_feedback = strip_code_block_markers(
                manager_feedback_raw) if manager_feedback_raw else "No feedback from Manager Agent."
            # Store the latest manager review
            version_data["manager_review"] = manager_feedback
            version_data["log"].append("A-3: Manager review completed.")
            try:
                with open(os.path.join(version_output_dir, f"manager_review_{version_id}_attempt_{attempt+1}.txt"), "w", encoding="utf-8") as f:
                    f.write(manager_feedback)
            except IOError as e:
                logger.warning(
                    f"Could not save manager review for {version_id}, attempt {attempt+1}: {e}")

            current_attempt_feedback += f"\n--- Manager (A-3) Python Review ---\n{manager_feedback}\n--- End Manager Review ---\n"

            # --- A-5 (Part 1): Check Python Syntax ---
            logger.info(
                f"A-5: Checking Python Syntax for ({current_py_file_path})...")
            syntax_stdout, syntax_stderr, syntax_retcode = check_python_syntax(
                current_py_file_path)
            version_data["log"].append(
                f"A-5: Main code syntax check: retcode={syntax_retcode}, stdout='{syntax_stdout[:100]}...', stderr='{syntax_stderr[:100]}...'")

            if syntax_retcode != 0:
                logger.warning(
                    f"Python syntax check failed for {version_id}.")
                version_data["syntax_check_status"] = "Failed"
                error_analysis = generate_llm_response(
                    ALL_PROMPTS["error_interpreter_prompt"],
                    agent_safety_settings=DEFAULT_SAFETY_SETTINGS,
                    execution_context=f"Checking Python syntax for main script {current_py_file_name}",
                    stdout=syntax_stdout, stderr=syntax_stderr, return_code=syntax_retcode
                ) or "Error interpreter failed for syntax check."
                current_attempt_feedback += f"\n--- Python Syntax Check (A-5) Feedback ---\n{error_analysis}\n--- End Syntax Check Feedback ---\n"
                version_data["final_feedback_summary_for_developer"] = current_attempt_feedback
                if attempt == MAX_FIX_ATTEMPTS_PER_VERSION - 1:
                    logger.error(
                        f"Max fix attempts reached for version {version_id} due to syntax errors.")
                continue

            logger.info(f"Python syntax OK for {version_id}!")
            version_data["syntax_check_status"] = "Success"
            current_attempt_feedback += "\n--- Python Syntax Check (A-5) Feedback ---\nPython syntax is OK.\n--- End Syntax Check Feedback ---\n"

            # --- START: NEW A-X Deep Reviewer Agent Call ---
            if current_code_content:  # Only if code is syntactically valid and exists
                logger.info(
                    f"A-X: Performing Deep Review for {version_id}, attempt {attempt+1}...")
                ax_reviewer_args = {
                    "version_id": version_id,
                    "python_code_content": current_code_content,
                    "syntax_check_status": version_data["syntax_check_status"],
                    # Use latest test results
                    "test_results_summary": version_data["test_results"],
                    # Use manager feedback from this attempt
                    "manager_review_summary": manager_feedback,
                    "specifications": specifications
                }
                ax_response_raw = generate_llm_response(
                    ALL_PROMPTS["AX_DEEP_REVIEWER_PROMPT"],
                    agent_safety_settings=DEFAULT_SAFETY_SETTINGS,  # Non-code generation
                    **ax_reviewer_args
                )

                ax_structured_feedback = None  # To store parsed JSON
                ax_feedback_for_developer_text = ""
                if ax_response_raw:
                    ax_response_cleaned = strip_code_block_markers(
                        ax_response_raw)
                    try:
                        ax_review_data = json.loads(ax_response_cleaned)
                        # Store for logging
                        ax_structured_feedback = ax_review_data
                        version_data["deep_reviewer_feedback_log"].append(
                            {f"attempt_{attempt+1}": ax_structured_feedback})

                        ax_feedback_for_developer_text += "\n--- Deep Review (A-X) Analysis ---\n"
                        if ax_review_data.get("identified_pros"):
                            ax_feedback_for_developer_text += "Positive Aspects (try to maintain these):\n"
                            for pro_item in ax_review_data["identified_pros"]:
                                ax_feedback_for_developer_text += f"- {pro_item}\n"
                            ax_feedback_for_developer_text += "---\n"

                        if ax_review_data.get("identified_cons_and_advice"):
                            ax_feedback_for_developer_text += "Areas for Improvement:\n"
                            for item in ax_review_data["identified_cons_and_advice"]:
                                ax_feedback_for_developer_text += f"Identified Con: {item.get('con')}\nSuggested Fix: {item.get('advice')}\n---\n"
                        ax_feedback_for_developer_text += "--- End Deep Review Analysis ---\n"

                        # Save this specific A-X review
                        try:
                            with open(os.path.join(version_output_dir, f"ax_deep_review_{version_id}_attempt_{attempt+1}.json"), "w", encoding="utf-8") as f_ax:
                                json.dump(ax_review_data, f_ax, indent=2)
                        except IOError as e_ax_save:
                            logger.warning(
                                f"Could not save A-X deep review JSON for {version_id}, attempt {attempt+1}: {e_ax_save}")

                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse A-X Deep Reviewer response for {version_id} as JSON: {e}")
                        ax_feedback_for_developer_text += f"\n--- Deep Review (A-X) Feedback ---\nDeep reviewer output was not valid JSON and could not be parsed. Raw response (check logs for full): {ax_response_cleaned[:300]}...\n--- End Deep Review Feedback ---\n"
                        version_data["deep_reviewer_feedback_log"].append(
                            {f"attempt_{attempt+1}": {"error": "JSONDecodeError", "raw_response": ax_response_cleaned}})
                    except Exception as e_ax:  # Catch any other parsing errors
                        logger.warning(
                            f"Error processing A-X Deep Reviewer response for {version_id}: {e_ax}")
                        ax_feedback_for_developer_text += f"\n--- Deep Review (A-X) Feedback ---\nError processing deep review: {str(e_ax)}\n--- End Deep Review Feedback ---\n"
                        version_data["deep_reviewer_feedback_log"].append(
                            {f"attempt_{attempt+1}": {"error": str(e_ax), "raw_response": ax_response_raw}})
                else:
                    logger.warning(
                        f"A-X Deep Reviewer failed to generate a response for {version_id}, attempt {attempt+1}.")
                    ax_feedback_for_developer_text += "\n--- Deep Review (A-X) Feedback ---\nDeep Reviewer Agent (A-X) did not provide a response.\n--- End Deep Review Feedback ---\n"
                    version_data["deep_reviewer_feedback_log"].append(
                        {f"attempt_{attempt+1}": "No response from A-X"})

                current_attempt_feedback += ax_feedback_for_developer_text
            # --- END: NEW A-X Deep Reviewer Agent Call ---

            # --- A-4: Python Quality Assurance Agent (Generate Test Code) ---
            logger.info(
                f"A-4: Generating Python Test Code for {version_id}...")
            # (Rest of A-4, A-5 Test Syntax, A-5 Test Run, A-5 Main Script Run remains largely the same,
            # but their feedback will be appended to current_attempt_feedback)

            raw_test_code = generate_llm_response(
                ALL_PROMPTS["python_qa_agent_prompt"],
                agent_safety_settings=CODE_GENERATION_SAFETY_SETTINGS,
                python_code_content=current_code_content,
                source_module_name=current_module_name,
                target_test_file_name=current_test_py_file_name
            )
            if not raw_test_code:
                logger.warning(
                    f"Failed to generate Python test code for {version_id}.")
                version_data["log"].append(
                    "A-4: Failed to generate Python test code.")
                current_attempt_feedback += "\n--- Test Code Generation (A-4) Feedback ---\nFailed to generate Python test code. The Programmer agent (A-2) might need to ensure function signatures are clear or standard for the test generator to work with, or that the main script is importable.\n--- End Test Code Generation Feedback ---\n"
                version_data["final_feedback_summary_for_developer"] = current_attempt_feedback
                version_data["test_results"] = "Test Generation Failed"
                if attempt == MAX_FIX_ATTEMPTS_PER_VERSION - 1:
                    logger.error(
                        f"Max fix attempts reached for version {version_id} due to test generation failure.")
                continue

            test_code = strip_code_block_markers(raw_test_code)
            if not test_code.strip():
                logger.warning(
                    f"Generated Python test code for {version_id} was empty after stripping.")
                version_data["log"].append(
                    "A-4: Generated Python test code was empty after stripping.")
                current_attempt_feedback += "\n--- Test Code Generation (A-4) Feedback ---\nGenerated Python test code was empty after stripping. Ensure test generation produces valid Python test code.\n--- End Test Code Generation Feedback ---\n"
                version_data["final_feedback_summary_for_developer"] = current_attempt_feedback
                version_data["test_results"] = "Test Generation Failed (Empty)"
                if attempt == MAX_FIX_ATTEMPTS_PER_VERSION - 1:
                    logger.error(
                        f"Max fix attempts reached for version {version_id} due to empty test generation.")
                continue

            try:
                with open(current_test_py_file_path, "w", encoding="utf-8") as f:
                    f.write(test_code)
                logger.info(
                    f"A-4: Python test code generated and saved to {current_test_py_file_path}")
            except IOError as e:
                logger.error(
                    f"Failed to save generated test code to {current_test_py_file_path}: {e}")
                version_data["log"].append(
                    f"A-4: Failed to save test code: {e}")
                version_data["test_results"] = "Test File Save Failed"
                pass  # Allow to continue if saving test file fails, but log it

            if os.path.exists(current_test_py_file_path):
                logger.info(
                    f"A-5: Checking Python Syntax for Test Code ({current_test_py_file_path})...")
                test_syntax_stdout, test_syntax_stderr, test_syntax_retcode = check_python_syntax(
                    current_test_py_file_path)
                version_data["log"].append(
                    f"A-5: Test code syntax check: retcode={test_syntax_retcode}, stdout='{test_syntax_stdout[:100]}...', stderr='{test_syntax_stderr[:100]}...'")

                if test_syntax_retcode != 0:
                    logger.warning(
                        f"Python syntax check failed for test code {current_test_py_file_name}.")
                    version_data["test_results"] = "Test Syntax Failed"
                    error_analysis = generate_llm_response(
                        ALL_PROMPTS["error_interpreter_prompt"],
                        agent_safety_settings=DEFAULT_SAFETY_SETTINGS,
                        execution_context=f"Checking Python syntax for test script {current_test_py_file_name}",
                        stdout=test_syntax_stdout, stderr=test_syntax_stderr, return_code=test_syntax_retcode
                    ) or "Error interpreter failed for test syntax check."
                    current_attempt_feedback += f"\n--- Test Code Syntax Check (A-5) Feedback ---\n{error_analysis}\n--- End Test Syntax Check Feedback ---\n"
                    version_data["final_feedback_summary_for_developer"] = current_attempt_feedback
                    if attempt == MAX_FIX_ATTEMPTS_PER_VERSION - 1:
                        logger.error(
                            f"Max fix attempts reached for version {version_id} due to test code syntax errors.")
                    continue
                logger.info(f"Python syntax OK for test code {version_id}!")
                current_attempt_feedback += "\n--- Test Code Syntax Check (A-5) Feedback ---\nTest code syntax is OK.\n--- End Test Syntax Check Feedback ---\n"

                logger.info(
                    f"A-5: Running Python Test Suite ({current_test_py_file_path})...")
                test_run_stdout, test_run_stderr, test_run_retcode = run_python_tests(
                    current_test_py_file_path, current_py_file_path)
                version_data["log"].append(
                    f"A-5: Test execution: retcode={test_run_retcode}, stdout='{test_run_stdout[:200]}...', stderr='{test_run_stderr[:200]}...'")
                # Update test_results in version_data for A-X in next iteration
                if test_run_retcode == 0:
                    # Or a summary of passed/failed
                    version_data["test_results"] = "All tests passed"
                else:
                    version_data["test_results"] = f"Failed (retcode {test_run_retcode})"

                if test_run_retcode != 0:
                    logger.warning(
                        f"Python tests failed or test script crashed for {version_id}.")
                    # version_data["test_results"] already updated
                    error_analysis = generate_llm_response(
                        ALL_PROMPTS["error_interpreter_prompt"],
                        agent_safety_settings=DEFAULT_SAFETY_SETTINGS,
                        execution_context=f"Running Python test suite {current_test_py_file_name}",
                        stdout=test_run_stdout, stderr=test_run_stderr, return_code=test_run_retcode
                    ) or "Error interpreter failed for test run."
                    current_attempt_feedback += f"\n--- Test Execution (A-5) Feedback ---\n{error_analysis}\n--- End Test Execution Feedback ---\n"
                    current_attempt_feedback += f"\nTest STDOUT:\n{test_run_stdout}\nTest STDERR:\n{test_run_stderr}\n"
                    version_data["final_feedback_summary_for_developer"] = current_attempt_feedback
                    if attempt == MAX_FIX_ATTEMPTS_PER_VERSION - 1:
                        logger.error(
                            f"Max fix attempts reached for version {version_id} due to test failures.")
                    continue

                logger.info(
                    f"All Python tests passed for {version_id} (or no tests found by runner)!")
                # version_data["test_results"] already updated
                current_attempt_feedback += "\n--- Test Execution (A-5) Feedback ---\nAll tests passed or ran successfully.\n--- End Test Execution Feedback ---\n"
            else:
                logger.warning(
                    f"Skipping test execution for {version_id} as test file was not available or generation failed.")
                version_data["test_results"] = "Skipped (No Test File/Gen Fail)"
                current_attempt_feedback += "\n--- Test Execution (A-5) Feedback ---\nTests skipped as test file was not available or generation failed.\n--- End Test Execution Feedback ---\n"

            logger.info(
                f"A-5: Running Main Python Ray Tracer Script ({current_py_file_path})...")
            main_run_stdout, main_run_stderr, main_run_retcode, main_exec_time = run_python_script(
                current_py_file_path)
            version_data["log"].append(
                f"A-5: Main script exec: retcode={main_run_retcode}, time={main_exec_time:.2f}s, stdout='{main_run_stdout[:200]}...', stderr='{main_run_stderr[:200]}...'")
            version_data["execution_time_seconds"] = main_exec_time

            if main_run_retcode != 0:
                logger.warning(
                    f"Main Python script failed or crashed for {version_id}.")
                version_data["main_exec_status"] = "Runtime Error"
                error_analysis = generate_llm_response(
                    ALL_PROMPTS["error_interpreter_prompt"],
                    agent_safety_settings=DEFAULT_SAFETY_SETTINGS,
                    execution_context=f"Running main Python script {current_py_file_name}",
                    stdout=main_run_stdout, stderr=main_run_stderr, return_code=main_run_retcode
                ) or "Error interpreter failed for main script execution."
                current_attempt_feedback += f"\n--- Main Script Run (A-5) Feedback ---\n{error_analysis}\n--- End Main Script Run Feedback ---\n"
                current_attempt_feedback += f"\nMain Script STDOUT:\n{main_run_stdout}\nMain Script STDERR:\n{main_run_stderr}\n"
                version_data["final_feedback_summary_for_developer"] = current_attempt_feedback
                if attempt == MAX_FIX_ATTEMPTS_PER_VERSION - 1:
                    logger.error(
                        f"Max fix attempts reached for version {version_id} due to main script execution failure.")
                continue

            logger.info(
                f"Main Python script for {version_id} ran successfully (Time: {main_exec_time:.2f}s)!")
            version_data["main_exec_status"] = "Success"
            current_attempt_feedback += "\n--- Main Script Run (A-5) Feedback ---\nMain Python script ran successfully.\n--- End Main Script Run Feedback ---\n"

            # If all checks so far passed, this iteration was successful for this version.
            # The mathematician review is done *after* this loop succeeds.
            # Updated summary
            version_data["final_feedback_summary_for_developer"] = "All checks passed in this iteration. See Deep Reviewer for potential refinements."
            logger.info(
                f"Version {version_id} (Python) successfully processed iteration {attempt + 1}.")

            # If main script ran successfully, and tests passed, this is a good candidate.
            if version_data["main_exec_status"] == "Success" and \
               ("All tests passed" in version_data["test_results"] or "Skipped" in version_data["test_results"]):  # Allow skipping tests if gen failed
                logger.info(
                    f"Version {version_id} passed main execution and tests (or tests skipped). Proceeding to Mathematician review and completion for this version.")
                break  # Break from the fix attempts loop for this version

        else:  # Else for the 'for attempt' loop (inner fix loop) - executed if loop finishes without break
            logger.error(
                f"Version {version_id} (Python) FAILED to complete a successful iteration after {MAX_FIX_ATTEMPTS_PER_VERSION} attempts.")
            version_data["log"].append(
                f"Version {version_id} failed all {MAX_FIX_ATTEMPTS_PER_VERSION} fix attempts to pass all checks.")
            # final_feedback_summary_for_developer is already set from the last failed attempt
            if version_data["syntax_check_status"] == "Not Attempted":
                version_data["syntax_check_status"] = "Failed before attempt"
            if version_data["main_exec_status"] == "Not Run" and \
               version_data["syntax_check_status"] == "Success":
                version_data["main_exec_status"] = "Not Run (previous step failed)"

        # --- A-7: Mathematician Agent (Python) ---
        # This runs if the above loop broke (meaning success) OR if it exhausted attempts but we still want a review
        if current_code_content:  # Only if there's code to review
            logger.info(
                f"A-7: Reviewing math/physics in Python code for {version_id}...")
            mathematician_review_raw = generate_llm_response(
                ALL_PROMPTS["mathematician_agent_prompt"],
                agent_safety_settings=DEFAULT_SAFETY_SETTINGS,
                specifications=specifications,
                python_code_content=current_code_content
            )
            mathematician_review = strip_code_block_markers(
                mathematician_review_raw) if mathematician_review_raw else "No feedback from Mathematician Agent."
            version_data["mathematician_review"] = mathematician_review
            version_data["log"].append(
                "A-7: Mathematician review completed.")
            try:
                with open(os.path.join(version_output_dir, f"mathematician_review_{version_id}.txt"), "w", encoding="utf-8") as f:
                    f.write(mathematician_review)
            except IOError as e:
                logger.warning(
                    f"Could not save mathematician review for {version_id}: {e}")
        else:
            logger.warning(
                f"Skipping Mathematician review for {version_id} as no valid code content was produced.")
            version_data["mathematician_review"] = "Skipped - No valid code."

        all_versions_data.append(version_data)
        try:
            # Make a copy for saving to avoid trying to serialize potentially unserializable live objects if any creep in
            saveable_version_data = version_data.copy()
            if 'python_code_content' in saveable_version_data and saveable_version_data['python_code_content'] is not None:
                # Optionally, truncate code content in log if too long, or just store a flag
                saveable_version_data['python_code_content_summary'] = f"Exists, length {len(saveable_version_data['python_code_content'])}"
                # Don't save full code in this summary log again
                del saveable_version_data['python_code_content']

            with open(os.path.join(version_output_dir, f"version_{version_id}_log_and_data.json"), "w", encoding="utf-8") as f:
                json.dump(saveable_version_data, f, indent=2)
            logger.info(
                f"Saved detailed log and data for version {version_id}.")
        except TypeError as e:
            logger.error(
                f"Failed to serialize version_data for {version_id} to JSON: {e}. Some content might not be serializable.")
            logger.debug(
                f"Problematic data for version {version_id}: {version_data}")
        except IOError as e:
            logger.error(
                f"Failed to save log/data JSON for {version_id}: {e}")

    # --- Filter for candidate versions for selection ---
    # (Selector logic remains largely the same, but now it can use "deep_reviewer_feedback_log" if needed)
    candidate_versions_for_selection = [
        v for v in all_versions_data if v.get("python_code_content") and
        v.get("syntax_check_status") == "Success" and
        v.get("main_exec_status") == "Success"
    ]
    if not candidate_versions_for_selection:
        candidate_versions_for_selection = [
            v for v in all_versions_data if v.get("python_code_content") and
            v.get("syntax_check_status") == "Success"
        ]
    if not candidate_versions_for_selection:
        candidate_versions_for_selection = [
            v for v in all_versions_data if v.get("python_code_content")
        ]

    best_version_id = None
    best_version_code_content = None
    final_commented_code_content = None

    if not candidate_versions_for_selection:
        logger.critical(
            "No Python versions were successfully processed or generated usable code. Cannot proceed with selection.")
    else:
        logger.info(
            f"--- Step A-6: Selecting Best Python Version from {len(candidate_versions_for_selection)} candidates ---")
        selector_input_data = []
        for v_data in candidate_versions_for_selection:
            slim_v_data = v_data.copy()
            if slim_v_data.get("python_code_content") and \
               len(slim_v_data["python_code_content"]) > 5000:
                slim_v_data["python_code_content"] = slim_v_data["python_code_content"][:5000] + \
                    "\n... (code truncated for selector prompt)"
            if "log" in slim_v_data:
                del slim_v_data["log"]
            # Add summary of deep reviewer feedback if available
            if slim_v_data.get("deep_reviewer_feedback_log"):
                # Take the last deep review for this version as most relevant
                last_deep_review = slim_v_data["deep_reviewer_feedback_log"][-1]
                # Extract the actual review part, not the "attempt_X" key
                actual_review_content = next(iter(
                    last_deep_review.values())) if isinstance(last_deep_review, dict) else last_deep_review
                slim_v_data["deep_reviewer_feedback"] = actual_review_content
            else:
                slim_v_data["deep_reviewer_feedback"] = "Not available or not run."
            if "deep_reviewer_feedback_log" in slim_v_data:
                del slim_v_data["deep_reviewer_feedback_log"]

            selector_input_data.append(slim_v_data)

        selector_response_raw = generate_llm_response(
            ALL_PROMPTS["multi_version_selector_agent_prompt"],
            agent_safety_settings=DEFAULT_SAFETY_SETTINGS,
            num_versions=len(selector_input_data),
            versions_data_json=json.dumps(selector_input_data, indent=2),
            specifications=specifications
        )
        selection_output_path = os.path.join(
            OUTPUT_BASE_DIR, "a6_python_version_selection_details.json")
        if selector_response_raw:
            selector_response_raw_cleaned = strip_code_block_markers(
                selector_response_raw)
            try:
                selection_results = json.loads(selector_response_raw_cleaned)
                try:
                    with open(selection_output_path, "w", encoding="utf-8") as f:
                        json.dump(selection_results, f, indent=2)
                    logger.info(
                        f"Python version selection details saved to {selection_output_path}")
                except IOError as e:
                    logger.warning(f"Could not save selection details JSON: {e}")

                best_version_id = selection_results.get("best_version_id")
                logger.info(
                    f"A-6 Selector chose Python version: {best_version_id}")
                logger.info(
                    f"Selector Reasoning: {selection_results.get('overall_reasoning')}")

                # Check for None string and actual None
                if best_version_id and best_version_id != "None" and best_version_id is not None:
                    original_best_v_data = next(
                        (v_full for v_full in all_versions_data if v_full["version_id"] == best_version_id), None)
                    if original_best_v_data and original_best_v_data.get("python_code_content"):
                        best_version_code_content = original_best_v_data["python_code_content"]
                        best_version_dir_source = os.path.join(
                            OUTPUT_BASE_DIR, f"version_{best_version_id}")
                        final_best_dir_dest = os.path.join(
                            OUTPUT_BASE_DIR, "final_best_python_version")

                        if os.path.exists(best_version_dir_source):
                            if os.path.exists(final_best_dir_dest):
                                shutil.rmtree(final_best_dir_dest)
                            shutil.copytree(
                                best_version_dir_source, final_best_dir_dest)
                            logger.info(
                                f"Files for best Python version ({best_version_id}) copied to {final_best_dir_dest}")
                        else:
                            logger.warning(
                                f"Source directory for best version {best_version_id} not found: {best_version_dir_source}")
                            if best_version_code_content:
                                os.makedirs(final_best_dir_dest, exist_ok=True)
                                try:
                                    with open(os.path.join(final_best_dir_dest, f"bbh_visualizer_{best_version_id}.py"), "w", encoding="utf-8") as f_bf:
                                        f_bf.write(best_version_code_content)
                                    logger.info(
                                        f"Saved best version code directly to {final_best_dir_dest}")
                                except IOError as e_bf:
                                    logger.error(
                                        f"Could not save best version code directly: {e_bf}")
                                    best_version_code_content = None
                                    best_version_id = None
                    else:
                        logger.error(
                            f"Best Python version ID '{best_version_id}' selected, but its full content or original data entry not found.")
                        best_version_id = None
                        best_version_code_content = None
                else:
                    logger.warning(
                        "A-6 Selector did not choose a best Python version or returned 'None'.")
                    best_version_id = None
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse A-6 Python selector response as JSON: {e}")
                logger.debug(
                    f"Raw response from A-6 (selector):\n{selector_response_raw_cleaned}")
                try:
                    with open(selection_output_path.replace(".json", "_error.txt"), "w", encoding="utf-8") as f_err:
                        f_err.write(selector_response_raw_cleaned)
                except IOError:
                    pass
                best_version_id = None
        else:
            logger.error(
                "A-6 Python Selector agent failed to provide a response.")
            best_version_id = None

    # Commenting and Documentation steps (A-8, A-9) remain largely the same
    if best_version_code_content and best_version_id:
        logger.info(
            f"--- Step A-8: Commenting Best Python Code Version ({best_version_id}) ---")
        commented_code_raw = generate_llm_response(
            ALL_PROMPTS["code_commenter_agent_prompt"],
            agent_safety_settings=CODE_GENERATION_SAFETY_SETTINGS,
            python_code_content=best_version_code_content
        )
        if commented_code_raw:
            final_commented_code_content = strip_code_block_markers(
                commented_code_raw)
            if not final_commented_code_content.strip():
                logger.warning(
                    "Code commenter produced empty output after stripping. Using un-commented version.")
                final_commented_code_content = best_version_code_content

            commented_code_file_path_specific = os.path.join(
                OUTPUT_BASE_DIR, "final_best_python_version", f"bbh_visualization_{best_version_id}_commented.py")
            generic_final_code_path = os.path.join(
                OUTPUT_BASE_DIR, "bbh_visualization_final_commented.py")

            os.makedirs(os.path.dirname(
                commented_code_file_path_specific), exist_ok=True)
            try:
                with open(commented_code_file_path_specific, "w", encoding="utf-8") as f:
                    f.write(final_commented_code_content)
                shutil.copyfile(commented_code_file_path_specific,
                                generic_final_code_path)
                logger.info(
                    f"Commented Python code saved to {commented_code_file_path_specific} and {generic_final_code_path}")
            except IOError as e:
                logger.error(
                    f"Failed to save commented code: {e}. Using un-commented version for documentation.")
                final_commented_code_content = best_version_code_content
        else:
            logger.warning(
                "A-8 Python Code Commenter failed. Using un-commented best version for documentation.")
            final_commented_code_content = best_version_code_content

        if final_commented_code_content:
            logger.info(
                f"--- Step A-9: Generating Documentation for Best Python Version ({best_version_id}) ---")
            documentation_raw = generate_llm_response(
                ALL_PROMPTS["documentation_agent_prompt"],
                agent_safety_settings=DEFAULT_SAFETY_SETTINGS,
                commented_python_code_content=final_commented_code_content,
                specifications=specifications
            )
            if documentation_raw:
                documentation_content = strip_code_block_markers(
                    documentation_raw)
                if not documentation_content.strip():
                    logger.warning(
                        "Documentation agent produced empty output after stripping.")
                else:
                    doc_file_path = os.path.join(
                        OUTPUT_BASE_DIR, "USER_MANUAL_PYTHON.md")
                    final_best_dir_doc_path = os.path.join(
                        OUTPUT_BASE_DIR, "final_best_python_version", "USER_MANUAL_PYTHON.md")
                    try:
                        with open(doc_file_path, "w", encoding="utf-8") as f:
                            f.write(documentation_content)
                        logger.info(
                            f"Python documentation saved to {doc_file_path}")
                        if os.path.exists(os.path.dirname(final_best_dir_doc_path)):
                            shutil.copyfile(
                                doc_file_path, final_best_dir_doc_path)
                    except IOError as e:
                        logger.error(f"Failed to save documentation: {e}")
            else:
                logger.warning(
                    "A-9 Python Documentation Agent failed to produce output.")
    elif candidate_versions_for_selection:
        logger.warning(
            "No single best Python version was selected or processed by A-6. Skipping A-8 (Commenting) and A-9 (Documentation).")
    else:
        logger.warning(
            "No Python code versions were suitable for final processing (A-8, A-9).")

    all_versions_summary_path = os.path.join(
        OUTPUT_BASE_DIR, "all_python_versions_summary_data.json")
    try:
        summary_data_for_json = []
        for v_data_orig in all_versions_data:
            item = v_data_orig.copy()
            if item.get("python_code_content"):
                item["python_code_content_exists"] = True
                del item["python_code_content"]
            else:
                item["python_code_content_exists"] = False
            if "log" in item:
                item["log_summary"] = f"{len(item['log'])} entries, first: {item['log'][0] if item['log'] else 'N/A'}"
                del item["log"]
            summary_data_for_json.append(item)

        with open(all_versions_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data_for_json, f, indent=2)
        logger.info(
            f"Summary of all generated Python versions saved to {all_versions_summary_path}")
    except TypeError as e:
        logger.error(
            f"Failed to serialize all_versions_data for summary JSON: {e}")
    except IOError as e:
        logger.error(
            f"Failed to save all_versions_summary_data JSON: {e}")

    logger.info("--- Agentic Python Software Development Process Completed ---")
    if best_version_id and final_commented_code_content:
        logger.info(
            f"Successfully selected and finalized Python version: {best_version_id}")
        logger.info(
            f"Final commented Python code: {os.path.join(OUTPUT_BASE_DIR, 'bbh_visualization_final_commented.py')}")
        logger.info(
            f"User Manual (Python): {os.path.join(OUTPUT_BASE_DIR, 'USER_MANUAL_PYTHON.md')}")
    else:
        logger.warning(
            "Process finished, but a final 'best' Python version was not fully processed or selected.")
        logger.info(
            f"Please check the '{OUTPUT_BASE_DIR}' directory for all generated artifacts and logs.")
if __name__ == "__main__":
    main()
