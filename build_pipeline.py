# Main driver script

import argparse
import os
import subprocess
import sys
from python_scripts.question1 import run_q1
from python_scripts.question2 import run_q2
from python_scripts.question3 import run_q3
from python_scripts.question4 import run_q4

# def setup_environment():
#     script_path = "bash_scripts/setup_env.sh"
#     if os.path.exists(script_path):
#         try:
#             subprocess.run(["bash", script_path], check=True)
#         except subprocess.CalledProcessError as e:
#             print(f"Error executing environment setup script: {e}")
#             sys.exit(1)
#     else:
#         print(f"Error: {script_path} does not exist.")
#         sys.exit(1)

#     venv_python = os.path.abspath(".venv/bin/python")
#     if os.path.exists(venv_python) and sys.executable != venv_python:
#         print(f"Switching Python interpreter to virtual environment: {venv_python}\n")
#         os.execv(venv_python, [venv_python] + sys.argv)

def main():
    # setup_environment()

    QUESTION_RUNNERS = {
        1: run_q1,
        2: run_q2,
        3: run_q3,
        4: run_q4,
    }

    parser = argparse.ArgumentParser(description="Run the pipeline for the assignment.")
    parser.add_argument(
        "--question",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Specify which question to run (1, 2, 3, or 4). If not given, all four run in order.",
    )
    args = parser.parse_args()

    if args.question is None:
        for q in sorted(QUESTION_RUNNERS):
            print(f"\nRunning Question {q}\n")
            QUESTION_RUNNERS[q]()
    else:
        QUESTION_RUNNERS[args.question]()

if __name__ == "__main__":
    main()