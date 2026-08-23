# Main driver script

import argparse
import os
import subprocess
import sys

def setup_environment():
    script_path = "bash_scripts/setup_env.sh"
    if os.path.exists(script_path):
        try:
            subprocess.run(["bash", script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing environment setup script: {e}")
            sys.exit(1)
    else:
        print(f"Error: {script_path} does not exist.")
        sys.exit(1)

    venv_python = os.path.abspath(".venv/bin/python")
    if os.path.exists(venv_python) and sys.executable != venv_python:
        print(f"Switching Python interpreter to virtual environment: {venv_python}\n")
        os.execv(venv_python, [venv_python] + sys.argv)

# Run setup / venv switch BEFORE importing anything that depends on
# packages from requirements.txt (e.g. sentence-transformers). If we're
# not already running inside .venv, this will re-exec into it.
setup_environment()

# Safe to import now: either .venv didn't exist yet and setup_env.sh just
# installed into the currently running interpreter's environment, or we
# already got re-exec'd into .venv/bin/python.
from python_scripts.question1 import run_q1
from python_scripts.question2 import run_q2
from python_scripts.question3 import run_q3
from python_scripts.question4 import run_q4

def main():
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