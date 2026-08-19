# Main driver script

import argparse
import sys
from python_scripts.question1 import run_q1
from python_scripts.question2 import run_q2
from python_scripts.question3 import run_q3
from python_scripts.question4 import run_q4

QUESTION_RUNNERS = {
    1: run_q1,
    2: run_q2,
    3: run_q3,
    4: run_q4,
}

def main():
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