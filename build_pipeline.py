# Main driver script

import argparse
import sys
from python_scripts.question1 import run_q1
from python_scripts.question2 import run_q2
from python_scripts.question3 import run_q3

def main():
    parser = argparse.ArgumentParser(description="Run the pipeline for the assignment.")
    parser.add_argument("--question", type=int, choices=[1, 2, 3], required=True, help="Specify which question to run (1 or 2).")
    args = parser.parse_args()

    if args.question == 1:
        run_q1()
    elif args.question == 2:
        run_q2()
    elif args.question == 3:
        run_q3()
    else:
        print("Invalid question number. Please specify either 1, 2, or 3.")
        sys.exit(1)

if __name__ == "__main__":
    main()