# run_local_test.py
import argparse
from agent_core.dispatcher import handle_query

def main():
    parser = argparse.ArgumentParser(description="Run Palona AI Agent locally.")
    parser.add_argument("query", type=str, help="Text query to test")
    args = parser.parse_args()

    response = handle_query(text=args.query)
    print("\n🧠 Palona Response:\n", response)

if __name__ == "__main__":
    main()
