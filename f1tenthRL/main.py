from code.f1tenth_environment import F1TenthEnvironment
from code.utils import parse_arguments
from code.train import train_agent
from code.evaluate import evaluate_agent

def main():
    args = parse_arguments()
    if args.mode == "train":
        train_agent()
    elif args.mode == "evaluate":
            print("Initializing evaluation pipeline...")
            evaluate_agent()

if __name__ == "__main__":
    main()