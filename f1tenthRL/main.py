from src.f1tenth_environment import F1TenthEnvironment
from src.utils import parse_arguments
from src.train import train_agent
from src.evaluate import evaluate_agent

def main():
    args = parse_arguments()
    if args.mode == "train":
        train_agent()
    elif args.mode == "evaluate":
            print("Initializing evaluation pipeline...")
            evaluate_agent()

if __name__ == "__main__":
    main()