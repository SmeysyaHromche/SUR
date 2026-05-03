import argparse
from src import Train, Production

def parse_args():
    parser = argparse.ArgumentParser(
        description="SUR project 2025/2026"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", action="store_true", help="train mode")
    group.add_argument("-p", action="store_true", help="production mode")

    parser.add_argument("path_to_config", type=str, help="Path to config. Required!")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.t:
        train = Train(args.path_to_config)
        train.train()
    elif args.p:
        production = Production(args.path_to_config)
        production.run()
    else:
        pass


if __name__ == "__main__":
    main()