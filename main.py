import argparse
from src import Train, Production, DataSplitter

def parse_args():
    parser = argparse.ArgumentParser(
        description="SUR project 2025/2026"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", action="store_true", help="train mode")
    group.add_argument("-p", action="store_true", help="production mode")
    group.add_argument("-d", action="store_true", help="split your data to project format")

    parser.add_argument("path_to_config", type=str, help="Path to config. Required!")

    return parser.parse_args()


def main():
    args = parse_args()
    path_to_config = args.path_to_config
    if args.t:
        train = Train(path_to_config)
        train.train()
    elif args.p:
        production = Production(path_to_config)
        production.run()
    elif args.d:
        splitter = DataSplitter(path_to_config)
        splitter.run()
    else:
        pass


if __name__ == "__main__":
    main()