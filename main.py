import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="SUR project 2025/2026"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", action="store_true", help="train mode")
    group.add_argument("-p", action="store_true", help="production mode")

    parser.add_argument("path/to/config", type=str, help="Path to config. Required!")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()