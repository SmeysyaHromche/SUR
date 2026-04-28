# author: Myron Kukhta (xkukht01)
# info: splitt original dataset by PERSON x SESSION pair for cross validation
import csv
import shutil

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, List

DATA_DIRS = [
    "/home/xkukht01/Dev/SUR/data/target_train",
    "/home/xkukht01/Dev/SUR/data/target_dev",
    "/home/xkukht01/Dev/SUR/data/non_target_train",
    "/home/xkukht01/Dev/SUR/data/non_target_dev",
]

TARGET_ID = "m431"

OUTPUT_DIR = "/home/xkukht01/Dev/SUR/data/sur_data"

DEV_SIZE = 0.25
N_SPLITS = 5
RANDOM_STATE = 42


IMAGE_EXTS = {".png"}
AUDIO_EXTS = {".wav"}

def clear_output_dir(path: Path) -> None:
    '''
    Clean target directory

    Args:
        path: target directory
    
    Returns: None
    '''
    if not path.exists():
        return

    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    print(f"Cleared directory: {path}")


def parse_filename(path: Path) -> Tuple[str, str]:
    '''
    Parse filename

    Args:
        path: path to file
    
    Returns:
        pair in format ( PERSON_ID, SESSION_ID) 
    '''
    parts = path.stem.split("_")

    if len(parts) < 2:
        raise ValueError(f"Bad filename format: {path.name}")

    person_id = parts[0]
    session_id = parts[1]
    group_id = f"{person_id}_{session_id}"

    return person_id, group_id


def collect_files() -> Tuple[List[str], List[str]]:
    '''
    Collect and split files based on data type
    
    Args:
        None
    
    Returns:
        pair of lists [IMAGE_DATA, AUDIO_DATA]
    '''
    image_samples = []
    audio_samples = []

    for data_dir in DATA_DIRS:
        data_dir = Path(data_dir)

        if not data_dir.exists():
            print(f"Error! Directory does not exist: {data_dir}")
            continue

        for path in data_dir.rglob("*"):
            if not path.is_file():
                continue

            ext = path.suffix.lower()

            if ext not in IMAGE_EXTS and ext not in AUDIO_EXTS:
                continue

            person_id, group_id = parse_filename(path)
            target = 1 if person_id == TARGET_ID else 0

            sample = {
                "path": str(path.resolve()),
                "target": target,
                "group": group_id,
                "stem": path.stem,
            }

            if ext in IMAGE_EXTS:
                image_samples.append(sample)
            elif ext in AUDIO_EXTS:
                audio_samples.append(sample)

    return image_samples, audio_samples


def write_csv(path: Path, samples:List[str]) -> None:
    '''
    Store metadata of samples in csv file in format:
        absolute/path/to/file, label
    
    Where label is 0 when data from target person

    Args:
        path: path to target csv file
        sample: list of path to data samples

    Returns:
        None
    '''
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "target"])

        for sample in samples:
            writer.writerow([
                sample["path"],
                sample["target"],
            ])


def create_group_splits(samples: List[str], output_subdir: Path) -> None:
    '''
    Split samples of data to diff groups (in each group on div, train files) for 
    cross validation.

    Args:
        samples: list of paths to data
        output_subdir: path to group data location
    '''
    if not samples:
        print(f"Error! No samples for {output_subdir}")
        return

    seq_id = list(range(len(samples)))
    labels = [s["target"] for s in samples]
    groups = [s["group"] for s in samples]

    splitter = GroupShuffleSplit(
        n_splits=N_SPLITS,
        test_size=DEV_SIZE,
        random_state=RANDOM_STATE,
    )

    for fold_id, (train_idx, dev_idx) in enumerate(splitter.split(seq_id, labels, groups)):
        fold_dir = output_subdir / f"fold_{fold_id:02d}"

        train_samples = [samples[i] for i in train_idx]
        dev_samples = [samples[i] for i in dev_idx]

        write_csv(fold_dir / "train.csv", train_samples)
        write_csv(fold_dir / "dev.csv", dev_samples)

        print(
            f"{output_subdir.name} fold_{fold_id:02d}: "
            f"train={len(train_samples)}, dev={len(dev_samples)}"
        )

if __name__ == "__main__":
    output_dir = Path(OUTPUT_DIR)
    
    clear_output_dir(output_dir)

    image_samples, audio_samples = collect_files()

    print(f"Images found: {len(image_samples)}")
    print(f"Audio found:  {len(audio_samples)}")

    create_group_splits(image_samples, output_dir / "image")
    create_group_splits(audio_samples, output_dir / "audio")
