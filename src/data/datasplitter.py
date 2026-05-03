# author: Myron Kukhta (xkukht01)
import csv

from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from typing import Tuple, List, Dict, Any

from .dataconfig import DataConfig

class DataSplitter:

    def __init__(self, config_path:str):
        config_path = Path(config_path)
        self.config = DataConfig.model_validate_json(config_path.read_text())

    DATA_DIRS = [
        "/home/xkukht01/Dev/SUR/data/target_train",
        "/home/xkukht01/Dev/SUR/data/target_dev",
        "/home/xkukht01/Dev/SUR/data/non_target_train",
        "/home/xkukht01/Dev/SUR/data/non_target_dev",
    ]

    TARGET_ID = "m431"

    OUTPUT_DIR = "/home/xkukht01/Dev/SUR/data/sur_data"

    N_SPLITS = 3
    RANDOM_STATE = 42

    IMAGE_EXTS = {".png"}
    AUDIO_EXTS = {".wav"}


    def parse_filename(self, path: Path) -> Tuple[str, str]:
        parts = path.stem.split("_")

        if len(parts) < 2:
            raise ValueError(f"Bad filename format: {path.name}")

        person_id = parts[0]
        session_id = parts[1]
        group_id = f"{person_id}_{session_id}"

        return person_id, group_id


    def collect_files(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        image_samples = []
        audio_samples = []

        for data_dir in self.config.source_data_dirs:
            data_dir = Path(data_dir)

            if not data_dir.exists():
                print(f"Error! Directory does not exist: {data_dir}")
                continue

            for path in data_dir.rglob("*"):
                if not path.is_file():
                    continue

                ext = path.suffix.lower()

                if ext not in self.IMAGE_EXTS and ext not in self.AUDIO_EXTS:
                    continue

                person_id, group_id = self.parse_filename(path)
                target = 1 if person_id == self.config.trg_person_id else 0

                sample = {
                    "path": str(path.resolve()),
                    "target": target,
                    "group": group_id,
                    "stem": path.stem,
                }

                if ext in self.IMAGE_EXTS:
                    image_samples.append(sample)
                elif ext in self.AUDIO_EXTS:
                    audio_samples.append(sample)

        return image_samples, audio_samples


    def write_csv(self, path: Path, samples: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "target"])

            for sample in samples:
                writer.writerow([
                    sample["path"],
                    sample["target"],
                ])


    def create_group_splits(self, samples: List[Dict[str, Any]], output_subdir: Path) -> None:
        if not samples:
            print(f"Error! No samples for {output_subdir}")
            return

        seq_id = list(range(len(samples)))
        labels = [s["target"] for s in samples]
        groups = [s["group"] for s in samples]

        unique_groups = set(groups)

        if len(unique_groups) < self.N_SPLITS:
            print(f"Error! Not enough groups for {output_subdir}")
            print(f"groups={len(unique_groups)}, n_splits={self.N_SPLITS}")
            print()
            return

        if len(set(labels)) == 1:
            splitter = GroupKFold(n_splits=self.N_SPLITS)
            split_iter = splitter.split(seq_id, labels, groups)
        else:
            splitter = StratifiedGroupKFold(
                n_splits=self.N_SPLITS,
                shuffle=True,
                random_state=self.RANDOM_STATE,
            )
            split_iter = splitter.split(seq_id, labels, groups)

        for fold_id, (train_idx, dev_idx) in enumerate(split_iter):
            fold_dir = output_subdir / f"fold_{fold_id:02d}"

            train_samples = [samples[i] for i in train_idx]
            dev_samples = [samples[i] for i in dev_idx]

            self.write_csv(fold_dir / "train.csv", train_samples)
            self.write_csv(fold_dir / "dev.csv", dev_samples)

            print(f"{output_subdir.name} fold_{fold_id:02d}:")
            print(f"train={len(train_samples)}, dev={len(dev_samples)}")
            print(f"train_pos={sum(s['target'] for s in train_samples)}")
            print(f"dev_pos={sum(s['target'] for s in dev_samples)}")
            print()


    def create_total_csv(self, samples: List[Dict[str, Any]], output_path: Path) -> None:
        if not samples:
            print(f"Error! No samples for {output_path}")
            return

        self.write_csv(output_path, samples)

        print(f"Saved total csv: {output_path}")
        print(f"samples={len(samples)}")
        print(f"positive={sum(s['target'] for s in samples)}")
        print()


    def split_by_target(
        self, samples: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        target_samples = [s for s in samples if s["target"] == 1]
        nontarget_samples = [s for s in samples if s["target"] == 0]

        return target_samples, nontarget_samples

    def run(self):

        output_dir = Path(self.config.output_data_dir)

        image_samples, audio_samples = self.collect_files()

        print(f"Images found: {len(image_samples)}")
        print(f"Audio found:  {len(audio_samples)}")
        print()

        # create split with full dataset for both data training
        if self.config.with_folds:
            self.create_group_splits(image_samples, output_dir / "image" / "folds")
            self.create_group_splits(audio_samples, output_dir / "audio" / "folds")

        # split based on label
        self.create_total_csv(image_samples, output_dir / "image" / "total.csv")
        self.create_total_csv(audio_samples, output_dir / "audio" / "total.csv")

