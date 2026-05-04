# SUR 2025/2026

Binary classification project for image and audio data using classical machine learning approaches.
The system supports:

* image classification,
* audio classification,
* metadata preparation,
* cross-validation,
* model training,
* production inference pipelines.

---

# Implementation

The project was implemented in Python using an object-oriented approach and standard machine learning frameworks, including `scikit-learn`, as well as libraries for mathematical computations (`NumPy`, `Pandas`), image processing (`OpenCV`), and audio processing (`Librosa`). Additional utilities and helper modules were used to organize the training and production pipelines.

## Project Structure

```text
project/
|-- configs/            # example configuration files
|-- src/                # project source code
|   |-- data/           # data preparation and processing
|   |-- model/          # model implementations and components
|   |-- production/     # production inference pipelines
|   |-- train/          # training pipelines
|-- models_output/
|   |-- audio_out.txt   # audio evaluation results
|   |-- image_out.txt   # image evaluation results
|-- train_out/          # trained models, logs, and .pkl files
|-- documentation.pdf
|-- main.py             # project entry point
|-- README.md           # project documentation
|-- requirements.txt    # project dependencies
```

In addition to the source code, the repository also contains:

* example configuration files (`configs/`),
* training outputs (`train_out/`),
* evaluation outputs (`models_output/`).

---

# Installation and Startup

This project is primarily designed for Linux environments.

## Create Python Virtual Environment

```bash
python3 -m venv venv
```

## Activate Environment

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

After completing these steps, the project will be ready for use.

The system supports several operating modes:

* data preparation,
* training,
* production inference.

For detailed CLI information:

```bash
python3 main.py --help
```

---

# Data Preparation Mode

This mode generates a metadata map of the dataset and currently supports only `.wav` and `.png` files.

The program automatically:

* separates data by modality (image/audio),
* generates `.csv` files with detected samples,
* assigns labels when possible.

When extended mode is enabled, the system additionally creates:

* 3-fold cross-validation splits,
* separate CSV files for each fold.

## Launch

```bash
python3 main.py -d [PATH_TO_CONFIG]
```

The command expects a configuration file in JSON format.

Example configuration files can be found in the `configs/` directory.

## Configuration Parameters

| Parameter          | Type           | Description                                           |
| ------------------ | -------------- | ----------------------------------------------------- |
| `source_data_dirs` | `list[string]` | Paths to dataset directories.                         |
| `output_data_dir`  | `string`       | Output directory for generated metadata.              |
| `trg_person_id`    | `string`       | Target person ID used for labeling (example: `m431`). |
| `with_folds`       | `boolean`      | Enables generation of cross-validation folds.         |

---

# Training Mode

This mode assumes that the metadata preparation step has already been completed.

The system:

* trains the selected model,
* performs evaluation,
* outputs metrics to the command line.

Supported functionality:

* cross-validation,
* full dataset training,
* model weight saving,
* log exporting.

## Launch

```bash
python3 main.py -t [PATH_TO_CONFIG]
```

The command expects a JSON configuration file.

## Configuration Parameters

| Parameter                | Type      | Description                                 |
| ------------------------ | --------- | ------------------------------------------- |
| `model`                  | `string`  | Classifier type: `"audio"` or `"image"`.    |
| `total_data_csv_path`    | `string`  | The path to the `CSV` file with the total samples of actual type of data generated in `-d` mode.             |
| `folds_dir_path`         | `string`  | The path to the directory with all folds separations generated in `-d` mode.         |
| `out`                    | `string`  | The path to the directory where will be stored model `.pkl` and logs.       |
| `model_name`             | `string`  | ID for actual model running, only for logs and better separability.          |
| `is_full_train`          | `boolean` | Enables final training on the full dataset. |
| `is_save_validation_log` | `boolean` | Enables saving cross-validation logs.       |

---

# Production Mode

This mode performs inference using trained models.

The metadata preparation step must already be completed before execution.

The system:

* loads trained models,
* classifies input samples,
* generates output reports compatible with the expected evaluation format.

## Launch

```bash
python3 main.py -p [PATH_TO_CONFIG]
```

The command expects a JSON configuration file.

Only the paths to the required model weights must be specified depending on the selected classifier.

## Configuration Parameters

| Parameter                     | Type     | Description                                        |
| ----------------------------- | -------- | -------------------------------------------------- |
| `model`                       | `string` | Classifier type: `"audio"` or `"image"`.           |
| `data_path`                   | `string` | The path to the `CSV` file with the total samples of actual type of data generated in `-d` mode. |
| `audio_target_model_path`     | `string` | The path to the `.pkl` file of weights of GMM for target audio.             |
| `audio_non_target_model_path` | `string` | The path to the `.pkl` file of weights of GMM for nontarget audio.         |
| `image_model_path`            | `string` |  The path to the `.pkl` file of weights of image classifier.           |
| `classification_out_path`     | `string` | Output path for classification report `.txt` file. |
