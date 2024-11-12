# cervical-cms

This repository provides the code for paper "A Deep Learning Framework for Predicting Prognostically Relevant Consensus Molecular Subtypes in HPV-Positive Cervical Squamous Cell Carcinoma from Routine Histology Images" https://doi.org/10.1101/2024.08.16.608264. 

Here is how to train TripletMIL for consensus molecular subtype (CMS) classification in cervical cancer. 


## Usage

### Running the Pipeline

The main entry point to run the training pipeline is `main_train.py`. The script supports various arguments that you can configure through the command line.

Below is an example command to run the training pipeline with custom settings:

```bash
python main_train.py --lr 3e-3 --epochs 20 --split_path ./splits/train_val_split.pkl --feature_dict_str "dataset1:/path1,dataset2:/path2"
```

### Argument Details

Below is a list of important arguments you can use when running `main_train.py`:

- **Experiment Arguments**:
  - **--split_path**: Path to the file containing the split of training and validation sets, should be a pickle file with keys of 'train' and 'val' and each value should be a list of patient ids.
  - **--feature_dict_str**: String specifying feature dictionaries in the format `dataset_key:path,dataset_key:path,...`. The path should be a dictionary containing pickle files, each file contains a list of feature vectors for patches extracted from a patient's slide.

Refer to `arguments.py` for more details on additional arguments you can use.

## Model Architecture

The MLP model is defined in `MLP_model.py` and includes customizable layers and activation functions as specified in the script. Adjustments to the model architecture can be made directly in `MLP_model.py` if needed.

## Utilities

- **Data Loading** (`data_utils.py`): Handles dataset preparation and preprocessing.
