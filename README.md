# ML pipeline for automatic ARDS diagnosis  

This repository contains the Jupyter notebooks and Python modules for the paper "Open-source machine learning pipeline automatically flags instances of acute respiratory distress syndrome from electronic health records" by Morales et al. (2024). The paper is available at <https://doi.org/10.1101/2024.05.21.24307715>.  

If you use this work, please cite the paper.

## Getting started

### Prerequisites  

You need to have Python 3.10.12 installed on your system. You can download Python from the official website. Alternatively, you can use the Anaconda distribution of Python, which includes pip and other useful tools.  

### Installation

Using conda environments:

1. Using your Anaconda Prompt, clone the repository to your local machine.
2. Navigate to the project directory.  
3. Create a fresh environment using the following command: `conda env create -f environment.yml`. See the [conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more information.  
4. Activate your newly-created environment with `conda activate [name_of_env]`

Using pyenv or venv environments:  

1. Using git bash or your preferred terminal, clone the repository to your local machine.  
2. Navigate to the project directory.
3. Create a fresh environment (Python==3.10.12 recommended, but not required) using pyenv or venv.
4. Activate your newly-created environment, and run the following command to install the required packages: `pip install -r requirements.txt`

## Notebook description  

Notebooks follow a numbering scheme indicating at which data processing step the notebook inserts itself into. For example, notebooks with the number `03` use the most "mature" or processed versions of their datasets. Furthermore, the first part of the name indicates the purpose of the notebook, while the last part indicates the cohort of the notebook. The following is a brief description of each notebook.  

- `01_preprocess_mimiciii.ipynb`:  
  - **Description:** Preprocesses MIMIC-III dataset (standardizing column names, removing duplicates, etc.).  
  - **Runtime:** 50 seconds on Windows 11 machine (CPU: Intel Core i7-1165G7, RAM: 16GB). It depends on the user, since it requires manual input to proceed, but most of the time is spent writing the csv files to disk.
  - **Expected outputs:** It will write csv files into the `Preprocessed_Data/MIMIC_III/labeled_subset` directory.
- `02_segment_mimiciii.ipynb` and `02_segment_mc1t1.ipynb`:  
  - **Description:** Segments chest imaging reports and attending physician notes from MIMIC-III and MC1-T1. Necessary step for our ML implementation.  
  - **Runtime:** 44 seconds (MIMIC-III) to 9 minutes (MC1-T1) on Windows 11 machine (CPU: Intel Core i7-1165G7, RAM: 16GB). MC1-T1 takes longer due to larger dataset size. Most time is spent writing the csv files to disk.
  - **Expected outputs:** Both notebooks will write csv files to `Analysis_Data` directory.  
- `03_diagnose_mimiciii.ipynb` and `03_diagnose_mc1t1.ipynb`:  
  - **Description:** Implements ARDS' Berlin definition to diagnose encounters from MIMIC-III and MC1-T1. *These notebooks reproduce Figure 8b and 7b, respectively*.  
  - **Runtime:** 26-115 seconds on Windows 11 machine (CPU: Intel Core i7-1165G7, RAM: 16GB). MC1-T1 takes longer due to larger dataset size.  
  - **Expected outputs:** These notebooks simply display encounter counts and a confusion matrix (*Figure 8b and 7b, respectively*).

## Data description  

To run these notebooks, you will need to download the MC1-T1 dataset at <https://arch.library.northwestern.edu/>, or the MIMIC-III dataset at <https://physionet.org/content/mimiciii/1.4/>. The MIMIC-III dataset is free to access once you sign a Data Use Agreement and open an account at physionet.org.

Once you secure either of these datasets, you will need to place them in directories following the structure below:  

```
├───Analysis_Data
│   ├───mc1_t1
│   ├───MIMIC_III
│   │   └───labeled_subset
│   └───train_ML
├───Anonymized_Data
│   ├───mc1_t1
├───ARDS_diagnosis (this repository)
│   └───src
├───Preprocessed_Data
│   └───MIMIC_III
│       └───labeled_subset
├───Raw_Data
│   └───MIMIC_III
│       ├───labeled_subset
```  

The MC1-T1 dataset should already by anonymized, so you can place the files in the `Anonymized_Data/mc1_t1` directory. The raw MIMIC-III dataset from PhysioNet should be placed in the `Raw_Data/MIMIC_III` directory. However, a subset of 100 labeled encounters from MIMIC-III should be placed in the `Raw_Data/MIMIC_III/labeled_subset` directory.

From there on, the notebooks should take care of writing files to the proper folders, and should be able to execute fully.

To clarify:

- If you are only interested in running the notebooks with MC1-T1 data, you only need to run `02_segment_mc1t1.ipynb` and `03_diagnose_mc1t1.ipynb`, and don't need to worry about any other data folder besides `Anonymized_Data/mc1_t1` and `Analysis_Data/mc1_t1`.  
- If you are only interested in running the notebooks with MIMIC-III data, you'd need to run all `mimicii`-suffixed notebooks, and place the correct files in the `Raw_Data/MIMIC_III` directory.
