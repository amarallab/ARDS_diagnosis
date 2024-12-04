# ML pipeline for automatic ARDS diagnosis  

This repository contains the Jupyter notebooks and Python modules for the paper "Open-source machine learning pipeline automatically flags instances of acute respiratory distress syndrome from electronic health records" by Morales et al. (2024). The paper is available at <https://doi.org/10.1101/2024.05.21.24307715>.  

If you use this work, please cite the paper.

## Getting started

### Prerequisites  

You need to have Python installed on your system. You can download Python from the official website. Alternatively, you can use the Anaconda distribution of Python, which includes pip and other useful tools.  

### Installation

1. Clone the repository:  

```bash
git clone https://github.com/amarallab/ARDS_diagnosis.git
```  

2. Navigate to the project directory:  

```bash
cd ARDS_diagnosis
```  

#### Using conda environments

3. Create a fresh conda environment:  

```bash
conda env create -f environment.yml
```  

4. Activate your newly-created environment:  

```bash
conda activate ards
```

#### Using `pyenv` or `venv` environments  

3. Create a fresh environment (Python==3.10.12 recommended):  

```bash
pyenv virtualenv 3.10.12 [name_of_env]  # For pyenv
python -m venv [name_of_env]  # For venv
```

4. Activate the environment:  

```bash
pyenv activate [name_of_env]  # For pyenv
source [name_of_env]/bin/activate  # For venv on Unix
source [name_of_env]\Scripts\activate  # For venv on Windows
```

5. Install required packages:  

```bash
pip install -r requirements.txt
```  

#### Preventing Accidental Notebook Outputs  

To avoid accidentally pushing Jupyter notebooks with their outputs, configure Git locally:

```bash
git config --local include.path ../.gitconfig
```

## Notebook Description  

Notebooks are numbered to indicate at which data processing step the notebook inserts itself into. The first part of the name indicates the purpose, and the last part indicates the cohort. Below is a brief description of each notebook, along with their runtime on a on Windows 11 machine (CPU: Intel Core i7-1165G7, RAM: 16GB).  

- `01_preprocess_mimiciii.ipynb`: Preprocesses MIMIC (2001-12) dataset (standardizes column names, removes duplicates, etc.).  
  - **Runtime:** 50 seconds (depends on user interaction).
  - **Outputs:** Writes CSV files to `Preprocessed_Data/MIMIC_III/labeled_subset`.
- `02_segment_mimiciii.ipynb` and `02_segment_hospital_a_2013.ipynb`: Segments chest imaging reports and attending physician notes from MIMIC (2001-12) and Hospital A (2013).  
  - **Runtime:** 44 seconds (MIMIC (2001-12)) to 9 minutes (Hospital A (2013)).
  - **Outputs:** Write CSV files to `Analysis_Data`.  
- `03_diagnose_mimiciii.ipynb` and `03_diagnose_hospital_a_2013.ipynb`: Implements ARDS' Berlin definition to diagnose encounters from MIMIC (2001-12) and Hospital A (2013). *These notebooks reproduce Figure 9b and 8b, respectively.*  
  - **Runtime:** 26-115 seconds.  
  - **Outputs:** These notebooks display encounter counts and a confusion matrix.

## Data Description  

To run these notebooks, download the [Hospital A (2013) dataset](https://arch.library.northwestern.edu/), or the [MIMIC (2001-12) dataset](https://physionet.org/content/mimiciii/1.4/). MIMIC-III is open access once you sign a Data Use Agreement and open an account at physionet.org.

### Directory Structure  

```plaintext
├───Analysis_Data
│   ├───hospital_a_2013
│   ├───MIMIC_III
│   │   └───labeled_subset
│   └───train_ML
├───Anonymized_Data
│   ├───hospital_a_2013
├───ARDS_diagnosis (this repository)
│   └───src
├───Preprocessed_Data
│   └───MIMIC_III
│       └───labeled_subset
├───Raw_Data
│   └───MIMIC_III
│       ├───labeled_subset
```  

- Place the Hospital A (2013) dataset in `Anonymized_Data/hospital_a_2013`.  
- Place the subset of 100 labeled encounters from MIMIC-III (MIMIC (2001-12)) in `Raw_Data/MIMIC_III/labeled_subset`.

The notebooks will write files to the appropriate folders as they execute.

## Running the Notebooks

- For Hospital A (2013) data: Run `hospital_a_2013`-suffixed notebooks (in order, if the first time).  
- For MIMIC-III data: Run all `mimiciii`-suffixed notebooks (in order, if the first time).  

With this setup, you should be able to fully execute the notebooks and reproduce the results from the paper.
