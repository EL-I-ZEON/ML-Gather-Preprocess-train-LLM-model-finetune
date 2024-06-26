# Disinformation Censorship Project

## Project Overview

This project aims to develop a machine learning model to detect and censor disinformation using large language models (LLMs). The model is trained on a dataset containing text data and associated labels (e.g., disinformation or not disinformation). The project includes data preprocessing, model training, fine-tuning, and evaluation.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your_username/your_project.git
   cd your_project
2. **Create a Python virtual environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
3. **Install required packages**:
    ```bash
    pip install -r requirements.txt

## Explanation of required packages


'pandas' and 'numpy' for data manipulation and numerical computing.

'matplotlib' and 'seaborn' for data visualization.

'torch' for working with PyTorch, a popular machine learning library.

'transformers' for using large language models (LLMs) such as GPT and BERT from the Hugging Face Transformers library.

'datasets' for accessing and managing datasets from the Hugging Face Datasets library.

'PyYAML' for working with YAML configuration files.

'wordcloud' for generating word clouds from text data. 

## Usage
1. **Data Preprocessing**:Run the data preprocessing script to clean and prepare the dataset:
    ```bash
    python src/data_preprocessing.py

2. **Training the Model**:Use the training notebook (train_notebook.ipynb) to train the model:
    ```bash
    jupyter notebook notebooks/train_notebook.ipynb

3. **Fine-Tuning the Model**:Use the fine-tuning script to fine-tune the model on additional data:
    ```bash
    python src/fine_tune.py
4. **Evaluating the Model**:Evaluate the model's performance using the evaluation script:
    ```bash
    python src/evaluation.py --model-path models/trained_model --dataset-path data/processed/test_data.csv
## Project Structure
data/: Raw and processed data.

models/: Trained models.

notebooks/: Jupyter notebooks for exploratory analysis and model training.

src/: Source directory containing scripts for data preprocessing, model training, fine-tuning, and evaluation.

config/: Configuration file (config.yaml).

utils/: Utility functions for data and model handling.

requirements.txt: File listing project dependencies.

## Configuration
The project uses a configuration file (config/config.yaml) to manage parameters such as data paths, model hyperparameters, and evaluation settings. Customize this file according to your requirements.

## Data
The data used in this project includes text data labeled as disinformation or not disinformation. The data is preprocessed and split into training, validation, and test sets.

## Results
Upon completion of the project, you may find the results of model training and evaluation, including metrics such as accuracy, precision, recall, and F1-score, in the respective scripts and notebooks.
