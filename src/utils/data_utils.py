import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_text(text):
    # Add text cleaning steps (e.g., removing special characters, lowercasing)
    return text.strip().lower()