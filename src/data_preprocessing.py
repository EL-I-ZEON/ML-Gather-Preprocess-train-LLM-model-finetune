import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.data_utils import clean_text, load_data

# Load raw data
raw_data_path = 'data/raw/raw_dataset.csv'
data = load_data(raw_data_path)

# Clean and preprocess data
data['text'] = data['text'].apply(clean_text)

# Split data into train, validation, and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Save processed data
train_data.to_csv('data/processed/train_data.csv', index=False)
val_data.to_csv('data/processed/val_data.csv', index=False)
test_data.to_csv('data/processed/test_data.csv', index=False)

print("Data preprocessed and saved.")