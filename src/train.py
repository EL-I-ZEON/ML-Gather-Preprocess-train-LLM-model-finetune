import transformers
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import utils.model_utils as model_utils

# Load preprocessed data
train_data = load_dataset('csv', data_files='data/processed/train_data.csv')['train']
val_data = load_dataset('csv', data_files='data/processed/val_data.csv')['train']

# Load LLM model and tokenizer
model_name = "bert-base-uncased"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize data
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

train_data = train_data.map(tokenize, batched=True)
val_data = val_data.map(tokenize, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='models/',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    logging_dir='logs/',
    logging_steps=10,
    save_steps=10,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("models/trained_model")

print("Model trained and saved.")