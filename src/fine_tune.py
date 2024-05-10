import transformers
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load fine-tuning dataset
fine_tune_data = load_dataset('csv', data_files='data/processed/fine_tune_data.csv')['train']

# Load the trained model and tokenizer
model_name = "models/trained_model"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize fine-tuning data
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

fine_tune_data = fine_tune_data.map(tokenize, batched=True)

# Define fine-tuning arguments
fine_tuning_args = TrainingArguments(
    output_dir='models/fine_tuned_model/',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    logging_dir='logs/fine_tuning/',
    logging_steps=10,
    save_steps=10,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=fine_tuning_args,
    train_dataset=fine_tune_data,
    eval_dataset=None,  # If you have a fine-tuning evaluation set, include it here
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("models/fine_tuned_model")

print("Model fine-tuned and saved.")