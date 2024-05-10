import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_model(model_name_or_path):
    """
    Load a model and tokenizer from a specified path or model name.

    Args:
        model_name_or_path (str): Path to the model directory or a model name.

    Returns:
        model: The loaded model.
        tokenizer: The tokenizer associated with the model.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def save_model(model, tokenizer, save_path):
    """
    Save a model and tokenizer to a specified path.

    Args:
        model: The model to save.
        tokenizer: The tokenizer to save.
        save_path (str): Path to save the model and tokenizer.
    """
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")


def evaluate_model(model, tokenizer, dataset):
    """
    Evaluate a model's performance on a given dataset.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer associated with the model.
        dataset: The dataset to evaluate the model on.

    Returns:
        dict: A dictionary containing evaluation metrics (accuracy, precision, recall, F1-score).
    """

    # Define a function to compute metrics
    def compute_metrics(pred):
        logits, labels = pred
        predictions = torch.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    # Use the model to evaluate the dataset
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Evaluate the model
    evaluation_results = trainer.evaluate(dataset)

    return evaluation_results


def train_model(model, tokenizer, train_dataset, val_dataset, training_args):
    """
    Train a model on a given dataset using specified training arguments.

    Args:
        model: The model to train.
        tokenizer: The tokenizer associated with the model.
        train_dataset: The dataset to train the model on.
        val_dataset: The dataset to use for validation during training.
        training_args: Training arguments for the training process.

    Returns:
        Trainer: A Trainer instance with the trained model.
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    return trainer


def fine_tune_model(model, tokenizer, fine_tune_dataset, fine_tuning_args):
    """
    Fine-tune a trained model on a fine-tuning dataset using specified fine-tuning arguments.

    Args:
        model: The trained model to fine-tune.
        tokenizer: The tokenizer associated with the model.
        fine_tune_dataset: The dataset to fine-tune the model on.
        fine_tuning_args: Fine-tuning arguments for the fine-tuning process.

    Returns:
        Trainer: A Trainer instance with the fine-tuned model.
    """
    trainer = Trainer(
        model=model,
        args=fine_tuning_args,
        train_dataset=fine_tune_dataset,
        tokenizer=tokenizer,
    )

    # Fine-tune the model
    trainer.train()

    return trainer