import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
import torch
from utils.model_utils import load_model


def evaluate_model(model, tokenizer, dataset):
    """
    Evaluate the model's performance on a given dataset.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer associated with the model.
        dataset: The dataset to evaluate the model on.

    Returns:
        dict: A dictionary containing evaluation metrics (accuracy, precision, recall, F1-score).
    """
    # Lists to store the predictions and actual labels
    all_preds = []
    all_labels = []

    # Loop through the dataset
    for example in dataset:
        # Tokenize the input text
        inputs = tokenizer(example['text'], return_tensors='pt', padding=True, truncation=True)

        # Move inputs to the device (CPU/GPU)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Get the model's predictions
        with torch.no_grad():
            logits = model(**inputs).logits

        # Convert logits to class predictions
        preds = torch.argmax(logits, dim=-1)

        # Append predictions and labels
        all_preds.append(preds.cpu().numpy())
        all_labels.append(example['label'])

    # Flatten the lists
    all_preds = [item for sublist in all_preds for item in sublist]
    all_labels = [item for sublist in all_labels for item in sublist]

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model directory.')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the evaluation dataset.')

    args = parser.parse_args()

    # Load the model and tokenizer
    model, tokenizer = load_model(args.model_path)

    # Load the dataset
    dataset = load_dataset('csv', data_files=args.dataset_path)['train']

    # Evaluate the model
    results = evaluate_model(model, tokenizer, dataset)

    # Print evaluation results
    print("Evaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")


if __name__ == '__main__':
    main()