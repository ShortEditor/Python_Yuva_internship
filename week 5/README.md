# Week 5 - Deep Learning Application in Data Science

## Project
Handwritten Digit Classification using a Convolutional Neural Network (CNN)

## Dataset
Scikit-learn Digits dataset: 1,797 grayscale 8x8 images across 10 digit classes (0-9). The dataset is a public benchmark originally associated with the UCI Machine Learning Repository.

## Framework
PyTorch

## Workflow
1. Load and normalize image data.
2. Stratified train/validation/test split (64%/16%/20%).
3. Train a compact CNN with two convolutional blocks.
4. Use ReLU activations, max pooling, dropout, Adam, and L2-style weight decay.
5. Monitor validation loss and use early stopping.
6. Evaluate on a held-out test set using accuracy, precision, recall, F1, loss, and a confusion matrix.
7. Analyze training curves, per-class F1, and example predictions.

## Main files
- `Week5_Deep_Learning_Report.docx` - comprehensive report
- `deep_learning_digits.py` - reproducible PyTorch training script
- `digits_dataset.csv` - tabular copy of the public dataset
- `training_history.csv` - epoch-level training/validation metrics
- `evaluation_metrics.csv` - final evaluation metrics
- `confusion_matrix.csv` - test confusion matrix
- `figures/` - diagrams and evaluation visualizations
