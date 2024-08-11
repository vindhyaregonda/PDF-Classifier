# PARSPEC Assignment

## Objective
Classify PDF URLs into four categories: **fuses**, **cable**, **lighting**, and **others**.

## Overview
Text extraction was performed directly from URLs to reduce space complexity. For PDFs embedded in viewers, the second page was processed to avoid irrelevant content. After cleaning and processing, **1,035 data points** were used for training.

## Requirements
- **OCR & PDF**: `PyPDF2`, `pytesseract`, `Pillow`, `PyMuPDF`
- **Embedding**: `sentence-transformers`
- **Data Handling**: `pandas`, `numpy`
- **Visualization**: `matplotlib`
- **Web Scraping**: `BeautifulSoup`
- **Machine Learning**: `scikit-learn`, `xgboost`
- **Utilities**: `os`, `re`, `io`, `time`, `concurrent.futures`

## Models

### Deep Learning Model
- **Architecture**: ANN with 128-neuron hidden layer.
- **Loss & Activation**: `Negative Log Likelihood`, `LogSoftmax`.
- **Evaluation**: F1 score, accuracy, confusion matrix.

### Machine Learning Model
- **Model**: XGBoost for multi-class classification.
- **Evaluation**: Same as the DL model.

## Results
Both models performed similarly on the test dataset. The DL model required adjustments to avoid overfitting.

