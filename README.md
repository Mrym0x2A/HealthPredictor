# HealthPredictor
Mental Health Prediction Project

## Project Overview

This project implements a complete machine learning pipeline to predict mental health history based on various demographic, behavioral, and psychological factors. The system compares multiple classification algorithms and provides comprehensive evaluation metrics to identify the best-performing model.
In this project, we utilize [Kaggle Datasets](https://www.kaggle.com/datasets/alamshihab075/mental-health-dataset) and perform preprocessing steps to convert it into both numeric and categorical datasets for analysis.

## Key Features

 - Multiple ML Algorithms: Logistic Regression, Random Forest, XGBoost
 - Comprehensive Evaluation: Precision, Recall, F1-Score, ROC-AUC, Confusion Matrices
 - Class Imbalance Handling: SMOTE for balanced training
 - Professional Visualization: Multiple plot types for model comparison
 - Modular Architecture: Clean separation of data processing, training, and visualization
 - Configuration Management: Centralized settings for easy experimentation
 
## Install dependencies

    pip install -r requirements.txt
    
## Usage
##### Run the complete pipeline:
   python main.py
   
## Models Implemented
 1. Logistic Regression
 2. Random Forest
 3. XGBoost

## Evaluation Metrics

##### The project evaluates models using multiple metrics:

 - Accuracy: Overall prediction correctness
 - Precision: Quality of positive predictions
 - Recall: Coverage of actual positive cases
 - F1-Score: Harmonic mean of precision and recall
 - ROC-AUC: Model discrimination ability
 - Confusion Matrix: Detailed error analysis
 
## Output Files
##### Generated Artifacts

 - Models: Saved as .pkl files in models/
 - Metrics: CSV file in results/model_metrics.csv
 - Visualizations: PNG files in results/figures/
