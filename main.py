import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from config import settings
from src.data_preprocessing import DataProcess
from src.model_training import ModelTrainer
from src.visualization import Visualizer
import config.settings

def main():
	
	print("Mental Health Prediction Project")
	print(f"Data Directory: {settings.DATA_DIR}")
	print(f"Models Directory: {settings.MODELS_DIR}")
	print(f"Results Directory: {settings.RESULTS_DIR}")
	
	processor = DataProcess()
	trainer = ModelTrainer()
	visualizer = Visualizer()
	df = processor.load_data()
	
	print(f"\nTarget variable '{settings.TARGET_COLUMN}' distribution:")
	print(df[settings.TARGET_COLUMN].value_counts().sort_index())
	
	(X_train_balanced, X_test_scaled, y_train_balanced, y_test, 
	X_train, X_test, y_train) = processor.process_data(df)
	
	print("\nTraining models...")
	trainer.initialize_models()
	trainer.train_models(X_train_balanced, X_test_scaled, y_train_balanced, y_test)
	metrics_df = trainer.get_metrics_df()
	roc_data = trainer.get_roc_curves(X_test_scaled, y_test)
	
	if settings.EXPORT["save_plots"]:
		print("\nGenerating visualizations...")
		visualizer.plot_metric_compare(metrics_df)
		visualizer.plot_confusion_matrices(trainer.metrics)
		n_classes = trainer.n_classes if hasattr(trainer, 'n_classes') else 2
		visualizer.plot_roc_curves(roc_data, n_classes)
		feature_names = df.drop(columns=[settings.TARGET_COLUMN]).columns.tolist()
		visualizer.plot_feature_importance(trainer.models, feature_names)
		visualizer.final_report(metrics_df, trainer.metrics)
	
	if settings.EXPORT["save_models"]:
		print("\nSaving models...")
		trainer.save_models()
	
	print("\n" + "="*80)
	print("FINAL RESULTS SUMMARY")
	print("="*80)
	print(metrics_df.round(4))
	
	if 'ROC-AUC' in metrics_df.columns:
		best_model = metrics_df['ROC-AUC'].idxmax()
		best_score = metrics_df.loc[best_model, 'ROC-AUC']
		print(f"\nBEST MODEL: {best_model} (ROC-AUC: {best_score:.4f})")
	else:
		best_model = metrics_df['F1-Score'].idxmax()
		best_score = metrics_df.loc[best_model, 'F1-Score']
		print(f"\nBEST MODEL: {best_model} (F1-Score: {best_score:.4f})")
	
	if settings.EXPORT["save_metrics"]:
		metrics_df.to_csv(settings.RESULTS_DIR / 'model_metrics.csv')
		print(f"\nResults saved to '{settings.RESULTS_DIR / 'model_metrics.csv'}'")

if __name__ == "__main__":
	main()
