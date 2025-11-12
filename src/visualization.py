import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from pathlib import Path
import os
import sys
from config import settings

class Visualizer:
	
	def __init__(self, results_dir=None):
		if results_dir is None:
			results_dir = settings.FIGURES_DIR
			
		self.results_dir = results_dir
		os.makedirs(results_dir, exist_ok=True)
		plt.style.use(settings.VISUALIZATION["style"])
		sns.set_palette(settings.VISUALIZATION["palette"])
	
	
	def plot_metric_compare(self, metrics_df, figsize=None):
		if figsize is None:
			figsize = settings.VISUALIZATION["figure_size"]["metrics_comparison"]
			
		fig, axes = plt.subplots(2, 3, figsize=figsize)
		axes = axes.ravel()	
		metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
		
		for i, metric in enumerate(metrics_to_plot):
			if metric in metrics_df.columns:
				metrics_df[metric].plot(
					kind='bar', 
					ax=axes[i], 
					color=settings.VISUALIZATION["colors"]["primary"], 
					edgecolor='black'
				)
				axes[i].set_title(f'{metric} Comparison')
				axes[i].set_ylabel(metric)
				axes[i].tick_params(axis='x', rotation=45)	
				for j, v in enumerate(metrics_df[metric]):
					axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
		
		for i in range(len(metrics_to_plot), len(axes)):
			axes[i].set_visible(False)
		
		plt.tight_layout()
		plt.savefig(
			f'{self.results_dir}/metrics_comparison.{settings.EXPORT["format"]}', 
			dpi=settings.VISUALIZATION["dpi"], 
			bbox_inches='tight'
		)
		plt.show()
	
	def plot_confusion_matrices(self, metrics_dict, figsize=None):
		if figsize is None:
			figsize = settings.VISUALIZATION["figure_size"]["confusion_matrices"]
			
		n_models = len(metrics_dict)
		fig, axes = plt.subplots(1, n_models, figsize=figsize)
		if n_models == 1:
			axes = [axes]
		model_names = list(metrics_dict.keys())
		
		for i, (name, metrics) in enumerate(metrics_dict.items()):
			cm = metrics['Confusion Matrix']
			n_classes = cm.shape[0]
			sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
					   xticklabels=[f'Pred {j}' for j in range(n_classes)],
					   yticklabels=[f'True {j}' for j in range(n_classes)])
			axes[i].set_title(f'Confusion Matrix - {name}')
		
		plt.tight_layout()
		plt.savefig(
			f'{self.results_dir}/confusion_matrices.{settings.EXPORT["format"]}', 
			dpi=settings.VISUALIZATION["dpi"], 
			bbox_inches='tight'
		)
		plt.show()
	
	def plot_roc_curves(self, roc_data, n_classes, figsize=None):
		if figsize is None:
			figsize = settings.VISUALIZATION["figure_size"]["roc_curves"]
			
		plt.figure(figsize=figsize)
		if n_classes == 2:
			for name, data in roc_data.items():
				if 'fpr' in data and 'tpr' in data:  # Binary format
					plt.plot(data['fpr'], data['tpr'], linewidth=2, 
							label=f'{name}')
			
			plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('ROC Curves - Model Comparison (Binary)')
		else:
			first_model = list(roc_data.keys())[0]
			if first_model in roc_data:
				for class_name, class_data in roc_data[first_model].items():
					if 'fpr' in class_data and 'tpr' in class_data:
						plt.plot(class_data['fpr'], class_data['tpr'], linewidth=2,
								label=f'{first_model} - {class_name}')
			
			plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title(f'ROC Curves - {first_model} (Multiclass)')
		
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.savefig(
			f'{self.results_dir}/roc_curves.{settings.EXPORT["format"]}', 
			dpi=settings.VISUALIZATION["dpi"], 
			bbox_inches='tight'
		)
		plt.show()
	
	def plot_feature_importance(self, models, feature_names, figsize=None):
		if figsize is None:
			figsize = settings.VISUALIZATION["figure_size"]["feature_importance"]
			
		tree_models = {name: model for name, model in models.items() 
					  if name in ['Random Forest', 'XGBoost']}
		
		if not tree_models:
			print("No tree-based models found for feature importance")
			return
		
		n_models = len(tree_models)
		fig, axes = plt.subplots(1, n_models, figsize=figsize)
		if n_models == 1:
			axes = [axes]
		
		for i, (name, model) in enumerate(tree_models.items()):
			# Get feature importance
			if hasattr(model, 'feature_importances_'):
				importances = model.feature_importances_
				indices = np.argsort(importances)[::-1]
				
				# Plot top 10 features
				top_n = min(10, len(feature_names))
				axes[i].barh(
					range(top_n), 
					importances[indices[:top_n]][::-1],
					color=settings.VISUALIZATION["colors"]["secondary"]
				)
				axes[i].set_yticks(range(top_n))
				axes[i].set_yticklabels([feature_names[j] for j in indices[:top_n]][::-1])
				axes[i].set_title(f'Feature Importance - {name}')
				axes[i].set_xlabel('Importance')
		
		plt.tight_layout()
		plt.savefig(
			f'{self.results_dir}/feature_importance.{settings.EXPORT["format"]}', 
			dpi=settings.VISUALIZATION["dpi"], 
			bbox_inches='tight'
		)
		plt.show()
	
	def final_report(self, metrics_df, metrics_dict, figsize=None):
		if figsize is None:
			figsize = settings.VISUALIZATION["figure_size"]["comprehensive_report"]
			
		plt.figure(figsize=figsize)
		
		plt.subplot(2, 2, 1)
		metrics_to_plot = [col for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'] 
						  if col in metrics_df.columns]
		metrics_df[metrics_to_plot].plot(kind='bar', ax=plt.gca())
		plt.title('Model Performance Metrics')
		plt.xticks(rotation=45)
		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		
		plt.subplot(2, 2, 2)
		if 'ROC-AUC' in metrics_df.columns:
			metrics_df['ROC-AUC'].plot(
				kind='bar', 
				color=settings.VISUALIZATION["colors"]["secondary"], 
				edgecolor='black'
			)
			plt.title('ROC-AUC Scores')
			plt.ylabel('ROC-AUC')
			plt.xticks(rotation=45)
			
			for i, v in enumerate(metrics_df['ROC-AUC']):
				plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
		
		plt.subplot(2, 2, 3)
		if 'Precision' in metrics_df.columns and 'Recall' in metrics_df.columns:
			width = 0.35
			x = np.arange(len(metrics_df))
			plt.bar(
				x - width/2, metrics_df['Precision'], width, 
				label='Precision', 
				alpha=0.7,
				color=settings.VISUALIZATION["colors"]["primary"]
			)
			plt.bar(
				x + width/2, metrics_df['Recall'], width, 
				label='Recall', 
				alpha=0.7,
				color=settings.VISUALIZATION["colors"]["tertiary"]
			)
			plt.xticks(x, metrics_df.index, rotation=45)
			plt.title('Precision vs Recall')
			plt.legend()
		
		plt.tight_layout()
		plt.savefig(
			f'{self.results_dir}/comprehensive_report.{settings.EXPORT["format"]}', 
			dpi=settings.VISUALIZATION["dpi"], 
			bbox_inches='tight'
		)
		plt.show()
