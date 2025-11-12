import pandas as pd
import numpy as np
import joblib
import os
from config import settings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
							 f1_score, roc_auc_score, confusion_matrix, 
							 classification_report, roc_curve)


class ModelTrainer:
	
	def __init__(self):
		self.models = {}
		self.predictions = {}
		self.metrics = {}
		self.n_classes = None
	
	
	def initialize_models(self):
		self.models = {}
		for name, model_config in settings.MODELS.items():  
			model_class = globals()[model_config["class"]]
			model = model_class(**model_config["params"])
			self.models[name] = model
		print(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
	
	
	def detect_type(self, y): #detecting multiclass
		unique_classes = np.unique(y)
		self.n_classes = len(unique_classes)
		if self.n_classes == 2:
			print("Binary classification detected")
			self.average_method = settings.EVALUATION["average_methods"]["binary"]
		else:
			print(f"Multiclass classification detected with {self.n_classes} classes")
			self.average_method = settings.EVALUATION["average_methods"]["multiclass"]
		return self.n_classes
	
	
	def calc_scale_weight(self, y_train): #calculating scale_pos_weight
		if self.n_classes == 2:
			class_counts = np.bincount(y_train)
			return class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1
		return 1
	
	
	def train_models(self, X_train, X_test, y_train, y_test):
		self.detect_type(y_train)
		if self.n_classes == 2 and 'XGBoost' in self.models:
			scale_pos_weight = self.calc_scale_weight(y_train)
			self.models['XGBoost'].set_params(scale_pos_weight=scale_pos_weight)
		
		for name, model in self.models.items():
			print(f"Training {name}...")
			model.fit(X_train, y_train)
			y_pred = model.predict(X_test)
			y_pred_proba = model.predict_proba(X_test)
			self.predictions[name] = {
				'y_pred': y_pred,
				'y_pred_proba': y_pred_proba,
				'model': model
			}
			self.calc_metrics(name, y_test, y_pred, y_pred_proba)
	
	
	def calc_metrics(self, model_name, y_true, y_pred, y_pred_proba): 		# Checking multiclass metrics
		if self.n_classes == 2:
			precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
			recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
			f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
			roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
		else:
			precision = precision_score(y_true, y_pred, average=self.average_method, zero_division=0)
			recall = recall_score(y_true, y_pred, average=self.average_method, zero_division=0)
			f1 = f1_score(y_true, y_pred, average=self.average_method, zero_division=0)
			try: # ROC-AUC metric
				roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=self.average_method)
			except:
				roc_auc = 0.5 
		
		metrics = {
			'Accuracy': accuracy_score(y_true, y_pred),
			'Precision': precision,
			'Recall': recall,
			'F1-Score': f1,
			'ROC-AUC': roc_auc,
			'Confusion Matrix': confusion_matrix(y_true, y_pred),
			'n_classes': self.n_classes
		}
		self.metrics[model_name] = metrics
		print(f"\n{model_name} Results:")
		print(f"Accuracy: {metrics['Accuracy']:.4f}")
		print(f"Precision: {metrics['Precision']:.4f}")
		print(f"Recall: {metrics['Recall']:.4f}")
		print(f"F1-Score: {metrics['F1-Score']:.4f}")
		print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
		print(f"Number of classes: {metrics['n_classes']}")
		print("Confusion Matrix:")
		print(metrics['Confusion Matrix'])
		print("-" * 80)
	
	
	def get_metrics_df(self):
		metrics_df = pd.DataFrame(self.metrics).T
		plot_metrics_df = metrics_df.drop(['Confusion Matrix', 'n_classes'], axis=1, errors='ignore') # Remove non-numeric columns
		return plot_metrics_df
	
	
	def get_roc_curves(self, X_test, y_test):
		roc_data = {}
		
		for name in self.models.keys():
			y_pred_proba = self.predictions[name]['y_pred_proba']
			if self.n_classes == 2:
				fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
				roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
			else:
				roc_data[name] = {}
				for i in range(self.n_classes):
					fpr, tpr, thresholds = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
					roc_data[name][f'Class_{i}'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
		return roc_data
	
	
	def save_models(self, directory=None):
		if directory is None:
			directory = settings.MODELS_DIR
		os.makedirs(directory, exist_ok=True)
		for name, model in self.models.items():
			filename = os.path.join(directory, f'{name.replace(" ", "_").lower()}.pkl')
			joblib.dump(model, filename)
			print(f"Saved {name} to {filename}")
