import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from pathlib import Path
import warnings
import sys
from config import settings
warnings.filterwarnings('ignore')

class DataProcess:
	def __init__(self):
		self.scaler = StandardScaler()
		self.imputer = SimpleImputer(strategy=settings.PREPROCESSING["imputer_strategy"])
		self.smote = SMOTE(random_state=settings.PREPROCESSING["smote_random_state"])
		
	def load_data(self, file_path=None):
		if file_path is None:
			file_path = settings.DATA_DIR / settings.DATA_FILE
			try:
				df = pd.read_csv(file_path)
				print("Dataset Shape:", df.shape)
				print("\nColumn Names:", df.columns.tolist())
				print(df[settings.TARGET_COLUMN].value_counts())
				print(f"Class Ratio: {df[settings.TARGET_COLUMN].value_counts(normalize=True)}")
				self.missing_count = df.isnull().sum().sum()
				return df
			except FileNotFoundError:
				print(f"Error: The file at {file_path} was not found.")
			except pd.errors.EmptyDataError:
				print(f"Error: The file at {file_path} is empty.")
			except pd.errors.ParserError:
				print(f"Error: The file at {file_path} could not be parsed")
			except Exception as error:
				print(f"An unexpected error occurred: {error}")
		
		



	def process_data(self, df, target_col=None, test_size=None):
		
		if target_col is None:
			target_col = settings.TARGET_COLUMN
		if test_size is None:
			test_size = settings.TEST_SIZE
		
		X = df.drop(columns=[target_col])
		y = df[target_col]


		X_train, X_test, y_train, y_test = train_test_split(
			X, y, 
			test_size=test_size, 
			random_state=settings.RANDOM_STATE, 
			stratify=y
		)
		
		X_train_scaled = self.scaler.fit_transform(X_train)
		X_test_scaled = self.scaler.transform(X_test)
		
		
		X_train_imputed = self.imputer.fit_transform(X_train_scaled)
		X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train_imputed, y_train)  

		
		print(f"Original training shape: {X_train.shape}")
		print(f"After preprocessing training shape: {X_train_balanced.shape}")
		print(f"Class distribution after preprocessing: {pd.Series(y_train_balanced).value_counts()}")
		
		return (X_train_balanced, X_test_scaled, y_train_balanced, y_test, 
				X_train, X_test, y_train)
