import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
	directory.mkdir(parents=True, exist_ok=True)

DATA_FILE = "Mental-Health-dataset.csv"
TARGET_COLUMN = "MentalHealthHistory"
TEST_SIZE = 0.2
RANDOM_STATE = 42

PREPROCESSING = {
	"imputer_strategy": "median",
	"scaler": "standard",
	"handle_imbalance": True,
	"smote_random_state": 42
}

MODELS = {
	"Logistic Regression": {
		"class": "LogisticRegression",
		"params": {
			"random_state": 42,
			"class_weight": "balanced",
			"max_iter": 1000,
			"C": 1.0
		}
	},
	"Random Forest": {
		"class": "RandomForestClassifier",
		"params": {
			"random_state": 42,
			"n_estimators": 100,
			"class_weight": "balanced",
			"max_depth": 10,
			"min_samples_split": 2,
			"min_samples_leaf": 1
		}
	},
	"XGBoost": {
		"class": "XGBClassifier",
		"params": {
			"random_state": 42,
			"eval_metric": "logloss",
			"use_label_encoder": False,
			"learning_rate": 0.1,
			"max_depth": 6,
			"n_estimators": 100
		}
	}
}

EVALUATION = {
	"metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
	"average_methods": {
		"binary": "binary",
		"multiclass": "weighted"
	},
	"cv_folds": 5
}

VISUALIZATION = {
	"figure_size": {
		"metrics_comparison": (12, 8),
		"confusion_matrices": (15, 5),
		"roc_curves": (10, 8),
		"feature_importance": (12, 8),
		"comprehensive_report": (16, 12)
	},
	"colors": {
		"primary": "skyblue",
		"secondary": "lightcoral",
		"tertiary": "lightgreen"
	},
	"style": "default",
	"palette": "husl",
	"dpi": 300
}

EXPORT = {
	"save_models": True,
	"save_predictions": True,
	"save_metrics": True,
	"save_plots": True,
	"format": "png"  
}
