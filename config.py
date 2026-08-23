"""
Configuration file for the Genomic Classification Pipeline
Contains all constants, model configurations, and hyperparameter grids
"""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# ========================================
# IMAGE PROCESSING CONSTANTS
# ========================================
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
DEFAULT_IMG_SIZE = (224, 224)

# ========================================
# GENOMIC SEQUENCE CONSTANTS
# ========================================
alphabet = sorted({'A','B','C','D','G','H','K','M','N','R','S','T','V','W','Y','Z'})
char_to_idx = {c:i for i,c in enumerate(alphabet)}

# ========================================
# DIRECTORY STRUCTURE
# ========================================
RESULTS_SUBDIRS = ['plots', 'models', 'reports', 'data_stats']

# ========================================
# MODEL CONFIGURATIONS
# ========================================
MODELS_CONFIG = {
    'ExtraTrees': {
        'model': ExtraTreesClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'XGBoost': {
        'model': None,  # Will be imported dynamically in models.py
        'params': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression,
        'params': {
            'max_iter': 2000,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'SVM': {
        'model': SVC,
        'params': {
            'kernel': 'rbf',
            'random_state': 42,
            'probability': True
        }
    },
    'KNN': {
        'model': KNeighborsClassifier,
        'params': {
            'n_neighbors': 5,
            'n_jobs': -1
        }
    },
    'NaiveBayes': {
        'model': GaussianNB,
        'params': {}
    },
    'LDA': {
        'model': LinearDiscriminantAnalysis,
        'params': {}
    },
    'QDA': {
        'model': QuadraticDiscriminantAnalysis,
        'params': {}
    }
}

# ========================================
# HYPERPARAMETER GRIDS FOR TUNING
# ========================================
PARAM_GRIDS = {
    'ExtraTrees': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    },
    'LogisticRegression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga']
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
}

# ========================================
# CLUSTERING PARAMETERS
# ========================================
DBSCAN_EPS_VALUES = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
DBSCAN_MIN_SAMPLES = 5
GMM_N_COMPONENTS_RANGE = [2, 3, 4, 5, 6, 7, 8]

# ========================================
# VISUALIZATION PARAMETERS
# ========================================
FIGURE_DPI = 300
PLOT_STYLE = 'seaborn-v0_8'
COLOR_PALETTE = 'Set3'
FIGSIZE_LARGE = (15, 10)
FIGSIZE_MEDIUM = (12, 8)
FIGSIZE_SMALL = (10, 6)

# ========================================
# FEATURE SELECTION PARAMETERS
# ========================================
FEATURE_SELECTION_K_RANGE = [50, 100, 200, 500, 1000]
MIN_FEATURE_VARIANCE = 0.01
CORRELATION_THRESHOLD = 0.95

# ========================================
# CROSS-VALIDATION PARAMETERS
# ========================================
CV_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2 #0.2
VALIDATION_SIZE = 0.3

# ========================================
# PERFORMANCE THRESHOLDS
# ========================================
MIN_ACCURACY_THRESHOLD = 0.5
LEARNING_CURVE_TRAIN_SIZES = np.linspace(0.1, 1.0, 10)

# ========================================
# FILE EXTENSIONS AND FORMATS
# ========================================
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
MODEL_SAVE_FORMAT = '.pkl'
PLOT_SAVE_FORMAT = '.png'
DATA_SAVE_FORMAT = '.csv'