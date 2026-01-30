import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
import os
import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import joblib
import json
df = pd.read_csv('../data/raw/bank-additional-full.csv', sep=';')
y = (df['y'] == 'yes').astype(int)
X = df.drop(columns = 'y')
y
# Импортируем функции обучения модели
from src.training import *
param = {'class_weight' : 'balanced'}
log_reg = ModelTrainer(LogisticRegression, param)
log_reg.fit(X, y, threshold = 0.5)
log_reg = ModelTrainer(LogisticRegression, param, use_smote = True)
log_reg.fit(X, y, threshold = 0.5)
log_reg = ModelTrainer(LogisticRegression, param, use_smote = False)
log_reg.fit(X, y, threshold = 0.5)
log_reg = ModelTrainer(LogisticRegression, param, use_smote = True)
log_reg.fit(X, y, threshold = 0.5)
log_reg.fit(X, y, threshold = 0.67)
log_reg.fit(X, y, threshold = 0.65)
log_reg.fit(X, y, threshold = 0.66)
log_reg.fit(X, y, threshold = 0.67)
log_reg.fit(X, y, threshold = 0.68)
log_reg.fit(X, y, threshold = 0.67)
log_reg = ModelTrainer(LogisticRegression, param, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
log_reg = ModelTrainer(LogisticRegression, param, use_smote = True)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 8,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = True)
log_reg.fit(X, y, threshold = 0.67)
log_reg.fit(X, y, threshold = 0.65)
log_reg.fit(X, y, threshold = 0.5)
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.5)
log_reg.fit(X, y, threshold = 0.67)
log_reg.fit(X, y, threshold = 0.65)
log_reg.tune(X, y, param_grid = params)
log_reg.fit(X, y, threshold = 0.68)
log_reg.fit(X, y, threshold = 0.7)
log_reg.fit(X, y, threshold = 0.75)
log_reg.fit(X, y, threshold = 0.74)
log_reg.fit(X, y, threshold = 0.73)
log_reg.fit(X, y, threshold = 0.69)
log_reg.fit(X, y, threshold = 0.67)
log_reg.fit(X, y, threshold = 0.66)
log_reg.fit(X, y, threshold = 0.67)
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False, use_scaler = False)
log_reg.fit(X, y, threshold = 0.67)
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 7,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 7.5,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 7.3,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 7.2,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 7.1,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 7,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 8,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 8.1,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 8.2,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 8.5,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 8.6,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 9,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 8,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 8,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 8,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.67)
log_reg.fit(X, y, threshold = 0.65)
log_reg.get_feature_importance()
log_reg.fit(X, y, threshold = 0.64)
log_reg.get_feature_importance()
log_reg.fit(X, y, threshold = 0.65)
log_reg.fit(X, y, threshold = 0.66)
log_reg.fit(X, y, threshold = 0.65)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 8.2,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.65)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 8.1,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.65)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 6,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.65)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 5,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.65)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 7.4,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.65)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 7.5,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.65)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 7.3,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.65)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 7.2,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.65)
from xgboost import XGBClassifier
import pandas as pd

# Базовые параметры (без scale_pos_weight)
xgb_params_base = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

print("Проверка scale_pos_weight от 5.0 до 10.0 с шагом 0.1:")
print("="*70)

# Сохраняем результаты
results = []

# Перебираем веса от 5.0 до 10.0 с шагом 0.1
for weight in [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 
               6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9,
               7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9,
               8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9,
               9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0]:
    
    # Обновляем scale_pos_weight
    current_params = xgb_params_base.copy()
    current_params['scale_pos_weight'] = weight
    
    # Создаем и обучаем модель
    log_reg = ModelTrainer(
        XGBClassifier, 
        current_params, 
        use_smote=False
    )
    
    log_reg.fit(X, y, threshold=0.65)
    
    # Сохраняем результаты
    results.append({
        'scale_pos_weight': weight,
        'f1': log_reg.metrics['f1'],
        'precision': log_reg.metrics['precision'],
        'recall': log_reg.metrics['recall'],
        'accuracy': log_reg.metrics['accuracy']
    })
    
    # Выводим прогресс каждый 5-й вес
    if weight in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        print(f"Проверено до weight={weight}...")

# Преобразуем в DataFrame
results_df = pd.DataFrame(results)

# Находим лучший вес по F1
best_f1_idx = results_df['f1'].idxmax()
best_weight_f1 = results_df.loc[best_f1_idx, 'scale_pos_weight']
best_f1 = results_df.loc[best_f1_idx, 'f1']

# Находим лучший вес по балансу Precision/Recall
results_df['f1_pr_balance'] = 2 * (results_df['precision'] * results_df['recall']) / (results_df['precision'] + results_df['recall'])
best_balance_idx = results_df['f1_pr_balance'].idxmax()
best_weight_balance = results_df.loc[best_balance_idx, 'scale_pos_weight']

print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ:")
print("="*70)
print(f"Всего проверено весов: {len(results_df)}")
print(f"\nЛучший scale_pos_weight по F1: {best_weight_f1:.1f}")
print(f"F1-score: {best_f1:.4f}")
print(f"Precision: {results_df.loc[best_f1_idx, 'precision']:.4f}")
print(f"Recall: {results_df.loc[best_f1_idx, 'recall']:.4f}")
print(f"\nЛучший balance по Precision/Recall: {best_weight_balance:.1f}")
from xgboost import XGBClassifier
import pandas as pd

# Базовые параметры (без scale_pos_weight)
xgb_params_base = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

print("Проверка scale_pos_weight от 5.0 до 10.0 с шагом 0.1:")
print("="*70)

# Сохраняем результаты
results = []

# Перебираем веса от 5.0 до 10.0 с шагом 0.1
for weight in [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 
               6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9,
               7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9,
               8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9,
               9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0]:
    
    # Обновляем scale_pos_weight
    current_params = xgb_params_base.copy()
    current_params['scale_pos_weight'] = weight
    
    # Создаем и обучаем модель
    log_reg = ModelTrainer(
        XGBClassifier, 
        current_params, 
        use_smote=False
    )
    
    log_reg.fit(X, y, threshold=0.67)
    
    # Сохраняем результаты
    results.append({
        'scale_pos_weight': weight,
        'f1': log_reg.metrics['f1'],
        'precision': log_reg.metrics['precision'],
        'recall': log_reg.metrics['recall'],
        'accuracy': log_reg.metrics['accuracy']
    })
    
    # Выводим прогресс каждый 5-й вес
    if weight in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        print(f"Проверено до weight={weight}...")

# Преобразуем в DataFrame
results_df = pd.DataFrame(results)

# Находим лучший вес по F1
best_f1_idx = results_df['f1'].idxmax()
best_weight_f1 = results_df.loc[best_f1_idx, 'scale_pos_weight']
best_f1 = results_df.loc[best_f1_idx, 'f1']

# Находим лучший вес по балансу Precision/Recall
results_df['f1_pr_balance'] = 2 * (results_df['precision'] * results_df['recall']) / (results_df['precision'] + results_df['recall'])
best_balance_idx = results_df['f1_pr_balance'].idxmax()
best_weight_balance = results_df.loc[best_balance_idx, 'scale_pos_weight']

print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ:")
print("="*70)
print(f"Всего проверено весов: {len(results_df)}")
print(f"\nЛучший scale_pos_weight по F1: {best_weight_f1:.1f}")
print(f"F1-score: {best_f1:.4f}")
print(f"Precision: {results_df.loc[best_f1_idx, 'precision']:.4f}")
print(f"Recall: {results_df.loc[best_f1_idx, 'recall']:.4f}")
print(f"\nЛучший balance по Precision/Recall: {best_weight_balance:.1f}")
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

# Базовые параметры
xgb_params_base = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

print("Грид поиск: scale_pos_weight от 5.0 до 10.0 и threshold от 0.6 до 0.75")
print("="*80)

# Сохраняем все результаты
all_results = []
counter = 0

# Перебираем оба параметра
weights = np.arange(5.0, 10.1, 0.1)  # 5.0, 5.1, ..., 10.0
thresholds = np.arange(0.6, 0.76, 0.01)  # 0.6, 0.61, ..., 0.75

total_combinations = len(weights) * len(thresholds)
print(f"Всего комбинаций: {total_combinations}")

for weight in weights:
    # Обновляем scale_pos_weight
    current_params = xgb_params_base.copy()
    current_params['scale_pos_weight'] = weight
    
    for threshold in thresholds:
        counter += 1
        
        # Создаем и обучаем модель
        log_reg = ModelTrainer(
            XGBClassifier, 
            current_params, 
            use_smote=False
        )
        
        log_reg.fit(X, y, threshold=threshold)
        
        # Сохраняем результаты
        all_results.append({
            'scale_pos_weight': weight,
            'threshold': threshold,
            'f1': log_reg.metrics['f1'],
            'precision': log_reg.metrics['precision'],
            'recall': log_reg.metrics['recall'],
            'accuracy': log_reg.metrics['accuracy']
        })
        
        # Прогресс каждые 50 комбинаций
        if counter % 50 == 0:
            print(f"Проверено {counter}/{total_combinations} комбинаций...")

# Преобразуем в DataFrame
results_df = pd.DataFrame(all_results)

# Находим лучшие комбинации
print("\n" + "="*80)
print("ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
print("="*80)

# 1. Лучший по F1
best_f1_idx = results_df['f1'].idxmax()
best_f1_row = results_df.loc[best_f1_idx]

print("\n1. ЛУЧШИЙ ПО F1:")
print(f"  scale_pos_weight = {best_f1_row['scale_pos_weight']:.1f}")
print(f"  threshold = {best_f1_row['threshold']:.2f}")
print(f"  F1-score = {best_f1_row['f1']:.4f}")
print(f"  Precision = {best_f1_row['precision']:.4f}")
print(f"  Recall = {best_f1_row['recall']:.4f}")

# 2. Лучший баланс Precision/Recall (F1 уже есть, но можно по harmonic mean)
results_df['f1_pr_diff'] = np.abs(results_df['precision'] - results_df['recall'])
balanced_idx = results_df['f1_pr_diff'].idxmin()
balanced_row = results_df.loc[balanced_idx]

print("\n2. НАИБОЛЕЕ СБАЛАНСИРОВАННЫЙ (минимальная разница P-R):")
print(f"  scale_pos_weight = {balanced_row['scale_pos_weight']:.1f}")
print(f"  threshold = {balanced_row['threshold']:.2f}")
print(f"  F1-score = {balanced_row['f1']:.4f}")
print(f"  Precision = {balanced_row['precision']:.4f}")
print(f"  Recall = {balanced_row['recall']:.4f}")
print(f"  |P-R| = {balanced_row['f1_pr_diff']:.4f}")

# 3. Топ-5 результатов
print("\n3. ТОП-5 КОМБИНАЦИЙ ПО F1:")
top5 = results_df.nlargest(5, 'f1')[['scale_pos_weight', 'threshold', 'f1', 'precision', 'recall']]
print(top5.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import numpy as np

def objective(trial):
    """
    Функция для Optuna, оптимизирующая F1-score с вашим ModelTrainer
    """
    # 1. Параметры XGBoost для подбора
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    # 2. Порог классификации
    threshold = trial.suggest_float('threshold', 0.5, 0.8)
    
    # 3. Разделяем данные на train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # 4. Создаем и обучаем ваш ModelTrainer
    trainer = ModelTrainer(
        model_class=XGBClassifier,
        model_params=xgb_params,
        use_smote=False,  # Без SMOTE как вы просили
        random_state=42
    )
    
    # 5. Обучаем на тренировочной части
    trainer.fit(X_train_split, y_train_split, threshold=threshold, test_size=0)  
    # test_size=0 чтобы не делать лишнего разделения
    
    # 6. Оцениваем на валидационной выборке
    y_val_pred = trainer.predict(X_val)
    f1_val = f1_score(y_val, y_val_pred)
    
    return f1_val

# Запуск оптимизации
print("Запуск Optuna оптимизации...")
study = optuna.create_study(
    direction='maximize',  # Максимизируем F1-score
    study_name='bank_marketing_xgb',
    sampler=optuna.samplers.TPESampler(seed=42)  # Для воспроизводимости
)

# Оптимизируем (можно увеличить n_trials для лучшего результата)
study.optimize(objective, n_trials=30, show_progress_bar=True)

# Вывод результатов
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ OPTUNA ОПТИМИЗАЦИИ")
print("="*60)
print(f"Лучший F1-score: {study.best_value:.4f}")
print(f"\nЛучшие параметры:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import numpy as np

def objective(trial):
    """
    Функция для Optuna, оптимизирующая F1-score с вашим ModelTrainer
    """
    # 1. Параметры XGBoost для подбора
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    # 2. Порог классификации
    threshold = trial.suggest_float('threshold', 0.5, 0.8)
    
    # 3. Разделяем данные на train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # 4. Создаем и обучаем ваш ModelTrainer
    trainer = ModelTrainer(
        model_class=XGBClassifier,
        model_params=xgb_params,
        use_smote=False,  # Без SMOTE как вы просили
        random_state=42
    )
    
    # 5. Обучаем на тренировочной части
    trainer.fit(X_train_split, y_train_split, threshold=threshold, test_size=0)  
    # test_size=0 чтобы не делать лишнего разделения
    
    # 6. Оцениваем на валидационной выборке
    y_val_pred = trainer.predict(X_val)
    f1_val = f1_score(y_val, y_val_pred)
    
    return f1_val

# Запуск оптимизации
print("Запуск Optuna оптимизации...")
study = optuna.create_study(
    direction='maximize',  # Максимизируем F1-score
    study_name='bank_marketing_xgb',
    sampler=optuna.samplers.TPESampler(seed=42)  # Для воспроизводимости
)

# Оптимизируем (можно увеличить n_trials для лучшего результата)
study.optimize(objective, n_trials=30, show_progress_bar=True)

# Вывод результатов
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ OPTUNA ОПТИМИЗАЦИИ")
print("="*60)
print(f"Лучший F1-score: {study.best_value:.4f}")
print(f"\nЛучшие параметры:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import numpy as np

def objective(trial):
    """
    Функция для Optuna, оптимизирующая F1-score с вашим ModelTrainer
    """
    # 1. Параметры XGBoost для подбора
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    # 2. Порог классификации
    threshold = trial.suggest_float('threshold', 0.5, 0.8)
    
    # 3. Разделяем данные на train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # 4. Создаем и обучаем ваш ModelTrainer
    trainer = ModelTrainer(
        model_class=XGBClassifier,
        model_params=xgb_params,
        use_smote=False,  # Без SMOTE как вы просили
        random_state=42
    )
    
    # 5. Обучаем на тренировочной части
    trainer.fit(X_train_split, y_train_split, threshold=threshold, test_size=0)  
    # test_size=0 чтобы не делать лишнего разделения
    
    # 6. Оцениваем на валидационной выборке
    y_val_pred = trainer.predict(X_val)
    f1_val = f1_score(y_val, y_val_pred)
    
    return f1_val

# Запуск оптимизации
print("Запуск Optuna оптимизации...")
study = optuna.create_study(
    direction='maximize',  # Максимизируем F1-score
    study_name='bank_marketing_xgb',
    sampler=optuna.samplers.TPESampler(seed=42)  # Для воспроизводимости
)

# Оптимизируем (можно увеличить n_trials для лучшего результата)
study.optimize(objective, n_trials=30, show_progress_bar=True)

# Вывод результатов
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ OPTUNA ОПТИМИЗАЦИИ")
print("="*60)
print(f"Лучший F1-score: {study.best_value:.4f}")
print(f"\nЛучшие параметры:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import numpy as np

def objective(trial):
    """
    Функция для Optuna, оптимизирующая F1-score с вашим ModelTrainer
    """
    # 1. Параметры XGBoost для подбора
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    # 2. Порог классификации
    threshold = trial.suggest_float('threshold', 0.5, 0.8)
    
    # 3. Разделяем данные на train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # 4. Создаем и обучаем ваш ModelTrainer
    trainer = ModelTrainer(
        model_class=XGBClassifier,
        model_params=xgb_params,
        use_smote=False,  # Без SMOTE как вы просили
        random_state=42
    )
    
    # 5. Обучаем на тренировочной части
    trainer.fit(X_train_split, y_train_split, threshold=threshold, test_size=0)  
    # test_size=0 чтобы не делать лишнего разделения
    
    # 6. Оцениваем на валидационной выборке
    y_val_pred = trainer.predict(X_val)
    f1_val = f1_score(y_val, y_val_pred)
    
    return f1_val

# Запуск оптимизации
print("Запуск Optuna оптимизации...")
study = optuna.create_study(
    direction='maximize',  # Максимизируем F1-score
    study_name='bank_marketing_xgb',
    sampler=optuna.samplers.TPESampler(seed=42)  # Для воспроизводимости
)

# Оптимизируем (можно увеличить n_trials для лучшего результата)
study.optimize(objective, n_trials=30, show_progress_bar=True)

# Вывод результатов
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ OPTUNA ОПТИМИЗАЦИИ")
print("="*60)
print(f"Лучший F1-score: {study.best_value:.4f}")
print(f"\nЛучшие параметры:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import numpy as np

def objective(trial):
    """
    Функция для Optuna, оптимизирующая F1-score с вашим ModelTrainer
    """
    # 1. Параметры XGBoost для подбора
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    # 2. Порог классификации
    threshold = trial.suggest_float('threshold', 0.5, 0.8)
    
    # 3. Разделяем данные на train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # 4. Создаем и обучаем ваш ModelTrainer
    trainer = ModelTrainer(
        model_class=XGBClassifier,
        model_params=xgb_params,
        use_smote=False,  # Без SMOTE как вы просили
        random_state=42
    )
    
    # 5. Обучаем на тренировочной части
    trainer.fit(X_train_split, y_train_split, threshold=threshold)  
    # test_size=0 чтобы не делать лишнего разделения
    
    # 6. Оцениваем на валидационной выборке
    y_val_pred = trainer.predict(X_val)
    f1_val = f1_score(y_val, y_val_pred)
    
    return f1_val

# Запуск оптимизации
print("Запуск Optuna оптимизации...")
study = optuna.create_study(
    direction='maximize',  # Максимизируем F1-score
    study_name='bank_marketing_xgb',
    sampler=optuna.samplers.TPESampler(seed=42)  # Для воспроизводимости
)

# Оптимизируем (можно увеличить n_trials для лучшего результата)
study.optimize(objective, n_trials=30, show_progress_bar=True)

# Вывод результатов
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ OPTUNA ОПТИМИЗАЦИИ")
print("="*60)
print(f"Лучший F1-score: {study.best_value:.4f}")
print(f"\nЛучшие параметры:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import numpy as np

def objective(trial):
    """
    Функция для Optuna, оптимизирующая F1-score с вашим ModelTrainer
    """
    # 1. Параметры XGBoost для подбора
   xgb_params = {
    # Основные параметры
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # увеличен диапазон
    'max_depth': trial.suggest_int('max_depth', 3, 15),  # глубже деревья
    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),  # шире диапазон
    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0),  # увеличен диапазон для дисбаланса
    
    # Сэмплирование
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # от 0.5
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),  # новый параметр
    'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),  # новый параметр
    
    # Регуляризация
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),  # L1 регуляризация
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),  # L2 регуляризация
    'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),  # минимальное уменьшение потерь
    
    # Параметры дерева
    'min_child_weight': trial.suggest_float('min_child_weight', 1, 20),  # минимальный вес в листе
    'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),  # ограничение шага
    
    # Метод бустинга
    'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),  # разные бустеры
    
    # Для метода dart (Dropouts meet Multiple Additive Regression Trees)
    'rate_drop': trial.suggest_float('rate_drop', 0.0, 0.5) if trial.params.get('booster') == 'dart' else 0.0,
    'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.5) if trial.params.get('booster') == 'dart' else 0.0,
    
    # Прочие параметры
    'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
    'tree_method': trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist', 'gpu_hist']),
    
    # Обучение
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': trial.suggest_categorical('eval_metric', [
        'logloss', 'auc', 'aucpr', 'error', 'merror', 'mlogloss', 
        'rmse', 'mae', 'map', 'ndcg', 'ndcg@n', 'ndcg-'
    ]),
    'objective': trial.suggest_categorical('objective', [
        'binary:logistic', 'binary:logitraw', 'binary:hinge'
    ])
}
    
    # 2. Порог классификации
    threshold = trial.suggest_float('threshold', 0.5, 0.8)
    
    # 3. Разделяем данные на train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # 4. Создаем и обучаем ваш ModelTrainer
    trainer = ModelTrainer(
        model_class=XGBClassifier,
        model_params=xgb_params,
        use_smote=False,  # Без SMOTE как вы просили
        random_state=42
    )
    
    # 5. Обучаем на тренировочной части
    trainer.fit(X_train_split, y_train_split, threshold=threshold)  
    # test_size=0 чтобы не делать лишнего разделения
    
    # 6. Оцениваем на валидационной выборке
    y_val_pred = trainer.predict(X_val)
    f1_val = f1_score(y_val, y_val_pred)
    
    return f1_val

# Запуск оптимизации
print("Запуск Optuna оптимизации...")
study = optuna.create_study(
    direction='maximize',  # Максимизируем F1-score
    study_name='bank_marketing_xgb',
    sampler=optuna.samplers.TPESampler(seed=42)  # Для воспроизводимости
)

# Оптимизируем (можно увеличить n_trials для лучшего результата)
study.optimize(objective, n_trials=30, show_progress_bar=True)

# Вывод результатов
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ OPTUNA ОПТИМИЗАЦИИ")
print("="*60)
print(f"Лучший F1-score: {study.best_value:.4f}")
print(f"\nЛучшие параметры:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import numpy as np

def objective(trial):
    
    """
    Функция для Optuna, оптимизирующая F1-score с вашим ModelTrainer
    """
    # 1. Параметры XGBoost для подбора
    xgb_params = {
    # Основные параметры
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # увеличен диапазон
    'max_depth': trial.suggest_int('max_depth', 3, 15),  # глубже деревья
    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),  # шире диапазон
    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0),  # увеличен диапазон для дисбаланса
    
    # Сэмплирование
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # от 0.5
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),  # новый параметр
    'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),  # новый параметр
    
    # Регуляризация
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),  # L1 регуляризация
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),  # L2 регуляризация
    'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),  # минимальное уменьшение потерь
    
    # Параметры дерева
    'min_child_weight': trial.suggest_float('min_child_weight', 1, 20),  # минимальный вес в листе
    'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),  # ограничение шага
    
    # Метод бустинга
    'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),  # разные бустеры
    
    # Для метода dart (Dropouts meet Multiple Additive Regression Trees)
    'rate_drop': trial.suggest_float('rate_drop', 0.0, 0.5) if trial.params.get('booster') == 'dart' else 0.0,
    'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.5) if trial.params.get('booster') == 'dart' else 0.0,
    
    # Прочие параметры
    'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
    'tree_method': trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist', 'gpu_hist']),
    
    # Обучение
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': trial.suggest_categorical('eval_metric', [
        'logloss', 'auc', 'aucpr', 'error', 'merror', 'mlogloss', 
        'rmse', 'mae', 'map', 'ndcg', 'ndcg@n', 'ndcg-'
    ]),
    'objective': trial.suggest_categorical('objective', [
        'binary:logistic', 'binary:logitraw', 'binary:hinge'
    ])
}
    
    # 2. Порог классификации
    threshold = trial.suggest_float('threshold', 0.5, 0.8)
    
    # 3. Разделяем данные на train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # 4. Создаем и обучаем ваш ModelTrainer
    trainer = ModelTrainer(
        model_class=XGBClassifier,
        model_params=xgb_params,
        use_smote=False,  # Без SMOTE как вы просили
        random_state=42
    )
    
    # 5. Обучаем на тренировочной части
    trainer.fit(X_train_split, y_train_split, threshold=threshold)  
    # test_size=0 чтобы не делать лишнего разделения
    
    # 6. Оцениваем на валидационной выборке
    y_val_pred = trainer.predict(X_val)
    f1_val = f1_score(y_val, y_val_pred)
    
    return f1_val

# Запуск оптимизации
print("Запуск Optuna оптимизации...")
study = optuna.create_study(
    direction='maximize',  # Максимизируем F1-score
    study_name='bank_marketing_xgb',
    sampler=optuna.samplers.TPESampler(seed=42)  # Для воспроизводимости
)

# Оптимизируем (можно увеличить n_trials для лучшего результата)
study.optimize(objective, n_trials=30, show_progress_bar=True)

# Вывод результатов
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ OPTUNA ОПТИМИЗАЦИИ")
print("="*60)
print(f"Лучший F1-score: {study.best_value:.4f}")
print(f"\nЛучшие параметры:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import numpy as np

def objective(trial):
    
    """
    Функция для Optuna, оптимизирующая F1-score с вашим ModelTrainer
    """
    # 1. Параметры XGBoost для подбора
    xgb_params = {
    # Основные параметры
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    'max_depth': trial.suggest_int('max_depth', 3, 15),
    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0),
    
    # Сэмплирование (убрал colsample_bynode для совместимости)
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
    
    # Регуляризация
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
    'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
    
    # Параметры дерева
    'min_child_weight': trial.suggest_float('min_child_weight', 1, 20),
    'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
    
    # Метод бустинга
    'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),  # убрал 'dart' для простоты
    
    # Метод построения деревьев
    'tree_method': trial.suggest_categorical('tree_method', ['auto', 'hist', 'approx']),  # убрал 'exact' и 'gpu_hist'
    
    # Политика роста дерева
    'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
    
    # Обучение
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'objective': 'binary:logistic'
}
    
    # 2. Порог классификации
    threshold = trial.suggest_float('threshold', 0.5, 0.8)
    
    # 3. Разделяем данные на train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # 4. Создаем и обучаем ваш ModelTrainer
    trainer = ModelTrainer(
        model_class=XGBClassifier,
        model_params=xgb_params,
        use_smote=False,  # Без SMOTE как вы просили
        random_state=42
    )
    
    # 5. Обучаем на тренировочной части
    trainer.fit(X_train_split, y_train_split, threshold=threshold)  
    # test_size=0 чтобы не делать лишнего разделения
    
    # 6. Оцениваем на валидационной выборке
    y_val_pred = trainer.predict(X_val)
    f1_val = f1_score(y_val, y_val_pred)
    
    return f1_val

# Запуск оптимизации
print("Запуск Optuna оптимизации...")
study = optuna.create_study(
    direction='maximize',  # Максимизируем F1-score
    study_name='bank_marketing_xgb',
    sampler=optuna.samplers.TPESampler(seed=42)  # Для воспроизводимости
)

# Оптимизируем (можно увеличить n_trials для лучшего результата)
study.optimize(objective, n_trials=30, show_progress_bar=True)

# Вывод результатов
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ OPTUNA ОПТИМИЗАЦИИ")
print("="*60)
print(f"Лучший F1-score: {study.best_value:.4f}")
print(f"\nЛучшие параметры:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 8.1,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 6.1,  # важно! ~1/0.117 для Bank Marketing
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 7.1,
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
log_reg.fit(X, y, threshold = 0.58)
log_reg.fit(X, y, threshold = 0.61
log_reg.fit(X, y, threshold = 0.61)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 5,
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 5.1,
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 5.2,
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 5.3,
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 5.4,
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 5.5,
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 5.6,
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 4,
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 4.5
    ,
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
xgb_params_basic = {
    # Основные параметры
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 6,
    
    # Регуляризация
    'reg_alpha': 0,      # L1 регуляризация
    'reg_lambda': 1,     # L2 регуляризация
    
    # Дисбаланс классов
    'scale_pos_weight': 5.1
    ,
    
    # Прочее
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
import numpy as np

# Создаем объект модели
model = ModelTrainer(XGBClassifier, xgb_params_basic)

best_score = 0
best_params = {}
best_metrics = {}

# Сохраняем все результаты для анализа
results = []

# Перебираем scale_pos_weight от 3 до 9
for weight in range(3, 10):  # 3, 4, 5, 6, 7, 8, 9
    
    # Обновляем параметры XGBoost
    current_params = {**xgb_params_basic, 'scale_pos_weight': weight}
    
    # Создаем новую модель с текущим весом
    model = ModelTrainer(XGBClassifier, current_params)
    
    # Обучаем модель с текущим порогом (порог будет перебираться внутри)
    # Сначала обучим модель с порогом по умолчанию (0.5)
    model.fit(X, y, threshold=0.5)
    
    # Перебираем пороги для уже обученной модели
    for threshold in np.arange(0.5, 0.91, 0.05):
        threshold = round(threshold, 2)
        
        # Пересчитываем метрики с новым порогом
        metrics = model._calculate_metrics(threshold)
        
        # Используем F1-score как основную метрику
        f1_score_value = metrics['f1']
        
        results.append({
            'scale_pos_weight': weight,
            'threshold': threshold,
            'f1': f1_score_value,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'roc_auc': metrics['roc_auc'],
            'accuracy': metrics['accuracy']
        })
        
        print(f"Weight: {weight}, Threshold: {threshold:.2f}, "
              f"F1: {f1_score_value:.4f}, "
              f"Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}")
        
        # Сохраняем лучшие параметры по F1-score
        if f1_score_value > best_score:
            best_score = f1_score_value
            best_params = {
                'scale_pos_weight': weight,
                'threshold': threshold
            }
            best_metrics = metrics.copy()

print("\n" + "="*60)
print("ЛУЧШИЕ ПАРАМЕТРЫ:")
print("="*60)
print(f"scale_pos_weight: {best_params['scale_pos_weight']}")
print(f"threshold: {best_params['threshold']:.2f}")
print(f"F1-score: {best_metrics['f1']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall: {best_metrics['recall']:.4f}")
print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")

# Создаем финальную модель с лучшими параметрами
best_xgb_params = {**xgb_params_basic, 'scale_pos_weight': best_params['scale_pos_weight']}
best_model = ModelTrainer(XGBClassifier, best_xgb_params)

# Обучаем финальную модель с лучшим порогом
best_model.fit(X, y, threshold=best_params['threshold'])
import numpy as np
from sklearn.model_selection import train_test_split

# Сначала разделим данные для согласованности
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,  # фиксируем random_state для воспроизводимости
    stratify=y
)

best_score = 0
best_params = {}
results = []

for weight in range(3, 10):
    for threshold in np.arange(0.5, 0.91, 0.05):
        threshold = round(threshold, 2)
        
        # Создаем и обучаем модель с текущими параметрами
        current_params = {**xgb_params_basic, 'scale_pos_weight': weight}
        model = ModelTrainer(XGBClassifier, current_params)
        
        # Обучаем на тренировочных данных
        model.fit(X_train, y_train, threshold=threshold)
        
        # Получаем метрики
        metrics = model.metrics  # Метрики уже вычислены в fit()
        
        results.append({
            'scale_pos_weight': weight,
            'threshold': threshold,
            **metrics
        })
        
        print(f"Weight: {weight}, Threshold: {threshold:.2f}, "
              f"F1: {metrics['f1']:.4f}")
        
        if metrics['f1'] > best_score:
            best_score = metrics['f1']
            best_params = {
                'scale_pos_weight': weight,
                'threshold': threshold
            }

print(f"\nBest: weight={best_params['scale_pos_weight']}, "
      f"threshold={best_params['threshold']:.2f}, "
      f"F1={best_score:.4f}")

# Обучение финальной модели на всех данных с лучшими параметрами
final_params = {**xgb_params_basic, 'scale_pos_weight': best_params['scale_pos_weight']}
final_model = ModelTrainer(XGBClassifier, final_params)
final_model.fit(X, y, threshold=best_params['threshold'])
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = True)
log_reg.fit(X, y, threshold = 0.61)
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def objective(trial):
    """
    Функция для Optuna, оптимизирующая F1-score
    """
    # Разделяем данные на тренировочные и валидационные
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Параметры для подбора
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 15),  # можно и int, но float дает больше гибкости
    }
    
    # Порог классификации
    threshold = trial.suggest_float('threshold', 0.3, 0.9)
    
    try:
        # Создаем и обучаем модель
        model = ModelTrainer(XGBClassifier, params)
        model.fit(X_train, y_train, threshold=threshold)
        
        # Получаем метрики на валидации
        y_val_pred = (model.pipeline.predict_proba(X_val)[:, 1] >= threshold).astype(int)
        f1 = f1_score(y_val, y_val_pred)
        
        # Можно использовать другую метрику или комбинацию
        # Например, учитывать precision и recall
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        
        # Можно оптимизировать F1 или F-beta score
        # Или добавить penalty за слишком низкий precision/recall
        return f1
        
    except Exception as e:
        # Если возникла ошибка, возвращаем плохой скор
        print(f"Ошибка в trial {trial.number}: {e}")
        return 0.0

# Создаем study и запускаем оптимизацию
study = optuna.create_study(
    direction='maximize',  # максимизируем F1-score
    study_name='xgb_threshold_optimization',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)

print("Начинаем оптимизацию с Optuna...")
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Выводим результаты
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
print("="*60)

print(f"Лучшее значение F1-score: {study.best_value:.4f}")
print(f"\nЛучшие параметры:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Визуализация результатов
try:
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
except:
    print("Не удалось создать визуализацию истории")

# Получаем все trials для анализа
trials_df = study.trials_dataframe()
print(f"\nВсего выполнено trials: {len(trials_df)}")

# Создаем финальную модель с лучшими параметрами
best_params = study.best_params.copy()
threshold = best_params.pop('threshold')  # отделяем порог

# Обучаем финальную модель на всех данных
print("\n" + "="*60)
print("ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ:")
print("="*60)

final_model = ModelTrainer(XGBClassifier, best_params)
final_model.fit(X, y, threshold=threshold)

print(f"\nФинальные метрики на тестовой выборке:")
for metric, value in final_model.metrics.items():
    print(f"  {metric}: {value:.4f}")

# Анализ важности параметров
try:
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
except:
    print("\nАнализ важности параметров:")
    # Ручной анализ корреляций
    important_params = trials_df.corr()['value'].sort_values(ascending=False)
    print("Корреляция параметров с F1-score:")
    print(important_params.head(10))
best_params = {
    'n_estimators': 226,
    'max_depth': 7,
    'learning_rate': 0.01921106148185441,
    'subsample': 0.7915715432646518,
    'colsample_bytree': 0.9251010796404475,
    'min_child_weight': 5,
    'gamma': 0.10900791869271426,
    'reg_alpha': 0.01961633761408354,
    'reg_lambda': 0.006040374222520123,
    'scale_pos_weight': 5.787866022971578
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.61)
log_reg.fit(X, y, threshold = 0.6)
log_reg.fit(X, y, threshold = 0.599)
log_reg.fit(X, y, threshold = 0.59)
log_reg.fit(X, y, threshold = 0.6)
import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold

def optimized_objective(trial):
    """
    Оптимизированная функция с фокусом на важные параметры
    Порядок важности: threshold, scale_pos_weight, colsample_bytree, 
                     n_estimators, learning_rate, max_depth, gamma
    """
    # Параметры с разными диапазонами в зависимости от важности
    params = {
        # 1. САМЫЙ ВАЖНЫЙ - THRESHOLD
        'threshold': trial.suggest_float('threshold', 0.3, 0.9),
        
        # 2. ОЧЕНЬ ВАЖНЫЙ - SCALE_POS_WEIGHT
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 15),
        
        # 3. ОЧЕНЬ ВАЖНЫЙ - COLSAMPLE_BYTREE
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        
        # 4. ВАЖНЫЙ - N_ESTIMATORS
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        
        # 5. ВАЖНЫЙ - LEARNING_RATE
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        
        # 6. ВАЖНЫЙ - MAX_DEPTH
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        
        # 7. ВАЖНЫЙ - GAMMA
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        
        # 8. МЕНЕЕ ВАЖНЫЕ ПАРАМЕТРЫ (фиксированные или простые диапазоны)
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
    }
    
    # Отделяем threshold от других параметров XGBoost
    threshold = params.pop('threshold')
    
    # Кросс-валидация для стабильной оценки
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        try:
            model = ModelTrainer(XGBClassifier, params)
            model.fit(X_train_fold, y_train_fold, threshold=threshold)
            
            y_val_pred = (model.pipeline.predict_proba(X_val_fold)[:, 1] >= threshold).astype(int)
            f1 = f1_score(y_val_fold, y_val_pred)
            cv_scores.append(f1)
            
        except Exception as e:
            cv_scores.append(0.0)
            continue
    
    return np.mean(cv_scores) if cv_scores else 0.0

def create_focused_study(n_trials=100):
    """
    Создает и запускает оптимизацию с фокусировкой на важных параметрах
    """
    print(f"Запуск фокусированной оптимизации с {n_trials} trials...")
    print("Приоритет параметров: threshold > scale_pos_weight > colsample_bytree > n_estimators > learning_rate > max_depth > gamma")
    
    # Создаем study с учетом важности параметров
    study = optuna.create_study(
        direction='maximize',
        study_name='xgb_focused_tuning',
        sampler=optuna.samplers.TPESampler(
            seed=42,
            consider_prior=True,
            prior_weight=1.0,
            consider_magic_clip=True,
            consider_endpoints=False,
            n_startup_trials=20  # первые 20 trials - случайный поиск
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=3
        )
    )
    
    # Callback для отслеживания прогресса
    def print_progress(study, trial):
        if trial.number % 10 == 0:
            print(f"Trial {trial.number}: Best F1 = {study.best_value:.4f}")
    
    # Запускаем оптимизацию
    study.optimize(
        optimized_objective, 
        n_trials=n_trials,
        callbacks=[print_progress],
        show_progress_bar=True
    )
    
    return study

# Запуск оптимизации
study = create_focused_study(n_trials=80)

# Анализ результатов
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ ФОКУСИРОВАННОЙ ОПТИМИЗАЦИИ:")
print("="*60)

print(f"Лучшее значение F1-score: {study.best_value:.4f}")
print(f"\nЛучшие параметры (по важности):")

# Сортируем параметры по важности
important_params_order = [
    'threshold',
    'scale_pos_weight', 
    'colsample_bytree',
    'n_estimators',
    'learning_rate',
    'max_depth',
    'gamma',
    'subsample',
    'min_child_weight',
    'reg_alpha',
    'reg_lambda'
]

for param in important_params_order:
    if param in study.best_params:
        value = study.best_params[param]
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")

# Визуализация важности параметров
try:
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
except:
    print("\nНе удалось создать визуализацию важности параметров")

# Создаем финальную модель с лучшими параметрами
print("\n" + "="*60)
print("СОЗДАНИЕ ФИНАЛЬНОЙ МОДЕЛИ:")
print("="*60)

best_params_xgb = study.best_params.copy()
final_threshold = best_params_xgb.pop('threshold')

print(f"Параметры XGBoost:")
for key, value in best_params_xgb.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

print(f"\nОптимальный порог классификации: {final_threshold:.4f}")

# Обучаем финальную модель
final_model = ModelTrainer(XGBClassifier, best_params_xgb)
final_model.fit(X, y, threshold=final_threshold)

print(f"\nФинальные метрики на тестовой выборке:")
for metric, value in final_model.metrics.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.4f}")
    else:
        print(f"  {metric}: {value}")

# Дополнительно: анализ распределения параметров в лучших trials
print("\n" + "="*60)
print("АНАЛИЗ ЛУЧШИХ КОМБИНАЦИЙ ПАРАМЕТРОВ:")
print("="*60)

# Получаем топ-10 trials
trials_df = study.trials_dataframe()
trials_df = trials_df[trials_df['state'] == 'COMPLETE']
trials_df = trials_df.sort_values('value', ascending=False).head(10)

# Анализируем важные параметры в топ-10
important_params = ['threshold', 'scale_pos_weight', 'colsample_bytree', 
                    'n_estimators', 'learning_rate', 'max_depth', 'gamma']

print("\nСтатистика по важным параметрам в топ-10 trials:")
for param in important_params:
    if f'params_{param}' in trials_df.columns:
        values = trials_df[f'params_{param}']
        if param in ['threshold', 'scale_pos_weight', 'colsample_bytree', 
                    'learning_rate', 'gamma']:
            print(f"{param}: {values.mean():.4f} ± {values.std():.4f} "
                  f"(min={values.min():.4f}, max={values.max():.4f})")
        else:
            print(f"{param}: {values.mean():.1f} ± {values.std():.1f} "
                  f"(min={values.min()}, max={values.max()})")

# Возвращаем модель и study для дальнейшего использования
return final_model, study
best_params = {
    'n_estimators': 230,
    'max_depth': 7,
    'learning_rate': 0.01921106148185441,
    'subsample': 0.7915715432646518,
    'colsample_bytree': 0.9251010796404475,
    'min_child_weight': 5,
    'gamma': 0.10900791869271426,
    'reg_alpha': 0.01961633761408354,
    'reg_lambda': 0.006040374222520123,
    'scale_pos_weight': 5.787866022971578
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.6)
best_params = {
    'n_estimators': 230,
    'max_depth': 8,
    'learning_rate': 0.01921106148185441,
    'subsample': 0.7915715432646518,
    'colsample_bytree': 0.9251010796404475,
    'min_child_weight': 5,
    'gamma': 0.10900791869271426,
    'reg_alpha': 0.01961633761408354,
    'reg_lambda': 0.006040374222520123,
    'scale_pos_weight': 5.787866022971578
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.6)
best_params = {
    'n_estimators': 230,
    'max_depth': 8,
    'learning_rate': 0.01,
    'subsample': 0.7915715432646518,
    'colsample_bytree': 0.9251010796404475,
    'min_child_weight': 5,
    'gamma': 0.10900791869271426,
    'reg_alpha': 0.01961633761408354,
    'reg_lambda': 0.006040374222520123,
    'scale_pos_weight': 5.787866022971578
}
log_reg = ModelTrainer(XGBClassifier, xgb_params_basic, use_smote = False)
log_reg.fit(X, y, threshold = 0.6)
log_reg = ModelTrainer(XGBClassifier, best_params, use_smote = False)
log_reg.fit(X, y, threshold = 0.6)
best_params = {
    'n_estimators': 230,
    'max_depth': 8,
    'learning_rate': 0.01,
    'subsample': 0.7915715432646518,
    'colsample_bytree': 0.9251010796404475,
    'min_child_weight': 5,
    'gamma': 0.10900791869271426,
    'reg_alpha': 0.01961633761408354,
    'reg_lambda': 0.006040374222520123,
    'scale_pos_weight': 5.787866022971578
}
log_reg = ModelTrainer(XGBClassifier, best_params, use_smote = False)
log_reg.fit(X, y, threshold = 0.6)
best_params = {
    'n_estimators': 230,
    'max_depth': 8,
    'learning_rate': 0.02,
    'subsample': 0.7915715432646518,
    'colsample_bytree': 0.9251010796404475,
    'min_child_weight': 5,
    'gamma': 0.10900791869271426,
    'reg_alpha': 0.01961633761408354,
    'reg_lambda': 0.006040374222520123,
    'scale_pos_weight': 5.787866022971578
}
log_reg = ModelTrainer(XGBClassifier, best_params, use_smote = False)
log_reg.fit(X, y, threshold = 0.6)
best_params = {
    'n_estimators': 230,
    'max_depth': 7,
    'learning_rate': 0.02,
    'subsample': 0.7915715432646518,
    'colsample_bytree': 0.9251010796404475,
    'min_child_weight': 5,
    'gamma': 0.10900791869271426,
    'reg_alpha': 0.01961633761408354,
    'reg_lambda': 0.006040374222520123,
    'scale_pos_weight': 5.787866022971578
}
log_reg = ModelTrainer(XGBClassifier, best_params, use_smote = False)
log_reg.fit(X, y, threshold = 0.6)
log_reg = ModelTrainer(XGBClassifier, basic_params, use_smote = False)
%history -f history.py
