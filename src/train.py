# src/train.py
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

# Ваши модули
from src.preprocessing import Preprocessor
from src.model import ModelTrainer

def main():
    # 1. Загружаем данные
    print("Загружаем данные...")
    df = pd.read_csv('data/raw/bank-additional-full.csv', sep=';')
    
    # 2. Разделяем на X и y
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    X = df.drop('y', axis=1)
    y = df['y']
    
    print(f"Всего данных: {X.shape[0]} строк, {X.shape[1]} признаков")
    print(f"Баланс классов: {y.sum()} 'yes' ({y.mean():.2%})")
    
    # 3. Показываем меню выбора модели
    print("\n" + "="*50)
    print("🤖 ВЫБЕРИТЕ МОДЕЛЬ ДЛЯ ОБУЧЕНИЯ:")
    print("="*50)
    print("1. XGBoost (оптимизированные параметры) - РЕКОМЕНДУЕТСЯ")
    print("2. Random Forest")
    print("3. Logistic Regression")
    print("4. XGBoost (параметры по умолчанию)")
    print("="*50)
    
    # Получаем выбор пользователя
    choice = input()
    
    # 4. Настройка параметров в зависимости от выбора
    if choice == '1':  # XGBoost оптимизированный
        model_class = XGBClassifier
        model_name = "xgboost"
        model_params = {
            'n_estimators': 226,
            'max_depth': 7,
            'learning_rate': 0.01921106148185441,
            'subsample': 0.7915715432646518,
            'colsample_bytree': 0.9251010796404475,
            'min_child_weight': 5,
            'gamma': 0.10900791869271426,
            'reg_alpha': 0.01961633761408354,
            'reg_lambda': 0.006040374222520123,
            'scale_pos_weight': 5.787866022971578,
            'random_state': 42
        }
        optimal_threshold = 0.5996857774477184
        
    elif choice == '2':  # Random Forest
        model_class = RandomForestClassifier
        model_name = "random_forest"
        model_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        optimal_threshold = 0.5

        
    elif choice == '3':  # Logistic Regression
        model_class = LogisticRegression
        model_name = "logistic_regression"
        model_params = {
            'C': 0.1,
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'liblinear'
        }
        optimal_threshold = 0.67

    else:  # XGBoost по умолчанию 
        model_class = XGBClassifier
        model_name = "xgboost_default"
        model_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
            'random_state': 42
        }
        optimal_threshold = 0.6
    
    print(f"\n✅ Выбрана модель: {model_name}")
    
    # 5. Создаем общие папки
    os.makedirs('models/preprocessor', exist_ok=True)   # Общая папка для препроцессора
    os.makedirs('models/metrics', exist_ok=True)        # Общая папка для метрик
    
    # 6. Обучаем модель
    pipline = 
    
    # 7. Обучение
    trainer.fit(X, y, test_size=0.2, threshold=optimal_threshold)
    
    # 8. Сохраняем препроцессор в общую папку preprocessor
    trainer.preprocessor.save('models/preprocessor/')
    print("✅ Препроцессор сохранен в models/preprocessor/")
    
    # 9. Сохраняем модель в корень models
    model_filename = f'{model_name}_model.joblib'
    trainer.save(f'models/{model_filename}')
    print(f"✅ Модель сохранена как models/{model_filename}")
    
    # 10. Сохраняем метрики в общую папку metrics
    metrics = trainer.metrics.copy()
    metrics['model_name'] = model_name
    metrics['model_class'] = model_class.__name__
    
    # Метрики
    with open('models/metrics/latest_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
        print("✅ Метрики сохранены в models/metrics/")
    
    # 11. Сохраняем параметры
    params_to_save = {
        'model_name': model_name,
        'model_params': model_params,
        'threshold': optimal_threshold,
    }
    
    with open('models/metrics/model_params.json', 'w') as f:
        json.dump(params_to_save, f, indent=4)
    print("✅ Параметры сохранены в models/metrics/model_params.json")
    
    # 12. Сохраняем важность признаков

    feature_importance = trainer.get_feature_importance()
    print(f"Топ-5 признаков для {model_name}:")
    print(feature_importance.head().to_string())

    feature_importance.to_csv(
        f'models/metrics/feature_importance_{model_name}.csv', 
        index=False)

    # 13. Итог
    print("\n" + "="*50)
    print(f"🤖 Модель: {model_name}")
    print(f"📊 F1-score: {trainer.metrics['f1']:.4f}")
    print(f"📈 ROC-AUC:  {trainer.metrics['roc_auc']:.4f}")
    print(f"🎯 Threshold: {optimal_threshold:.4f}")
    print("="*50)