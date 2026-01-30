# Импорт необходимых библиотек

import numpy as np
import pandas as pd
import joblib
import json
import os
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from src.preprocessing import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    # 1. Загружаем данные
    df = pd.read_csv('data/bank-additional-full.csv', sep=';')
    
    # 2. Разделяем на X и y
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    X = df.drop('y', axis=1)
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

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
    
    print("\n" + "="*50)
    answer = input("Убрать из обучения признак Duration? (да/нет): ").lower().strip()
    print("\n" + "="*50)

    if answer in ['да', 'д', 'yes', 'y']:
        duration = False

    elif answer in ['нет', 'н', 'no', 'n']:
        duration = True

    else:
        print("Пожалуйста, введите 'да' или 'нет'")

    # 4. Настройка параметров в зависимости от выбора
    if choice == '1':  # XGBoost оптимизированный
        model_class = XGBClassifier
        model_name = "xgboost"
        model_params = {'n_estimators': 232,
                        'max_depth': 8,
                        'learning_rate': 0.01454877020944003,
                        'subsample': 0.9955153022026433,
                        'colsample_bytree': 0.9679592682343201,
                        'min_child_weight': 7,
                        'gamma': 0.34955546586648234,
                        'reg_alpha': 1.1082729972353083e-08,
                        'reg_lambda': 5.343890477972791e-07,
                         'scale_pos_weight': 2.773973433925954}
        optimal_threshold = 0.34
        
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
    
    # 5. Создаем папки если нет
    os.makedirs('models/preprocessor', exist_ok=True)   
    os.makedirs('models/metrics', exist_ok=True)       
    
    # 6 Cоздаем препроцессор 
    if duration:
        preproc = Preprocessor(duration = True)

    else:
        preproc = Preprocessor(duration= False)
    
    X_train = preproc.fit_transform(X_train)
    X_test = preproc.transform(X_test)

    # 7 Обучаем модель
    model = model_class(**model_params)
    model.fit(X_train, y_train)

    # 8. Сохраняем препроцессор в общую папку preprocessor
    import joblib
    preproc.save('models/preprocessor/')
    print('Препроцессор сохранен в models/preprocessor')
    # 9. Сохраняем модель в корень models

    joblib.dump(model, f'models/{model_name}.pkl')
    print("Модель сохранена в models")
    
    # 10. Сохраняем предикт модели
    y_pred = model.predict_proba(X_test)[:, 1]
    y_pred_thresholded = (y_pred >= optimal_threshold).astype(int)

    # 10.1 Метрики
    metrics = {}
    metrics['f1'] = f1_score(y_test, y_pred_thresholded)
    metrics['precision'] = precision_score(y_test, y_pred_thresholded)
    metrics['recall'] = recall_score(y_test, y_pred_thresholded)
    metrics['roc_auc'] = roc_auc_score(y_test, y_pred)
    metrics['threshold'] = optimal_threshold


    # Сохраняем метрики
    os.makedirs('models/metrics', exist_ok=True)
    with open('models/metrics/latest_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
        print(" Метрики сохранены в models/metrics/")
    
    # 11. Сохраняем параметры
    params_to_save = {
        'model_name': model_name,
        'model_params': model_params,
        'threshold': optimal_threshold,
    }
    
    with open('models/model_params.json', 'w') as f:
        json.dump(params_to_save, f, indent=4)
    print(" Параметры сохранены в models/model_params.json")
    
    # 12. Сохраняем важность признаков
    # Создаем DataFrame с важностью признаков
    importance_df = pd.DataFrame({
            'feature': model.feature_names_in_,
            'importance': model.feature_importances_}).sort_values('importance', ascending=False)

    # Создаем папку если ее нет
    os.makedirs('models/metrics', exist_ok=True)

    # Сохраняем в CSV
    importance_df.to_csv(  
        f'models/metrics/feature_importance_{model_name}.csv', index=False)

    print(f"Feature importance сохранен в models/metrics/feature_importance_{model_name}.csv")
    print(f" Топ-5 признаков:")
    print(importance_df.head())

    # 13. Итог
    print("\n" + "="*50)
    print(f" Модель: {model_name}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Threshold: {optimal_threshold:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()