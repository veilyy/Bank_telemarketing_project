# Модуль обучения модели

# Импорт библиотек
import os
import sys
sys.path.append('..')
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from src.preprocessing import Preprocessor
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

class ModelTrainer():
    """
    Класс для обучения моделей.
    Принимает класс модели и параметры.
    """
    
    def __init__(self, model_class, model_params=None, use_scaler = True, use_encoder = True, random_state = 42, use_smote = False, smote_strategy = 0.3):
        """
        Parameters:
        -----------
        model_class : Класс модели

        model_params : Параметры для модели
        
        random_state : Воспроизводимость
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.random_state = random_state
        self.use_smote = use_smote
        self.smote_strategy = smote_strategy
        self.use_encoder = use_encoder
        self.use_scaler = use_scaler
        
                
        self.model = model_class(**self.model_params)
    
    def fit(self, X, y, test_size=0.3, threshold = 0.5):

        self.threshold = threshold
        
        # Обучение
        self.pipeline.fit(X_train, y_train)

          # Сохраняем вероятности для тестовой выборки
        self.test_probas = self.pipeline.predict_proba(X_test)[:, 1]

        # Оценка
        self.metrics = self._calculate_metrics(threshold)
        self._print_metrics()
        
        return self


    def _calculate_metrics(self, threshold):
        y_pred = (self.test_probas >= threshold).astype(int)
        
        return {
            'f1': float(f1_score(self.y_test, y_pred)),
            'precision': float(precision_score(self.y_test, y_pred)),
            'recall': float(recall_score(self.y_test, y_pred)),
            'roc_auc': float(roc_auc_score(self.y_test, self.test_probas)),
            'accuracy': float(accuracy_score(self.y_test, y_pred)),
            'threshold': threshold
        }

    def get_feature_importance(self):
        """
        Возвращает DataFrame c важностью признаков.
        """

        feature_names = self.preprocessor.feature_names
                
        # Для линейных моделей
        if hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        
        # Для моделей с feature_importances_
        elif hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
    
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importances': importances

        })
        
        # Сортируем по важности
        importance_df = importance_df.sort_values('importances', ascending=False)
        
        # Добавляем ранг
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df.reset_index(drop=True)

    def predict(self, X):
        """Предсказания"""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Вероятности"""
        return self.pipeline.predict_proba(X)
    
    def save(self, path):
        joblib.dump(self.pipeline, path)

    def _print_metrics(self):
        if self.metrics:
            print(f"\nМетрики модели (threshold={self.metrics['threshold']:.3f}):")
            print(f"F1-score: {self.metrics['f1']:.4f}")
            print(f"Precision: {self.metrics['precision']:.4f}")
            print(f"Recall: {self.metrics['recall']:.4f}")
            print(f"Accuracy: {self.metrics['accuracy']:.4f}")