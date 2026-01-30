# Модуль создан для подготовки данных к обучению

import os
import pandas as pd
import numpy as np
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42, use_scaler = True, use_encoder = True, duration=False):
        
        self.random_state = random_state
        
        self.scaler = None      
        self.encoder = None
        self.use_scaler = use_scaler   
        self.use_encoder = use_encoder

        self.numeric_cols = None    
        self.cat_cols = None        
        self.feature_names = None 
        self.is_fitted = False
        self.duration = duration
    
    def fit(self, X, y=None):
        """
        Обучает препроцессор на данных.
        """
        X = X.copy()
        
        # 1. Удаляем duration если нет данных
        if self.duration == False:
            if 'duration' in X.columns:
                X = X.drop('duration', axis=1)
    
        # 2. Определяем типы колонок
        self._define_column_types(X)
        
        # 3. Обрабатываем редкие категории
        X = self._handle_rare_categories(X)
        
        # 4. Обучаем трансформеры
        self._train_transformers(X)
        
        # 5. Запоминаем имена фичей
        self._feature_names(X)

        #6. Меняем флаг обучения
        self.is_fitted = True

        return self
    
    def transform(self, X):      
        X = X.copy()

        if self.duration == False:
           if 'duration' in X.columns:
            X = X.drop('duration', axis=1) 

        # Обрабатываем редкие категории
        X = self._apply_rare_categories_transform(X)
        
        # Применяем One-Hot Encoding
        if self.cat_cols and self.encoder:
            X_encoded = self.encoder.transform(X[self.cat_cols])
            encoded_df = pd.DataFrame(
                X_encoded,
                columns=self.encoder.get_feature_names_out(self.cat_cols),
                index=X.index
            )
            X = pd.concat([X.drop(self.cat_cols, axis=1), encoded_df], axis=1)
        
        # Масштабируем числовые признаки
        if self.numeric_cols and self.scaler:
            X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        
        return X

    def _define_column_types(self, X):
        """
        Определяет какие колонки числовые, какие категориальные.
        """
        # Определяем числовые
        self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Категориальные
        categorical_col = [
            'job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week', 'poutcome'
        ]
        self.cat_cols = [
            col for col in categorical_col
            if col in X.columns and X[col].dtype == 'object'
        ]

    def _handle_rare_categories(self, X):

        """
        Объеденяем редкие категории
        """
        X = X.copy()

        if 'default' in X.columns:
            self.default_mapping = {'yes': 'unknown'}
            X['default'] = X['default'].replace(self.default_mapping)
    
        if 'education' in X.columns:
            self.education_mapping = {'illiterate': 'unknown'}
            X['education'] = X['education'].replace(self.education_mapping)
    
        if 'marital' in X.columns:
            self.marital_mapping = {'unknown': 'single'}
            X['marital'] = X['marital'].replace(self.marital_mapping)
    
        return X  

    def _apply_rare_categories_transform(self, X):
        """
        Применяем обработку редких категорий.
        """
        X = X.copy()
        
        if hasattr(self, 'default_mapping') and 'default' in X.columns:
            X['default'] = X['default'].replace(self.default_mapping)
        
        if hasattr(self, 'education_mapping') and 'education' in X.columns:
            X['education'] = X['education'].replace(self.education_mapping)
        
        if hasattr(self, 'marital_mapping') and 'marital' in X.columns:
            X['marital'] = X['marital'].replace(self.marital_mapping)

        return X

    def _train_transformers(self, X):

        # StandardScaler
        if self.use_scaler:
            self.scaler = StandardScaler()
            if self.numeric_cols:
                self.scaler.fit(X[self.numeric_cols])
        
        # OneHotEncoder
        if self.use_encoder:
            self.encoder = OneHotEncoder(sparse_output=False)
            if self.cat_cols:
                self.encoder.fit(X[self.cat_cols])
    
    def _feature_names(self, X):
        """
        Сохраняет имена фичей после преобразования.
        """
        feature_names = []
        
        # Числовые колонки
        if self.numeric_cols:
            feature_names.extend(self.numeric_cols)
        
        # Категориальные после OHE
        if self.cat_cols and self.encoder:
            encoded_names = list(self.encoder.get_feature_names_out(self.cat_cols))
            feature_names.extend(encoded_names)
        
        self.feature_names = feature_names

    def save(self, filepath):
        """
        Сохраняет препроцессор в файл.
        """
        import joblib

        joblib.dump(self, f'{filepath}preprocessor.joblib')

        joblib.dump(self.encoder, f'{filepath}encoder.pkl')

        joblib.dump(self.scaler, f'{filepath}scaler.pkl')