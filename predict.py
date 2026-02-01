import pandas as pd
import joblib
import json
from datetime import datetime
import os

def main(): 
    # Загружаем модель и препроцессор
    model = joblib.load('models/xgboost.pkl')
    preprocessor = joblib.load('models/preprocessor/preprocessor.joblib')
    
    # Загружаем порог из параметров
    with open('models/model_params.json', 'r') as f:
        params = json.load(f)
    threshold = params['threshold']
    
    # Загружаем данные для предсказания
    data_path = 'data/to_predict/predict_data.csv'
    df = pd.read_csv(data_path, sep=';')
    
    # Предобработка
    X_processed = preprocessor.transform(df)
    
    # Предсказания
    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    # Сохраняем результаты
    results = pd.DataFrame({
        'client_id': df.index,
        'probability': probabilities,
        'prediction': predictions,
        'recommend_contact': ['YES' if p == 1 else 'NO' for p in predictions]
    })
    
    # Создаем папку если нет
    os.makedirs('models/predicts', exist_ok=True)
    
    # Сохраняем
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f'models/predicts/predictions_{timestamp}.csv'
    results.to_csv(output_path, index=False)
    
    print(f"✅ Готово! Предсказания сохранены: {output_path}")
    print(f"📊 Статистика:")
    print(f"   Всего клиентов: {len(results)}")
    print(f"   Рекомендуем связаться: {predictions.sum()} ({predictions.mean():.1%})")

if __name__ == "__main__":
    main()