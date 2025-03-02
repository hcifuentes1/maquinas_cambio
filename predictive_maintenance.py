# predictive_maintenance.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class PredictiveMaintenanceSystem:
    def __init__(self, db_path):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.config = self._load_config()
        
    def _load_config(self):
        """Cargar configuración de modelos desde archivo JSON"""
        try:
            with open('model_config.json') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {
                "model_types": {
                    "vibration": "lstm_autoencoder",
                    "electrical": "isolation_forest"
                },
                "retrain_interval": 86400,  # 24 horas
                "anomaly_thresholds": {
                    "vibration": 0.85,
                    "current": 0.75,
                    "voltage": 0.65
                }
            }
    
    def _create_lstm_autoencoder(self, input_shape):
        """Crea modelo LSTM Autoencoder para detección de anomalías"""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(input_shape[0], activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_models(self, machine_id):
        """Entrena modelos para una máquina específica"""
        try:
            data = self._load_training_data(machine_id)
            if data.empty:
                return False
            
            # Entrenar modelo para características eléctricas
            electrical_data = data[['voltaje', 'corriente']]
            self.scalers[machine_id] = RobustScaler().fit(electrical_data)
            scaled_data = self.scalers[machine_id].transform(electrical_data)
            
            # Modelo Isolation Forest
            iso_forest = IsolationForest(
                n_estimators=200,
                contamination=0.05,
                random_state=42
            )
            iso_forest.fit(scaled_data)
            self.models[f"{machine_id}_electrical"] = iso_forest
            
            # Modelo LSTM para vibración (datos secuenciales)
            if 'vibration' in data.columns:
                seq_data = self._create_sequences(data['vibration'].values)
                lstm_model = self._create_lstm_autoencoder((seq_data.shape[1], 1))
                lstm_model.fit(
                    seq_data, seq_data,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(patience=5)],
                    verbose=0
                )
                self.models[f"{machine_id}_vibration"] = lstm_model
            
            self._save_models(machine_id)
            return True
        except Exception as e:
            print(f"Error training models: {e}")
            return False
    
    def predict_anomalies(self, machine_id, real_time_data):
        """Predice anomalías en tiempo real"""
        try:
            results = {}
            
            # Preprocesamiento
            scaled_data = self.scalers[machine_id].transform(
                real_time_data[['voltaje', 'corriente']]
            )
            
            # Predicción eléctrica
            iso_pred = self.models[f"{machine_id}_electrical"].decision_function(scaled_data)
            results['electrical_anomaly'] = self._sigmoid(iso_pred)
            
            # Predicción de vibración (si existe modelo)
            if f"{machine_id}_vibration" in self.models:
                seq_data = self._create_sequences(real_time_data['vibration'])
                reconstruction = self.models[f"{machine_id}_vibration"].predict(seq_data)
                mse = np.mean(np.power(seq_data - reconstruction, 2), axis=1)
                results['vibration_anomaly'] = mse[-1]
            
            return self._interpret_results(results)
        except Exception as e:
            print(f"Prediction error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _interpret_results(self, raw_results):
        """Interpreta resultados brutos y genera recomendaciones"""
        recommendations = []
        health_status = "NORMAL"
        
        # Umbrales dinámicos del config
        thresholds = self.config['anomaly_thresholds']
        
        if raw_results.get('electrical_anomaly', 0) > thresholds['current']:
            recommendations.append("Revisar sistema eléctrico: Posible sobrecarga")
            health_status = "WARNING"
            
        if raw_results.get('vibration_anomaly', 0) > thresholds['vibration']:
            recommendations.append("Inspección mecánica requerida: Vibración anormal")
            health_status = "CRITICAL"
        
        return {
            "health_status": health_status,
            "recommendations": recommendations,
            "scores": raw_results
        }
    
    # Métodos auxiliares (data loading, sequence creation, etc.)
    # ... (implementar según necesidades específicas)