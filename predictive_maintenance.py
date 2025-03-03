# predictive_maintenance.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
# Importaciones actualizadas de TensorFlow
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks
import joblib
import os
import json
import warnings
from datetime import datetime, timedelta
import sqlite3


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
        model = keras.Sequential([
            layers.LSTM(64, input_shape=input_shape, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(input_shape[0], activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _load_training_data(self, machine_id):
        """Carga datos de entrenamiento para una máquina específica"""
        try:
            # Calcular ventana de tiempo para entrenamiento (últimos 30 días)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
            
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT timestamp, voltaje, corriente, posicion, ciclo_progreso
                FROM mediciones
                WHERE maquina_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=(
                    machine_id,
                    start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    end_time.strftime('%Y-%m-%d %H:%M:%S')
                )
            )
            
            conn.close()
            
            if len(df) < 100:
                print(f"Datos insuficientes para entrenamiento: {len(df)} filas")
                return pd.DataFrame()
                
            # Convertir timestamp a datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Agregar características derivadas
            df['hora_dia'] = df['timestamp'].dt.hour
            df['dia_semana'] = df['timestamp'].dt.dayofweek
            
            # Calcular características de voltaje y corriente
            for col in ['voltaje', 'corriente']:
                # Calcular diferencias
                df[f'{col}_delta'] = df[col].diff()
                # Calcular promedio móvil
                df[f'{col}_rolling_mean'] = df[col].rolling(window=20).mean()
                # Calcular desviación estándar móvil
                df[f'{col}_rolling_std'] = df[col].rolling(window=20).std()
            
            # Llenar valores NaN con 0
            df.fillna(0, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error cargando datos de entrenamiento: {e}")
            return pd.DataFrame()

    def _create_sequences(self, data, seq_length=20):
        """Crea secuencias para entrenamiento de LSTM"""
        result = []
        for i in range(len(data) - seq_length):
            result.append(data[i:i + seq_length])
        
        return np.array(result).reshape(-1, seq_length, 1)

    def _sigmoid(self, x):
        """Función sigmoide para normalizar scores"""
        return 1 / (1 + np.exp(-x))

    def _save_models(self, machine_id):
        """Guarda los modelos entrenados para una máquina"""
        try:
            # Crear directorio para modelos si no existe
            models_dir = os.path.join(os.path.dirname(self.db_path), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Guardar modelos
            if f"{machine_id}_electrical" in self.models:
                joblib.dump(
                    self.models[f"{machine_id}_electrical"],
                    os.path.join(models_dir, f"{machine_id}_electrical.pkl")
                )
            
            if f"{machine_id}_vibration" in self.models:
                self.models[f"{machine_id}_vibration"].save(
                    os.path.join(models_dir, f"{machine_id}_vibration.h5")
                )
            
            # Guardar scaler
            if machine_id in self.scalers:
                joblib.dump(
                    self.scalers[machine_id],
                    os.path.join(models_dir, f"{machine_id}_scaler.pkl")
                )
                
            print(f"Modelos guardados para {machine_id}")
            return True
        except Exception as e:
            print(f"Error guardando modelos: {e}")
            return False

    def load_models(self, machine_id):
        """Carga modelos pre-entrenados para una máquina"""
        try:
            import os
            import joblib
            
            models_dir = os.path.join(os.path.dirname(self.db_path), 'models')
            
            # Cargar modelo eléctrico
            electrical_model_path = os.path.join(models_dir, f"{machine_id}_electrical.pkl")
            if os.path.exists(electrical_model_path):
                self.models[f"{machine_id}_electrical"] = joblib.load(electrical_model_path)
            
            # Cargar modelo de vibración
            vibration_model_path = os.path.join(models_dir, f"{machine_id}_vibration.h5")
            if os.path.exists(vibration_model_path):
                self.models[f"{machine_id}_vibration"] = keras.models.load_model(vibration_model_path)
            
            # Cargar scaler
            scaler_path = os.path.join(models_dir, f"{machine_id}_scaler.pkl")
            if os.path.exists(scaler_path):
                self.scalers[machine_id] = joblib.load(scaler_path)
                
            return True
        except Exception as e:
            print(f"Error cargando modelos: {e}")
            return False
    
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
                    callbacks=[callbacks.EarlyStopping(patience=5)],
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
            if f"{machine_id}_vibration" in self.models and 'vibration' in real_time_data.columns:
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
        
    def get_machine_health(self, machine_id, recent_data=None):
        """
        Obtiene el estado de salud de una máquina basado en datos recientes
        
        Args:
            machine_id: ID de la máquina
            recent_data: DataFrame con datos recientes (opcional)
            
        Returns:
            dict: Información del estado de salud y recomendaciones
        """
        try:
            # Si no se proporcionan datos recientes, obtenerlos de la base de datos
            if recent_data is None:
                conn = sqlite3.connect(self.db_path)
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=1)
                
                query = """
                    SELECT timestamp, voltaje, corriente, posicion, ciclo_progreso
                    FROM mediciones
                    WHERE maquina_id = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                """
                
                recent_data = pd.read_sql_query(
                    query,
                    conn,
                    params=(
                        machine_id,
                        start_time.strftime('%Y-%m-%d %H:%M:%S'),
                        end_time.strftime('%Y-%m-%d %H:%M:%S')
                    )
                )
                
                conn.close()
            
            if recent_data.empty:
                return {
                    "status": "unknown",
                    "message": "No hay datos suficientes para análisis",
                    "health_score": 0,
                    "recommendations": [],
                    "maintenance_due": "unknown"
                }
            
            # Cargar modelos si no están ya cargados
            if (f"{machine_id}_electrical" not in self.models or 
                machine_id not in self.scalers):
                self.load_models(machine_id)
                
                # Si no hay modelos entrenados, entrenar con datos disponibles
                if (f"{machine_id}_electrical" not in self.models or 
                    machine_id not in self.scalers):
                    print(f"Entrenando nuevos modelos para {machine_id}")
                    self.train_models(machine_id)
                    
                    # Si aún no hay modelos, no podemos continuar
                    if (f"{machine_id}_electrical" not in self.models or 
                        machine_id not in self.scalers):
                        return {
                            "status": "unknown",
                            "message": "No se pudieron cargar o entrenar modelos",
                            "health_score": 0,
                            "recommendations": ["Revisar sistema de ML"],
                            "maintenance_due": "unknown"
                        }
            
            # Preparar datos para análisis
            electrical_data = recent_data[['voltaje', 'corriente']].iloc[-5:].mean().to_frame().T
            
            # Usar modelo eléctrico para predecir anomalías
            scaled_data = self.scalers[machine_id].transform(electrical_data)
            anomaly_score = self.models[f"{machine_id}_electrical"].decision_function(scaled_data)[0]
            normalized_score = self._sigmoid(anomaly_score)
            
            # Análisis de patrones básicos
            voltaje_std = recent_data['voltaje'].std()
            corriente_std = recent_data['corriente'].std()
            
            # Calcular ciclos completos
            ciclos_completos = len(recent_data[recent_data['ciclo_progreso'] < recent_data['ciclo_progreso'].shift(-1)])
            
            # Determinar estado y recomendaciones
            health_score = normalized_score * 100
            recommendations = []
            
            if health_score < 60:
                status = "critical"
                recommendations.append("Inspección urgente requerida - anomalías detectadas")
                if voltaje_std > 10:
                    recommendations.append("Fluctuaciones anormales de voltaje")
                if corriente_std > 1.5:
                    recommendations.append("Consumo irregular de corriente - revisar motor")
                maintenance_due = "inmediato"
            elif health_score < 80:
                status = "warning"
                recommendations.append("Programar inspección preventiva")
                if voltaje_std > 7:
                    recommendations.append("Fluctuaciones de voltaje - revisar alimentación")
                if corriente_std > 1.0:
                    recommendations.append("Variaciones de corriente - comprobar desgaste")
                maintenance_due = "próximos 7 días"
            else:
                status = "good"
                recommendations.append("Sistema funcionando correctamente")
                if ciclos_completos > 1000:
                    recommendations.append("Considerar mantenimiento preventivo por número de ciclos")
                    maintenance_due = "próximos 30 días"
                else:
                    maintenance_due = f"después de {1000 - ciclos_completos} ciclos"
            
            # Añadir métricas adicionales
            electrical_metrics = {
                "voltaje_avg": recent_data['voltaje'].mean(),
                "voltaje_std": voltaje_std,
                "corriente_avg": recent_data['corriente'].mean(),
                "corriente_std": corriente_std,
                "ciclos_recientes": ciclos_completos
            }
            
            return {
                "status": status,
                "message": f"Estado de salud: {health_score:.1f}%",
                "health_score": health_score,
                "recommendations": recommendations,
                "maintenance_due": maintenance_due,
                "metrics": electrical_metrics,
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            import traceback
            print(f"Error en análisis de salud: {e}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Error en análisis: {str(e)}",
                "health_score": 0,
                "recommendations": ["Error en sistema de análisis"],
                "maintenance_due": "unknown"
            }