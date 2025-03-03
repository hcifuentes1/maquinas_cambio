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
        
# Añadir esta función a la clase PredictiveMaintenanceSystem (en el archivo predictive_maintenance.py)

    def _load_config(self):
        """Cargar configuración de modelos desde archivo JSON"""
        try:
            import os
            import json
            
            config_path = 'model_config.json'
            
            # Si el archivo no existe, crearlo con valores predeterminados
            if not os.path.exists(config_path):
                default_config = {
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
                
                try:
                    with open(config_path, 'w') as f:
                        json.dump(default_config, f, indent=4)
                    print(f"Archivo de configuración creado: {config_path}")
                    return default_config
                except Exception as e:
                    print(f"Error al crear archivo de configuración: {e}")
                    return default_config
            
            # Cargar el archivo si existe
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            # Valores predeterminados en caso de error
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
        """Carga datos de entrenamiento para una máquina específica con nuevos parámetros"""
        try:
            # Calcular ventana de tiempo para entrenamiento (últimos 30 días)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
            
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT timestamp, corriente_f1, corriente_f2, corriente_f3, 
                    voltaje_ctrl_izq, voltaje_ctrl_der, posicion, ciclo_progreso
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
            
            # Calcular características para cada corriente y voltaje
            for col in ['corriente_f1', 'corriente_f2', 'corriente_f3', 
                        'voltaje_ctrl_izq', 'voltaje_ctrl_der']:
                # Calcular diferencias
                df[f'{col}_delta'] = df[col].diff()
                # Calcular promedio móvil
                df[f'{col}_rolling_mean'] = df[col].rolling(window=20).mean()
                # Calcular desviación estándar móvil
                df[f'{col}_rolling_std'] = df[col].rolling(window=20).std()
            
            # Calcular características de desbalance entre fases
            corriente_promedio = (df['corriente_f1'] + df['corriente_f2'] + df['corriente_f3']) / 3
            df['desbalance_f1'] = (df['corriente_f1'] - corriente_promedio).abs() / corriente_promedio * 100
            df['desbalance_f2'] = (df['corriente_f2'] - corriente_promedio).abs() / corriente_promedio * 100
            df['desbalance_f3'] = (df['corriente_f3'] - corriente_promedio).abs() / corriente_promedio * 100
            df['desbalance_max'] = df[['desbalance_f1', 'desbalance_f2', 'desbalance_f3']].max(axis=1)
            
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
            import os
            import joblib
            
            # Crear directorio para modelos si no existe
            models_dir = os.path.join(os.path.dirname(self.db_path), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Limpiar caracteres problemáticos del nombre de archivo (por ejemplo '/')
            safe_machine_id = machine_id.replace('/', '_').replace('\\', '_')
            
            # Guardar modelos
            if f"{machine_id}_electrical" in self.models:
                try:
                    joblib.dump(
                        self.models[f"{machine_id}_electrical"],
                        os.path.join(models_dir, f"{safe_machine_id}_electrical.pkl")
                    )
                    print(f"Modelo eléctrico guardado para {machine_id}")
                except Exception as e:
                    print(f"Error guardando modelo eléctrico: {e}")
            
            if f"{machine_id}_sequence" in self.models:
                try:
                    self.models[f"{machine_id}_sequence"].save(
                        os.path.join(models_dir, f"{safe_machine_id}_sequence.h5")
                    )
                    print(f"Modelo de secuencia guardado para {machine_id}")
                except Exception as e:
                    print(f"Error guardando modelo de secuencia: {e}")
            
            # Guardar scaler
            if machine_id in self.scalers:
                try:
                    joblib.dump(
                        self.scalers[machine_id],
                        os.path.join(models_dir, f"{safe_machine_id}_scaler.pkl")
                    )
                    print(f"Scaler guardado para {machine_id}")
                except Exception as e:
                    print(f"Error guardando scaler: {e}")
                
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
            os.makedirs(models_dir, exist_ok=True)  # Crear directorio si no existe
            
            # Limpiar caracteres problemáticos del nombre de archivo (por ejemplo '/')
            safe_machine_id = machine_id.replace('/', '_').replace('\\', '_')
            
            # Cargar modelo eléctrico
            electrical_model_path = os.path.join(models_dir, f"{safe_machine_id}_electrical.pkl")
            if os.path.exists(electrical_model_path):
                try:
                    self.models[f"{machine_id}_electrical"] = joblib.load(electrical_model_path)
                    print(f"Modelo eléctrico cargado: {electrical_model_path}")
                except Exception as e:
                    print(f"Error cargando modelo eléctrico: {e}")
                    if os.path.exists(electrical_model_path):
                        # Si hay error al cargar, eliminar el modelo corrupto
                        try:
                            os.remove(electrical_model_path)
                            print(f"Modelo corrupto eliminado: {electrical_model_path}")
                        except:
                            pass
            else:
                print(f"Modelo eléctrico no encontrado: {electrical_model_path}")
            
            # Cargar modelo de vibración/secuencia
            vibration_model_path = os.path.join(models_dir, f"{safe_machine_id}_sequence.h5")
            if os.path.exists(vibration_model_path):
                try:
                    self.models[f"{machine_id}_sequence"] = keras.models.load_model(vibration_model_path)
                    print(f"Modelo de secuencia cargado: {vibration_model_path}")
                except Exception as e:
                    print(f"Error cargando modelo de secuencia: {e}")
                    if os.path.exists(vibration_model_path):
                        # Si hay error al cargar, eliminar el modelo corrupto
                        try:
                            os.remove(vibration_model_path)
                            print(f"Modelo corrupto eliminado: {vibration_model_path}")
                        except:
                            pass
            
            # Cargar scaler
            scaler_path = os.path.join(models_dir, f"{safe_machine_id}_scaler.pkl")
            if os.path.exists(scaler_path):
                try:
                    self.scalers[machine_id] = joblib.load(scaler_path)
                    print(f"Scaler cargado: {scaler_path}")
                except Exception as e:
                    print(f"Error cargando scaler: {e}")
                    if os.path.exists(scaler_path):
                        try:
                            os.remove(scaler_path)
                            print(f"Scaler corrupto eliminado: {scaler_path}")
                        except:
                            pass
                    
            return True
        except Exception as e:
            print(f"Error general cargando modelos: {e}")
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
        
    def _create_multidimensional_sequences(self, data, seq_length=20):
        """Crea secuencias multidimensionales para entrenamiento de LSTM"""
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
        
        return np.array(sequences)
        
    def get_machine_health(self, machine_id, recent_data=None):
        """
        Obtiene el estado de salud de una máquina basado en datos recientes
        con los nuevos parámetros (3 fases y 2 controladores)
        
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
                
                # Obtener la estructura actual de la tabla
                c = conn.cursor()
                c.execute("PRAGMA table_info(mediciones)")
                columns = [col[1] for col in c.fetchall()]
                
                # Consulta adaptada a la estructura actual
                if 'corriente_f1' in columns:
                    query = """
                        SELECT timestamp, corriente_f1, corriente_f2, corriente_f3, 
                            voltaje_ctrl_izq, voltaje_ctrl_der, posicion, ciclo_progreso
                        FROM mediciones
                        WHERE maquina_id = ? AND timestamp BETWEEN ? AND ?
                        ORDER BY timestamp DESC
                        LIMIT 100
                    """
                else:
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
                
                # Si estamos en estructura antigua, convertir a nueva
                if 'corriente_f1' not in recent_data.columns and not recent_data.empty:
                    print("Convirtiendo datos de estructura antigua a nueva")
                    recent_data['corriente_f1'] = recent_data['corriente']
                    recent_data['corriente_f2'] = recent_data['corriente']
                    recent_data['corriente_f3'] = recent_data['corriente']
                    recent_data['voltaje_ctrl_izq'] = recent_data['voltaje'] / 10
                    recent_data['voltaje_ctrl_der'] = recent_data['voltaje'] / 10
                
                conn.close()
            
            if recent_data.empty:
                return {
                    "status": "unknown",
                    "message": "No hay datos suficientes para análisis",
                    "health_score": 0,
                    "recommendations": [],
                    "maintenance_due": "unknown"
                }
            
            # NUEVA LÓGICA: Si detectamos un cambio en la estructura de datos, 
            # forzamos un reentrenamiento
            needs_retraining = False
            
            if machine_id in self.scalers:
                # Verificar si el scaler existente es compatible con los nuevos datos
                try:
                    sample_data = pd.DataFrame({
                        'corriente_f1': [recent_data['corriente_f1'].iloc[-1]],
                        'corriente_f2': [recent_data['corriente_f2'].iloc[-1]],
                        'corriente_f3': [recent_data['corriente_f3'].iloc[-1]],
                        'voltaje_ctrl_izq': [recent_data['voltaje_ctrl_izq'].iloc[-1]],
                        'voltaje_ctrl_der': [recent_data['voltaje_ctrl_der'].iloc[-1]]
                    })
                    self.scalers[machine_id].transform(sample_data)
                except (ValueError, KeyError) as e:
                    print(f"Estructura de datos cambiada, forzando reentrenamiento: {e}")
                    needs_retraining = True
            
            # Cargar modelos si no están ya cargados o si necesitamos reentrenar
            if needs_retraining or f"{machine_id}_electrical" not in self.models or machine_id not in self.scalers:
                # Si necesitamos reentrenar, eliminamos los modelos existentes
                if needs_retraining:
                    if machine_id in self.scalers:
                        del self.scalers[machine_id]
                    if f"{machine_id}_electrical" in self.models:
                        del self.models[f"{machine_id}_electrical"]
                    if f"{machine_id}_sequence" in self.models:
                        del self.models[f"{machine_id}_sequence"]
                    print("Modelos anteriores eliminados para reentrenamiento")
                
                # Intentar cargar modelos (si existen y son compatibles)
                if not needs_retraining:
                    self.load_models(machine_id)
                
                # Si aún no tenemos modelos, entrenar con datos disponibles
                if f"{machine_id}_electrical" not in self.models or machine_id not in self.scalers:
                    print(f"Entrenando nuevos modelos para {machine_id}")
                    self.train_models(machine_id)
                    
                    # Si aún no hay modelos, proporcionar un análisis simple
                    if f"{machine_id}_electrical" not in self.models or machine_id not in self.scalers:
                        # Análisis simple sin ML
                        return self._simple_health_analysis(machine_id, recent_data)
            
            # Preparar datos para análisis - usando promedios de las últimas 5 muestras
            electrical_data = pd.DataFrame({
                'corriente_f1': [recent_data['corriente_f1'].iloc[-5:].mean()],
                'corriente_f2': [recent_data['corriente_f2'].iloc[-5:].mean()],
                'corriente_f3': [recent_data['corriente_f3'].iloc[-5:].mean()],
                'voltaje_ctrl_izq': [recent_data['voltaje_ctrl_izq'].iloc[-5:].mean()],
                'voltaje_ctrl_der': [recent_data['voltaje_ctrl_der'].iloc[-5:].mean()]
            })
            
            # Usar modelo eléctrico para predecir anomalías
            scaled_data = self.scalers[machine_id].transform(electrical_data)
            anomaly_score = self.models[f"{machine_id}_electrical"].decision_function(scaled_data)[0]
            normalized_score = self._sigmoid(anomaly_score)
            
            # Análisis de patrones básicos
            corriente_f1_std = recent_data['corriente_f1'].std()
            corriente_f2_std = recent_data['corriente_f2'].std()
            corriente_f3_std = recent_data['corriente_f3'].std()
            voltaje_ctrl_izq_std = recent_data['voltaje_ctrl_izq'].std()
            voltaje_ctrl_der_std = recent_data['voltaje_ctrl_der'].std()
            
            # Calcular desbalance de corrientes
            corriente_promedio = (
                recent_data['corriente_f1'].mean() + 
                recent_data['corriente_f2'].mean() + 
                recent_data['corriente_f3'].mean()
            ) / 3
            
            desbalance_f1 = abs(recent_data['corriente_f1'].mean() - corriente_promedio) / corriente_promedio * 100 if corriente_promedio > 0 else 0
            desbalance_f2 = abs(recent_data['corriente_f2'].mean() - corriente_promedio) / corriente_promedio * 100 if corriente_promedio > 0 else 0
            desbalance_f3 = abs(recent_data['corriente_f3'].mean() - corriente_promedio) / corriente_promedio * 100 if corriente_promedio > 0 else 0
            
            desbalance_max = max(desbalance_f1, desbalance_f2, desbalance_f3)
            fase_desbalanceada = ['F1', 'F2', 'F3'][[desbalance_f1, desbalance_f2, desbalance_f3].index(desbalance_max)]
            
            # Calcular ciclos completos
            ciclos_completos = len(recent_data[recent_data['ciclo_progreso'] < recent_data['ciclo_progreso'].shift(-1)]) if len(recent_data) > 1 else 0
            
            # Determinar estado y recomendaciones
            health_score = normalized_score * 100
            recommendations = []
            
            # Verificar desbalance de fases
            desbalance_critico = False
            if desbalance_max > 25:  # Más de 25% de desbalance
                desbalance_critico = True
                recommendations.append(f"Desbalance crítico en fase {fase_desbalanceada} ({desbalance_max:.1f}%)")
            elif desbalance_max > 15:  # Más de 15% de desbalance
                recommendations.append(f"Desbalance significativo en fase {fase_desbalanceada} ({desbalance_max:.1f}%)")
            
            # Verificar fluctuaciones anormales
            fluctuacion_critica = False
            if corriente_f1_std > 1.5 or corriente_f2_std > 1.5 or corriente_f3_std > 1.5:
                fluctuacion_critica = True
                fase_fluctuante = ['F1', 'F2', 'F3'][[corriente_f1_std, corriente_f2_std, corriente_f3_std].index(
                    max(corriente_f1_std, corriente_f2_std, corriente_f3_std))]
                recommendations.append(f"Fluctuaciones elevadas en corriente {fase_fluctuante}")
            
            # Verificar controladores
            ctrl_critico = False
            if voltaje_ctrl_izq_std > 1.0 or voltaje_ctrl_der_std > 1.0:
                ctrl_critico = True
                ctrl_afectado = "izquierdo" if voltaje_ctrl_izq_std > voltaje_ctrl_der_std else "derecho"
                recommendations.append(f"Inestabilidad en controlador {ctrl_afectado}")
            
            # Determinar estado general
            if health_score < 60 or desbalance_critico or fluctuacion_critica or ctrl_critico:
                status = "critical"
                recommendations.insert(0, "Inspección urgente requerida - anomalías detectadas")
                maintenance_due = "inmediato"
            elif health_score < 80 or desbalance_max > 15 or max(corriente_f1_std, corriente_f2_std, corriente_f3_std) > 1.0:
                status = "warning"
                recommendations.insert(0, "Programar inspección preventiva")
                maintenance_due = "próximos 7 días"
            else:
                status = "good"
                recommendations.append("Sistema funcionando correctamente")
                if ciclos_completos > 1000:
                    recommendations.append("Considerar mantenimiento preventivo por número de ciclos")
                    maintenance_due = "próximos 30 días"
                else:
                    maintenance_due = f"después de {1000 - ciclos_completos} ciclos" if ciclos_completos < 1000 else "inmediato"
            
            # Añadir métricas adicionales
            metrics = {
                "corriente_f1": {
                    "promedio": recent_data['corriente_f1'].mean(),
                    "std": corriente_f1_std,
                    "desbalance": desbalance_f1
                },
                "corriente_f2": {
                    "promedio": recent_data['corriente_f2'].mean(),
                    "std": corriente_f2_std,
                    "desbalance": desbalance_f2
                },
                "corriente_f3": {
                    "promedio": recent_data['corriente_f3'].mean(),
                    "std": corriente_f3_std,
                    "desbalance": desbalance_f3
                },
                "voltaje_ctrl_izq": {
                    "promedio": recent_data['voltaje_ctrl_izq'].mean(),
                    "std": voltaje_ctrl_izq_std
                },
                "voltaje_ctrl_der": {
                    "promedio": recent_data['voltaje_ctrl_der'].mean(),
                    "std": voltaje_ctrl_der_std
                },
                "ciclos_recientes": ciclos_completos
            }
            
            return {
                "status": status,
                "message": f"Estado de salud: {health_score:.1f}%",
                "health_score": health_score,
                "recommendations": recommendations,
                "maintenance_due": maintenance_due,
                "metrics": metrics,
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            import traceback
            print(f"Error en análisis de salud: {e}")
            traceback.print_exc()
            return self._simple_health_analysis(machine_id, recent_data)
            
    def _simple_health_analysis(self, machine_id, recent_data):
        """Realiza un análisis simple basado en reglas cuando el ML no está disponible"""
        try:
            # Análisis de patrones básicos
            corriente_f1_avg = recent_data['corriente_f1'].mean()
            corriente_f2_avg = recent_data['corriente_f2'].mean()
            corriente_f3_avg = recent_data['corriente_f3'].mean()
            
            corriente_f1_std = recent_data['corriente_f1'].std()
            corriente_f2_std = recent_data['corriente_f2'].std()
            corriente_f3_std = recent_data['corriente_f3'].std()
            
            # Calcular desbalance de corrientes
            corriente_promedio = (corriente_f1_avg + corriente_f2_avg + corriente_f3_avg) / 3
            
            desbalance_f1 = abs(corriente_f1_avg - corriente_promedio) / corriente_promedio * 100 if corriente_promedio > 0 else 0
            desbalance_f2 = abs(corriente_f2_avg - corriente_promedio) / corriente_promedio * 100 if corriente_promedio > 0 else 0
            desbalance_f3 = abs(corriente_f3_avg - corriente_promedio) / corriente_promedio * 100 if corriente_promedio > 0 else 0
            
            desbalance_max = max(desbalance_f1, desbalance_f2, desbalance_f3)
            fase_desbalanceada = ['F1', 'F2', 'F3'][[desbalance_f1, desbalance_f2, desbalance_f3].index(desbalance_max)]
            
            # Determinar estado y recomendaciones
            recommendations = []
            
            # Verificar desbalance de fases
            if desbalance_max > 25:  # Más de 25% de desbalance
                recommendations.append(f"Desbalance crítico en fase {fase_desbalanceada} ({desbalance_max:.1f}%)")
                status = "critical"
                health_score = 50
            elif desbalance_max > 15:  # Más de 15% de desbalance
                recommendations.append(f"Desbalance significativo en fase {fase_desbalanceada} ({desbalance_max:.1f}%)")
                status = "warning"
                health_score = 70
            else:
                status = "good"
                health_score = 90
                
            # Verificar fluctuaciones
            if corriente_f1_std > 1.5 or corriente_f2_std > 1.5 or corriente_f3_std > 1.5:
                fase_fluctuante = ['F1', 'F2', 'F3'][[corriente_f1_std, corriente_f2_std, corriente_f3_std].index(
                    max(corriente_f1_std, corriente_f2_std, corriente_f3_std))]
                recommendations.append(f"Fluctuaciones elevadas en corriente {fase_fluctuante}")
                if status == "good":
                    status = "warning"
                    health_score = 75
                    
            # Mensaje principal
            if status == "critical":
                message = "Análisis básico: Problemas críticos detectados"
                maintenance_due = "inmediato"
            elif status == "warning":
                message = "Análisis básico: Problemas potenciales detectados"
                maintenance_due = "programar revisión"
            else:
                message = "Análisis básico: Funcionamiento normal"
                maintenance_due = "según plan regular"
                recommendations.append("Sistema funcionando correctamente")
            
            # Si no hay recomendaciones específicas
            if not recommendations:
                recommendations.append("Monitorizar funcionamiento")
            
            return {
                "status": status,
                "message": f"Estado de salud: {health_score:.1f}% ({message})",
                "health_score": health_score,
                "recommendations": recommendations,
                "maintenance_due": maintenance_due,
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        except Exception as e:
            print(f"Error en análisis simple: {e}")
            return {
                "status": "unknown",
                "message": "No se pudo realizar análisis",
                "health_score": 50,
                "recommendations": ["Error en sistema de análisis", "Verificar manualmente"],
                "maintenance_due": "desconocido"
            }

    def train_models(self, machine_id):
        """Entrena modelos para una máquina específica con los nuevos parámetros"""
        try:
            data = self._load_training_data(machine_id)
            if data.empty:
                return False
            
            # Entrenar modelo para características eléctricas con nuevos parámetros
            electrical_data = data[['corriente_f1', 'corriente_f2', 'corriente_f3', 
                                    'voltaje_ctrl_izq', 'voltaje_ctrl_der']]
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
            
            # Modelo LSTM para secuencias temporales de corrientes
            if len(data) > 200:  # Suficientes datos para LSTM
                # Crear secuencias para LSTM con datos multicanal (3 fases + 2 controladores)
                seq_data = self._create_multidimensional_sequences(
                    data[['corriente_f1', 'corriente_f2', 'corriente_f3', 
                        'voltaje_ctrl_izq', 'voltaje_ctrl_der']].values
                )
                
                # Definir y entrenar modelo LSTM
                sequence_length = seq_data.shape[1]
                feature_dim = seq_data.shape[2]
                
                lstm_model = keras.Sequential([
                    layers.LSTM(64, input_shape=(sequence_length, feature_dim), return_sequences=True),
                    layers.Dropout(0.2),
                    layers.LSTM(32, return_sequences=False),
                    layers.Dropout(0.2),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(sequence_length * feature_dim, activation='linear'),
                    layers.Reshape((sequence_length, feature_dim))
                ])
                
                lstm_model.compile(optimizer='adam', loss='mse')
                
                try:
                    lstm_model.fit(
                        seq_data, seq_data,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.1,
                        callbacks=[callbacks.EarlyStopping(patience=5)],
                        verbose=0
                    )
                    self.models[f"{machine_id}_sequence"] = lstm_model
                except Exception as e:
                    print(f"Error entrenando modelo LSTM: {e}")
            
            self._save_models(machine_id)
            return True
        except Exception as e:
            print(f"Error training models: {e}")
            return False