# ml_monitor.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import sqlite3
from datetime import datetime
import pandas as pd
from prophet import Prophet
import os
import tempfile
import getpass

class MonitoringML:
    def __init__(self):
        # Obtener el directorio del usuario actual
        self.user_dir = os.path.expanduser('~')
        # Crear un directorio para los datos si no existe
        self.data_dir = os.path.join(self.user_dir, 'metro_data')
        os.makedirs(self.data_dir, exist_ok=True)
       
        # Ruta de la base de datos
        self.db_path = os.path.join(self.data_dir, 'historico_maquinas.db')
        print(f"Base de datos creada en: {self.db_path}")
       
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.prophet_models = {}
        self.setup_database()

    def get_connection(self):
        """Crear una nueva conexión a la base de datos con timeout"""
        try:
            return sqlite3.connect(self.db_path, timeout=30.0)
        except sqlite3.Error as e:
            print(f"Error al conectar con la base de datos: {e}")
            # Usar base de datos en memoria como fallback
            return sqlite3.connect(':memory:')

    def setup_database(self):
        """Inicializa la base de datos para almacenar el histórico"""
        try:
            conn = self.get_connection()
            c = conn.cursor()
           
            # Tabla para mediciones
            c.execute('''CREATE TABLE IF NOT EXISTS mediciones
                        (timestamp DATETIME,
                         maquina_id TEXT,
                         voltaje REAL,
                         corriente REAL,
                         posicion TEXT,
                         ciclo_progreso REAL)''')
           
            # Tabla para anomalías detectadas
            c.execute('''CREATE TABLE IF NOT EXISTS anomalias
                        (timestamp DATETIME,
                         maquina_id TEXT,
                         tipo TEXT,
                         valor REAL,
                         descripcion TEXT)''')
           
            conn.commit()
            conn.close()
           
            # Verificar permisos de escritura
            with open(self.db_path, 'a') as f:
                pass
               
        except (sqlite3.Error, IOError) as e:
            print(f"Error al configurar la base de datos: {e}")
            # Intentar crear en el directorio temporal como fallback
            temp_dir = tempfile.gettempdir()
            self.db_path = os.path.join(temp_dir, 'historico_maquinas.db')
            print(f"Intentando crear base de datos en: {self.db_path}")
            self.setup_database()

    def guardar_medicion(self, maquina_id, datos):
        """Guarda una nueva medición en la base de datos"""
        try:
            conn = self.get_connection()
            c = conn.cursor()
           
            c.execute('''INSERT INTO mediciones VALUES
                        (?, ?, ?, ?, ?, ?)''',
                     (datos['timestamp'], maquina_id,
                      datos['voltaje'], datos['corriente'],
                      datos['posicion'], datos['ciclo_progreso']))
           
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"Error al guardar medición: {e}")

    def detectar_anomalias(self, maquina_id, voltaje, corriente, ciclo_progreso):
        """Detecta anomalías usando Isolation Forest"""
        try:
            conn = self.get_connection()
            df_historico = pd.read_sql_query(
                '''SELECT voltaje, corriente, ciclo_progreso
                   FROM mediciones
                   WHERE maquina_id = ?
                   ORDER BY timestamp DESC LIMIT 1000''',
                conn,
                params=(maquina_id,)
            )
            conn.close()

            if len(df_historico) < 100:
                return []

            X = df_historico[['voltaje', 'corriente', 'ciclo_progreso']].values
            X_scaled = self.scaler.fit_transform(X)
           
            self.isolation_forest.fit(X_scaled)
           
            X_actual = np.array([[voltaje, corriente, ciclo_progreso]])
            X_actual_scaled = self.scaler.transform(X_actual)
           
            es_anomalia = self.isolation_forest.predict(X_actual_scaled)[0] == -1
           
            if es_anomalia:
                return [{
                    'timestamp': datetime.now(),
                    'tipo': 'comportamiento_anormal',
                    'descripcion': 'Patrón de comportamiento anormal detectado'
                }]
            return []
        except Exception as e:
            print(f"Error en detección de anomalías: {e}")
            return []

    def predecir_tendencias(self, maquina_id):
        """Predice tendencias futuras usando datos históricos"""
        try:
            conn = self.get_connection()
            df = pd.read_sql_query(
                '''SELECT timestamp, voltaje, corriente
                   FROM mediciones
                   WHERE maquina_id = ?
                   ORDER BY timestamp''',
                conn,
                params=(maquina_id,)
            )
            conn.close()

            if len(df) < 100:  # Necesitamos suficientes datos históricos
                return None

            predicciones = {}
            for variable in ['voltaje', 'corriente']:
                # Calcular tendencia usando promedio móvil
                valores = df[variable].rolling(window=20).mean()
                ultimo_valor = valores.iloc[-1]
                tendencia = valores.diff().mean()
               
                # Calcular límites basados en desviación estándar
                std = df[variable].std()
               
                predicciones[variable] = {
                    'proximo_valor': ultimo_valor + tendencia,
                    'limite_superior': ultimo_valor + 2*std,
                    'limite_inferior': ultimo_valor - 2*std,
                    'tendencia': 'aumentando' if tendencia > 0 else 'disminuyendo',
                    'confianza': 'alta' if len(df) > 200 else 'media'
                }
               
                # Agregar predicción de próximo mantenimiento
                if variable == 'corriente':
                    ciclos_totales = len(df[df['corriente'].diff() < 0])  # Contar cambios de dirección
                    predicciones['proximo_mantenimiento'] = {
                        'ciclos_hasta_revision': 1000 - (ciclos_totales % 1000),
                        'estado_general': 'bueno' if std < 0.5 else 'requiere revisión'
                    }

            return predicciones
        except Exception as e:
            print(f"Error en predicción de tendencias: {e}")
            return None

    def obtener_estadisticas(self, maquina_id, periodo='24H'):
        """Obtiene estadísticas del período especificado"""
        try:
            conn = self.get_connection()
            fecha_inicio = (datetime.now() - pd.Timedelta(periodo)).strftime('%Y-%m-%d %H:%M:%S')
           
            df = pd.read_sql_query(
                '''SELECT * FROM mediciones
                   WHERE maquina_id = ? AND timestamp > ?
                   ORDER BY timestamp''',
                conn,
                params=(maquina_id, fecha_inicio)
            )
            conn.close()

            if len(df) == 0:
                return None

            return {
                'voltaje': {
                    'promedio': df['voltaje'].mean(),
                    'max': df['voltaje'].max(),
                    'min': df['voltaje'].min(),
                    'std': df['voltaje'].std()
                },
                'corriente': {
                    'promedio': df['corriente'].mean(),
                    'max': df['corriente'].max(),
                    'min': df['corriente'].min(),
                    'std': df['corriente'].std()
                },
                'ciclos_completados': len(df[df['ciclo_progreso'] < df['ciclo_progreso'].shift(1)]),
                'tiempo_promedio_ciclo': df['ciclo_progreso'].diff().mean()
            }
        except Exception as e:
            print(f"Error al obtener estadísticas: {e}")
            return None

    def obtener_ultimas_mediciones(self, maquina_id, limite=10):
        """Obtiene las últimas mediciones de la base de datos"""
        try:
            conn = self.get_connection()
            df = pd.read_sql_query(
                '''SELECT * FROM mediciones
                   WHERE maquina_id = ?
                   ORDER BY timestamp DESC LIMIT ?''',
                conn,
                params=(maquina_id, limite)
            )
            conn.close()
            return df
        except sqlite3.Error as e:
            print(f"Error al obtener mediciones: {e}")
            return pd.DataFrame()  # Retornar DataFrame vacío en caso de error