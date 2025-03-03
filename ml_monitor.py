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
        """Inicializa la base de datos para almacenar el histórico con nuevos parámetros"""
        try:
            conn = self.get_connection()
            c = conn.cursor()
            
            # Tabla para mediciones actualizada con nuevos parámetros
            c.execute('''CREATE TABLE IF NOT EXISTS mediciones
                        (timestamp DATETIME,
                        maquina_id TEXT,
                        corriente_f1 REAL,
                        corriente_f2 REAL,
                        corriente_f3 REAL,
                        voltaje_ctrl_izq REAL,
                        voltaje_ctrl_der REAL,
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
        """Guarda una nueva medición en la base de datos (versión compatible con estructura antigua)"""
        try:
            conn = self.get_connection()
            c = conn.cursor()
            
            # Obtener la estructura actual de la tabla
            c.execute("PRAGMA table_info(mediciones)")
            columns = [col[1] for col in c.fetchall()]
            
            if 'corriente_f1' in columns:  # Nueva estructura
                try:
                    c.execute('''INSERT INTO mediciones VALUES
                            (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            (datos['timestamp'], maquina_id,
                            datos['corriente_f1'], datos['corriente_f2'], datos['corriente_f3'],
                            datos['voltaje_ctrl_izq'], datos['voltaje_ctrl_der'],
                            datos['posicion'], datos['ciclo_progreso']))
                    print(f"Datos guardados correctamente en nueva estructura para {maquina_id}")
                except Exception as e:
                    print(f"Error al insertar en nueva estructura: {e}")
                    print(f"Datos: {datos}")
            else:  # Estructura antigua
                try:
                    # Calcular promedio de corrientes para compatibilidad
                    corriente_promedio = (datos['corriente_f1'] + datos['corriente_f2'] + datos['corriente_f3']) / 3
                    # Usar el voltaje del controlador activo
                    voltaje = datos['voltaje_ctrl_izq'] * 10 if datos['posicion'] == 'Izquierda' else datos['voltaje_ctrl_der'] * 10
                    
                    c.execute('''INSERT INTO mediciones VALUES
                            (?, ?, ?, ?, ?, ?)''',
                            (datos['timestamp'], maquina_id,
                            voltaje, corriente_promedio,
                            datos['posicion'], datos['ciclo_progreso']))
                    print(f"Datos guardados correctamente en estructura antigua para {maquina_id}")
                except Exception as e:
                    print(f"Error al insertar en estructura antigua: {e}")
                    print(f"Datos: {datos}")
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"Error al guardar medición: {e}")
            print(f"Estructura de tabla: {columns if 'columns' in locals() else 'desconocida'}")

    def detectar_anomalias(self, maquina_id, corriente_f1, corriente_f2, corriente_f3, 
                        voltaje_ctrl_izq, voltaje_ctrl_der, ciclo_progreso):
        """Detecta anomalías usando Isolation Forest con los nuevos parámetros"""
        try:
            conn = self.get_connection()
            df_historico = pd.read_sql_query(
                '''SELECT corriente_f1, corriente_f2, corriente_f3, 
                        voltaje_ctrl_izq, voltaje_ctrl_der, ciclo_progreso
                FROM mediciones
                WHERE maquina_id = ?
                ORDER BY timestamp DESC LIMIT 1000''',
                conn,
                params=(maquina_id,)
            )
            conn.close()

            if len(df_historico) < 100:
                return []

            X = df_historico[['corriente_f1', 'corriente_f2', 'corriente_f3', 
                            'voltaje_ctrl_izq', 'voltaje_ctrl_der', 
                            'ciclo_progreso']].values
            X_scaled = self.scaler.fit_transform(X)
            
            self.isolation_forest.fit(X_scaled)
            
            X_actual = np.array([[corriente_f1, corriente_f2, corriente_f3, 
                                voltaje_ctrl_izq, voltaje_ctrl_der, 
                                ciclo_progreso]])
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
        """Predice tendencias futuras usando datos históricos con nuevos parámetros"""
        try:
            conn = self.get_connection()
            df = pd.read_sql_query(
                '''SELECT timestamp, corriente_f1, corriente_f2, corriente_f3, 
                        voltaje_ctrl_izq, voltaje_ctrl_der
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
            
            # Predecir tendencias para cada corriente de fase
            for variable in ['corriente_f1', 'corriente_f2', 'corriente_f3']:
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
            
            # Predecir tendencias para cada controlador de voltaje
            for variable in ['voltaje_ctrl_izq', 'voltaje_ctrl_der']:
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
                
            # Agregar predicción de próximo mantenimiento basado en todas las fases
            corriente_promedio = (df['corriente_f1'] + df['corriente_f2'] + df['corriente_f3']) / 3
            ciclos_totales = len(df[corriente_promedio.diff() < 0])  # Contar cambios de dirección
            predicciones['proximo_mantenimiento'] = {
                'ciclos_hasta_revision': 1000 - (ciclos_totales % 1000),
                'estado_general': 'bueno' if df[['corriente_f1', 'corriente_f2', 'corriente_f3']].std().mean() < 0.5 else 'requiere revisión'
            }

            return predicciones
        except Exception as e:
            print(f"Error en predicción de tendencias: {e}")
            return None

    def obtener_estadisticas(self, maquina_id, periodo='24H'):
        """Obtiene estadísticas del período especificado con nuevos parámetros"""
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

            estadisticas = {}
            
            # Estadísticas para cada corriente de fase
            for fase in ['corriente_f1', 'corriente_f2', 'corriente_f3']:
                estadisticas[fase] = {
                    'promedio': df[fase].mean(),
                    'max': df[fase].max(),
                    'min': df[fase].min(),
                    'std': df[fase].std()
                }
            
            # Estadísticas para cada controlador de voltaje
            for ctrl in ['voltaje_ctrl_izq', 'voltaje_ctrl_der']:
                estadisticas[ctrl] = {
                    'promedio': df[ctrl].mean(),
                    'max': df[ctrl].max(),
                    'min': df[ctrl].min(),
                    'std': df[ctrl].std()
                }
            
            # Métricas generales
            estadisticas['ciclos_completados'] = len(df[df['ciclo_progreso'] < df['ciclo_progreso'].shift(1)])
            estadisticas['tiempo_promedio_ciclo'] = df['ciclo_progreso'].diff().mean()
            
            # Desbalance entre fases
            if len(df) > 0:
                promedio_f1 = df['corriente_f1'].mean()
                promedio_f2 = df['corriente_f2'].mean()
                promedio_f3 = df['corriente_f3'].mean()
                promedio_total = (promedio_f1 + promedio_f2 + promedio_f3) / 3
                
                # Calcular desbalance como porcentaje de desviación del promedio
                if promedio_total > 0:
                    desbalance_f1 = abs((promedio_f1 - promedio_total) / promedio_total) * 100
                    desbalance_f2 = abs((promedio_f2 - promedio_total) / promedio_total) * 100
                    desbalance_f3 = abs((promedio_f3 - promedio_total) / promedio_total) * 100
                    
                    estadisticas['desbalance_fases'] = {
                        'f1': desbalance_f1,
                        'f2': desbalance_f2,
                        'f3': desbalance_f3,
                        'promedio': (desbalance_f1 + desbalance_f2 + desbalance_f3) / 3
                    }
            
            return estadisticas
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