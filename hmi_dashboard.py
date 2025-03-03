import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import pandas as pd
import random
from datetime import datetime, timedelta
import time
from threading import Thread, Lock
import dash_bootstrap_components as dbc
from ml_monitor import MonitoringML
from maquina_cambio import crear_svg_maquina, crear_indicador_progreso
from replay_system_basculacion import ReplaySystemBasculacion, create_replay_controls_basculacion, register_replay_callbacks_basculacion
from predictive_maintenance import PredictiveMaintenanceSystem

# Importar utilidades de depuración
from dash_debug import create_corrientes_graph_demo, create_voltajes_graph_demo, fix_graphs_demo

# Inicializar el monitor ML y sistema de replay
ml_monitor = MonitoringML()
replay_system = ReplaySystemBasculacion(ml_monitor.db_path)
predictive_system = PredictiveMaintenanceSystem(ml_monitor.db_path)  # Nuevo

# Mutex para sincronización de datos
data_lock = Lock()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Monitor de Máquinas de Cambio - Metro de Santiago"

# =========================================
# Configuración de simulación realista
# =========================================

# Actualización de la configuración de máquinas
MAQUINAS = {
    "VIM-11/21": {
        "ubicacion": "Línea 4A, Estación Vicuña Mackena",
        "voltaje_ctrl_nominal": 24,  # VDC nominal para controladores
        "corriente_maxima": 5.0,     # Corriente máxima para cada fase
        "ciclo_duracion": 6          # segundos
    },
    "TSE-54B": {
        "ubicacion": "Talleres San Eugenio",
        "voltaje_ctrl_nominal": 24,  # VDC nominal para controladores
        "corriente_maxima": 5.0,     # Corriente máxima para cada fase
        "ciclo_duracion": 6
    }
}

# Estado inicial de las máquinas con campos actualizados
# Estado inicial de las máquinas con campos actualizados
estado_maquinas = {
    "VIM-11/21": {
        "timestamp": [],
        "corriente_f1": [],
        "corriente_f2": [],
        "corriente_f3": [],
        "voltaje_ctrl_izq": [],
        "voltaje_ctrl_der": [],
        "posicion": "Izquierda",
        "ciclo_progreso": 0,
        "alertas": [],
        "predicciones": {},
        "estadisticas": {},
        "modo_operacion": "normal",  # normal/replay
        "health_status": {           # Para mantenimiento predictivo
            "status": "unknown",
            "health_score": 0,
            "recommendations": [],
            "maintenance_due": "Unknown",
            "last_updated": None
        }
    },
    "TSE-54B": {
        "timestamp": [],
        "corriente_f1": [],
        "corriente_f2": [],
        "corriente_f3": [],
        "voltaje_ctrl_izq": [],
        "voltaje_ctrl_der": [],
        "posicion": "Derecha",
        "ciclo_progreso": 0,
        "alertas": [],
        "predicciones": {},
        "estadisticas": {},
        "modo_operacion": "normal",  # normal/replay
        "health_status": {           # Para mantenimiento predictivo
            "status": "unknown",
            "health_score": 0,
            "recommendations": [],
            "maintenance_due": "Unknown",
            "last_updated": None
        }
    }
}

def inicializar_datos_prueba():
    """Inicializa datos de prueba para asegurar visualización inmediata"""
    print("Inicializando datos de prueba para visualización inmediata...")
    
    for maquina_id, estado in estado_maquinas.items():
        now = datetime.now()
        config = MAQUINAS[maquina_id]
        
        # Generar 20 puntos de datos históricos
        for i in range(20):
            tiempo = now - timedelta(seconds=i*1)
            
            # Valores realistas para simular el comportamiento
            if i % 6 < 3:  # Alternar entre espera y movimiento
                corriente_base = 0.3  # Menor corriente en espera
                prog = 0
            else:
                corriente_base = 4.0  # Mayor corriente en movimiento
                prog = 50 if i % 12 < 6 else 80
            
            # Aplicar factores para crear variaciones entre fases
            estado["timestamp"].append(tiempo)
            estado["corriente_f1"].append(corriente_base * (1.0 + random.uniform(-0.1, 0.1)))
            estado["corriente_f2"].append(corriente_base * (0.95 + random.uniform(-0.1, 0.1)))
            estado["corriente_f3"].append(corriente_base * (1.05 + random.uniform(-0.1, 0.1)))
            
            # Voltajes de los controladores
            if estado["posicion"] == "Izquierda":
                estado["voltaje_ctrl_izq"].append(24.0 + random.uniform(-0.5, 0.5))
                estado["voltaje_ctrl_der"].append(1.2 + random.uniform(-0.1, 0.1))
            else:
                estado["voltaje_ctrl_izq"].append(1.2 + random.uniform(-0.1, 0.1))
                estado["voltaje_ctrl_der"].append(24.0 + random.uniform(-0.5, 0.5))
            
            # Guardar en la base de datos para consistencia
            ml_monitor.guardar_medicion(maquina_id, {
                "timestamp": tiempo,
                "corriente_f1": estado["corriente_f1"][-1],
                "corriente_f2": estado["corriente_f2"][-1],
                "corriente_f3": estado["corriente_f3"][-1],
                "voltaje_ctrl_izq": estado["voltaje_ctrl_izq"][-1],
                "voltaje_ctrl_der": estado["voltaje_ctrl_der"][-1],
                "posicion": estado["posicion"],
                "ciclo_progreso": prog
            })
        
        # También actualizar el estado actual con el último valor
        estado["ciclo_progreso"] = prog
        
        print(f"Datos de prueba generados para {maquina_id}: {len(estado['timestamp'])} puntos")
        print(f"Valores actuales - F1: {estado['corriente_f1'][-1]:.2f}A, "
              f"F2: {estado['corriente_f2'][-1]:.2f}A, F3: {estado['corriente_f3'][-1]:.2f}A")
        print(f"Voltajes - Izq: {estado['voltaje_ctrl_izq'][-1]:.2f}V, "
              f"Der: {estado['voltaje_ctrl_der'][-1]:.2f}V")

# =========================================
# Simulación de comportamiento realista
# =========================================

def simular_maquina(maquina_id):
    """Simula el comportamiento de una máquina de cambio con múltiples fases y controladores"""
    config = MAQUINAS[maquina_id]
    estado = estado_maquinas[maquina_id]
    tiempo_espera = 3  # segundos de espera antes del movimiento
   
    # Contador para actualización periódica del estado de salud
    contador_health = 0
    
    # Factor de variación entre fases (para simular desbalance)
    factor_f1 = 1.0
    factor_f2 = 0.95
    factor_f3 = 1.05
    
    # Pre-llenar con algunos datos iniciales para tener algo que mostrar de inmediato
    with data_lock:
        now = datetime.now()
        # Solo pre-llenar si no hay datos
        if not estado["timestamp"]:
            for i in range(10):
                tiempo_simulado = now - timedelta(seconds=i*0.5)
                
                corriente_base = random.uniform(0.2, 0.5)
                voltaje_base = config["voltaje_ctrl_nominal"]
                
                estado["timestamp"].insert(0, tiempo_simulado)
                estado["corriente_f1"].insert(0, corriente_base * factor_f1)
                estado["corriente_f2"].insert(0, corriente_base * factor_f2)
                estado["corriente_f3"].insert(0, corriente_base * factor_f3)
                
                if estado["posicion"] == "Izquierda":
                    estado["voltaje_ctrl_izq"].insert(0, voltaje_base + random.uniform(-0.5, 0.5))
                    estado["voltaje_ctrl_der"].insert(0, voltaje_base * 0.05 + random.uniform(-0.1, 0.1))
                else:
                    estado["voltaje_ctrl_izq"].insert(0, voltaje_base * 0.05 + random.uniform(-0.1, 0.1))
                    estado["voltaje_ctrl_der"].insert(0, voltaje_base + random.uniform(-0.5, 0.5))
            
            print(f"Datos iniciales generados para {maquina_id}: {len(estado['timestamp'])} puntos")
    
    while True:
        try:
            with data_lock:
                # Verificar si estamos en modo replay
                if estado["modo_operacion"] == "replay":
                    # Permitir que el callback de interval maneje la lógica de replay
                    print(f"Simulador: Máquina {maquina_id} en modo replay, saltando simulación")
                    time.sleep(0.5)
                    continue
                
                # Modo normal de simulación
                now = datetime.now()
                tiempo_total = config["ciclo_duracion"] + tiempo_espera
                tiempo_actual = time.time() % tiempo_total
               
                # Fase de espera
                if tiempo_actual < tiempo_espera:
                    fase = 0
                else:
                    # Fase de movimiento
                    fase = (tiempo_actual - tiempo_espera) / config["ciclo_duracion"]
               
                # Determinar posición
                ciclo_completo = (time.time() // tiempo_total) % 2
                estado["posicion"] = "Derecha" if ciclo_completo else "Izquierda"
                
                # Simular voltajes de controladores
                voltaje_base = config["voltaje_ctrl_nominal"]
                # El controlador activo tendrá más variación
                if estado["posicion"] == "Izquierda":
                    voltaje_ctrl_izq = voltaje_base + random.uniform(-0.5, 0.5)
                    voltaje_ctrl_der = voltaje_base * 0.05 + random.uniform(-0.1, 0.1)
                else:
                    voltaje_ctrl_izq = voltaje_base * 0.05 + random.uniform(-0.1, 0.1)
                    voltaje_ctrl_der = voltaje_base + random.uniform(-0.5, 0.5)
               
                # Simular corrientes con variación por fase durante el movimiento
                if fase == 0:  # En espera
                    corriente_f1 = random.uniform(0.2, 0.5) * factor_f1
                    corriente_f2 = random.uniform(0.2, 0.5) * factor_f2
                    corriente_f3 = random.uniform(0.2, 0.5) * factor_f3
                elif fase < 0.2 or fase > 0.8:  # Inicio/fin de movimiento
                    corriente_f1 = random.uniform(0.5, 1.5) * factor_f1
                    corriente_f2 = random.uniform(0.5, 1.5) * factor_f2
                    corriente_f3 = random.uniform(0.5, 1.5) * factor_f3
                else:  # Durante el movimiento
                    corriente_base = config["corriente_maxima"] * (0.8 + random.uniform(-0.1, 0.3))
                    corriente_f1 = corriente_base * factor_f1
                    corriente_f2 = corriente_base * factor_f2
                    corriente_f3 = corriente_base * factor_f3
               
                # Generar alertas básicas
                alertas = []
                if max(corriente_f1, corriente_f2, corriente_f3) > 5.5:
                    fase_afectada = ["F1", "F2", "F3"][[corriente_f1, corriente_f2, corriente_f3].index(max(corriente_f1, corriente_f2, corriente_f3))]
                    alertas.append(f"ALERTA: Sobrecorriente en {fase_afectada} ({max(corriente_f1, corriente_f2, corriente_f3):.1f}A)")
                
                # Alertar si hay desbalance importante entre fases
                if fase > 0 and max(corriente_f1, corriente_f2, corriente_f3) > 2.0:
                    promedio = (corriente_f1 + corriente_f2 + corriente_f3) / 3
                    desviaciones = [
                        abs(corriente_f1 - promedio) / promedio,
                        abs(corriente_f2 - promedio) / promedio,
                        abs(corriente_f3 - promedio) / promedio
                    ]
                    if max(desviaciones) > 0.2:  # Más de 20% de desviación
                        fase_desbalanceada = ["F1", "F2", "F3"][desviaciones.index(max(desviaciones))]
                        alertas.append(f"ALERTA: Desbalance en fase {fase_desbalanceada} ({max(desviaciones)*100:.1f}%)")
                
                # Alerta de voltaje del controlador
                if voltaje_ctrl_izq > config["voltaje_ctrl_nominal"] * 1.15 or voltaje_ctrl_der > config["voltaje_ctrl_nominal"] * 1.15:
                    ctrl_afectado = "izquierdo" if voltaje_ctrl_izq > voltaje_ctrl_der else "derecho"
                    voltaje_afectado = max(voltaje_ctrl_izq, voltaje_ctrl_der)
                    alertas.append(f"ALERTA: Sobrevoltaje en controlador {ctrl_afectado} ({voltaje_afectado:.1f}V)")
               
                # Actualizar estado - IMPORTANTE: verificar si los arreglos existen antes de append
                # Asegurar que todos los arreglos estén inicializados
                if "timestamp" not in estado or estado["timestamp"] is None:
                    estado["timestamp"] = []
                if "corriente_f1" not in estado or estado["corriente_f1"] is None:
                    estado["corriente_f1"] = []
                if "corriente_f2" not in estado or estado["corriente_f2"] is None:
                    estado["corriente_f2"] = []
                if "corriente_f3" not in estado or estado["corriente_f3"] is None:
                    estado["corriente_f3"] = []
                if "voltaje_ctrl_izq" not in estado or estado["voltaje_ctrl_izq"] is None:
                    estado["voltaje_ctrl_izq"] = []
                if "voltaje_ctrl_der" not in estado or estado["voltaje_ctrl_der"] is None:
                    estado["voltaje_ctrl_der"] = []
                if "alertas" not in estado or estado["alertas"] is None:
                    estado["alertas"] = []
                
                # Ahora añadir los nuevos valores
                estado["timestamp"].append(now)
                estado["corriente_f1"].append(corriente_f1)
                estado["corriente_f2"].append(corriente_f2)
                estado["corriente_f3"].append(corriente_f3)
                estado["voltaje_ctrl_izq"].append(voltaje_ctrl_izq)
                estado["voltaje_ctrl_der"].append(voltaje_ctrl_der)
                estado["ciclo_progreso"] = fase * 100 if fase > 0 else 0
               
                # Agregar alertas si existen
                for alerta in alertas:
                    estado["alertas"].append(f"{now.strftime('%H:%M:%S')} - {alerta}")
                
                # Imprimir información de depuración
                if len(estado["timestamp"]) % 50 == 0:
                    print(f"Datos simulados en {maquina_id}:")
                    print(f"  Estado tiene {len(estado['timestamp'])} registros")
                    print(f"  Corriente F1: {estado['corriente_f1'][-1]:.2f}A")
                    print(f"  Corriente F2: {estado['corriente_f2'][-1]:.2f}A")
                    print(f"  Corriente F3: {estado['corriente_f3'][-1]:.2f}A")
                    print(f"  Voltaje Ctrl Izq: {estado['voltaje_ctrl_izq'][-1]:.2f}V")
                    print(f"  Voltaje Ctrl Der: {estado['voltaje_ctrl_der'][-1]:.2f}V")
                    print(f"  Posición: {estado['posicion']}")
                    print(f"  Progreso: {estado['ciclo_progreso']:.2f}%")
               
                # Guardar datos en la base de datos
                ml_monitor.guardar_medicion(maquina_id, {
                    "timestamp": now,
                    "corriente_f1": corriente_f1,
                    "corriente_f2": corriente_f2,
                    "corriente_f3": corriente_f3,
                    "voltaje_ctrl_izq": voltaje_ctrl_izq,
                    "voltaje_ctrl_der": voltaje_ctrl_der,
                    "posicion": estado["posicion"],
                    "ciclo_progreso": estado["ciclo_progreso"]
                })
               
                # Actualizar predicciones y estadísticas cada 10 segundos
                if len(estado["timestamp"]) % 20 == 0:
                    estado["predicciones"] = ml_monitor.predecir_tendencias(maquina_id)
                    estado["estadisticas"] = ml_monitor.obtener_estadisticas(maquina_id)
                
                # Actualizar estado de salud cada 60 segundos
                contador_health += 1
                if contador_health >= 120:
                    contador_health = 0
                    try:
                        # Crear DataFrame con datos recientes para análisis
                        recent_data = pd.DataFrame({
                            'timestamp': estado["timestamp"][-100:],
                            'corriente_f1': estado["corriente_f1"][-100:],
                            'corriente_f2': estado["corriente_f2"][-100:],
                            'corriente_f3': estado["corriente_f3"][-100:],
                            'voltaje_ctrl_izq': estado["voltaje_ctrl_izq"][-100:],
                            'voltaje_ctrl_der': estado["voltaje_ctrl_der"][-100:],
                            'posicion': [estado["posicion"]] * min(100, len(estado["timestamp"])),
                            'ciclo_progreso': [estado["ciclo_progreso"]] * min(100, len(estado["timestamp"]))
                        })
                        
                        # Calcular estado de salud
                        health_status = predictive_system.get_machine_health(maquina_id, recent_data)
                        estado["health_status"] = health_status
                        
                        # Añadir alerta si el estado es crítico
                        if health_status["status"] == "critical":
                            estado["alertas"].append(
                                f"{now.strftime('%H:%M:%S')} - ALERTA DE MANTENIMIENTO: {health_status['message']}"
                            )
                            
                        print(f"Actualizado estado de salud para {maquina_id}: {health_status['status']}")
                    except Exception as e:
                        print(f"Error actualizando estado de salud: {e}")
               
                # Mantener solo los últimos N puntos para la visualización
                max_points = 100
                if len(estado["timestamp"]) > max_points:
                    estado["timestamp"] = estado["timestamp"][-max_points:]
                    estado["corriente_f1"] = estado["corriente_f1"][-max_points:]
                    estado["corriente_f2"] = estado["corriente_f2"][-max_points:]
                    estado["corriente_f3"] = estado["corriente_f3"][-max_points:]
                    estado["voltaje_ctrl_izq"] = estado["voltaje_ctrl_izq"][-max_points:]
                    estado["voltaje_ctrl_der"] = estado["voltaje_ctrl_der"][-max_points:]
                    estado["alertas"] = estado["alertas"][-10:]  # Mantener solo las últimas 10 alertas
                    
        except Exception as e:
            import traceback
            print(f"Error en simulación de {maquina_id}: {str(e)}")
            print(traceback.format_exc())
            
        time.sleep(0.5)
        
def create_corrientes_graph(estado):
    """Crea gráfico de corrientes con datos o con mensaje de espera"""
    fig = go.Figure()
    
    if len(estado.get("timestamp", [])) > 1:
        # Añadir cada fase como una línea separada
        fig.add_trace(go.Scatter(
            x=estado["timestamp"],
            y=estado["corriente_f1"],
            line=dict(color='#FF9800'),
            name='Fase 1'
        ))
        
        fig.add_trace(go.Scatter(
            x=estado["timestamp"],
            y=estado["corriente_f2"],
            line=dict(color='#FFA726'),
            name='Fase 2'
        ))
        
        fig.add_trace(go.Scatter(
            x=estado["timestamp"],
            y=estado["corriente_f3"],
            line=dict(color='#FFB74D'),
            name='Fase 3'
        ))
    else:
        # Si no hay suficientes datos, añadir un punto inicial para que al menos se vea algo
        now = datetime.now()
        times = [now - timedelta(seconds=1), now]
        fig.add_trace(go.Scatter(
            x=times,
            y=[0.5, 0.5],
            line=dict(color='#FF9800'),
            name='Fase 1'
        ))
        
        fig.add_trace(go.Scatter(
            x=times,
            y=[0.5, 0.5],
            line=dict(color='#FFA726'),
            name='Fase 2'
        ))
        
        fig.add_trace(go.Scatter(
            x=times,
            y=[0.5, 0.5],
            line=dict(color='#FFB74D'),
            name='Fase 3'
        ))
    
    fig.update_layout(
        title="Tendencia de Corrientes",
        yaxis_title="Corriente (A)",
        margin=dict(l=30, r=30, t=40, b=30),
        height=300,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.add_hline(y=5.5, line_dash="dot", line_color="red")
    
    return fig

def create_voltajes_graph(estado):
    """Crea gráfico de voltajes con datos o con mensaje de espera"""
    fig = go.Figure()
    
    if len(estado.get("timestamp", [])) > 1:
        # Añadir cada controlador como una línea separada
        fig.add_trace(go.Scatter(
            x=estado["timestamp"],
            y=estado["voltaje_ctrl_izq"],
            line=dict(color='#2196F3'),
            name='Ctrl Izquierdo'
        ))
        
        fig.add_trace(go.Scatter(
            x=estado["timestamp"],
            y=estado["voltaje_ctrl_der"],
            line=dict(color='#64B5F6'),
            name='Ctrl Derecho'
        ))
    else:
        # Si no hay suficientes datos, añadir un punto inicial
        now = datetime.now()
        times = [now - timedelta(seconds=1), now]
        fig.add_trace(go.Scatter(
            x=times,
            y=[24.0, 24.0],
            line=dict(color='#2196F3'),
            name='Ctrl Izquierdo'
        ))
        
        fig.add_trace(go.Scatter(
            x=times,
            y=[24.0, 24.0],
            line=dict(color='#64B5F6'),
            name='Ctrl Derecho'
        ))
    
    fig.update_layout(
        title="Tendencia de Voltajes de Controladores",
        yaxis_title="Voltaje (VDC)",
        margin=dict(l=30, r=30, t=40, b=30),
        height=300,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Añadir línea de referencia para el voltaje nominal
    if "machine_id" in estado:
        voltaje_nominal = MAQUINAS[estado["machine_id"]]["voltaje_ctrl_nominal"]
    else:
        voltaje_nominal = 24.0  # Valor predeterminado
        
    fig.add_hline(y=voltaje_nominal, line_color="green")
    fig.add_hline(y=voltaje_nominal * 1.15, line_dash="dot", line_color="red")
    
    return fig

        
# Actualización de la función crear_tarjeta_health
def crear_tarjeta_health(maquina_id):
    """Crea una tarjeta con la información de salud de la máquina adaptada a los nuevos parámetros"""
    estado = estado_maquinas[maquina_id]
    health = estado.get("health_status", {})
    
    # Determinar color según estado
    status = health.get("status", "unknown")
    if status == "good":
        color = "success"
        icon = "✓"
    elif status == "warning":
        color = "warning"
        icon = "⚠️"
    elif status == "critical":
        color = "danger"
        icon = "⛔"
    else:
        color = "secondary"
        icon = "?"
    
    # Crear componente
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                f"{icon} Estado de Salud ", 
                dbc.Badge(
                    f"{health.get('health_score', 0):.1f}%", 
                    color=color,
                    className="ms-2"
                )
            ], className="d-flex justify-content-between align-items-center")
        ], className=f"bg-{color} text-white"),
        
        dbc.CardBody([
            html.Div([
                html.Strong("Diagnóstico: "),
                html.Span(health.get("message", "Sin datos suficientes"))
            ], className="mb-2"),
            
            html.Div([
                html.Strong("Próximo mantenimiento: "),
                html.Span(health.get("maintenance_due", "Desconocido"))
            ], className="mb-3"),
            
            html.H6("Recomendaciones:"),
            html.Ul([
                html.Li(rec) for rec in health.get("recommendations", ["Sin recomendaciones"])
            ], className="small"),
            
            html.Div([
                html.Small(f"Última actualización: {health.get('last_updated', 'Nunca')}")
            ], className="text-muted mt-2 text-end")
        ])
    ], className="mb-3")

# =========================================
# Interfaz de usuario profesional
# =========================================

# Actualización de la función crear_tarjeta_maquina
def crear_tarjeta_maquina(maquina_id):
    """Crea una tarjeta con la información de una máquina de cambio con los nuevos parámetros"""
    config = MAQUINAS[maquina_id]
    estado = estado_maquinas[maquina_id]
    
    # NUEVO: Verificar y proporcionar valores predeterminados si es necesario
    corriente_f1 = f"{estado['corriente_f1'][-1]:.1f}" if estado.get('corriente_f1') and len(estado['corriente_f1']) > 0 else "0.5"
    corriente_f2 = f"{estado['corriente_f2'][-1]:.1f}" if estado.get('corriente_f2') and len(estado['corriente_f2']) > 0 else "0.5"
    corriente_f3 = f"{estado['corriente_f3'][-1]:.1f}" if estado.get('corriente_f3') and len(estado['corriente_f3']) > 0 else "0.5"
    voltaje_izq = f"{estado['voltaje_ctrl_izq'][-1]:.1f}" if estado.get('voltaje_ctrl_izq') and len(estado['voltaje_ctrl_izq']) > 0 else "24.0"
    voltaje_der = f"{estado['voltaje_ctrl_der'][-1]:.1f}" if estado.get('voltaje_ctrl_der') and len(estado['voltaje_ctrl_der']) > 0 else "24.0"
   
    return dbc.Card([
        dbc.CardHeader([
            html.H4(f"Máquina: {maquina_id}", className="card-title"),
            html.Small(config["ubicacion"], className="text-muted")
        ]),
       
        dbc.CardBody([
            dbc.Row([
                # Columna 1: Indicadores principales y visualización
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.Div("Última posición", className="text-center mb-2"),
                            html.Div(
                                estado["posicion"],
                                id=f"posicion-{maquina_id}",
                                className="h3 text-center",
                                style={'color': '#4CAF50' if estado["posicion"] == "Izquierda" else '#FF5722'}
                            )
                        ], width=6),
                       
                        dbc.Col([
                            dcc.Graph(
                                id=f"ciclo-progreso-{maquina_id}",
                                figure=crear_indicador_progreso(estado["ciclo_progreso"]),
                                config={'displayModeBar': False},
                                style={'height': '120px'}
                            )
                        ], width=6)
                    ]),
                   
                    # Visualización de la máquina de cambio
                    html.Div(
                        id=f"maquina-cambio-{maquina_id}",
                        children=crear_svg_maquina(
                            estado["posicion"],
                            estado["ciclo_progreso"]
                        ),
                        className="mt-4"
                    ),
                    
                    # Nuevos indicadores de corriente (3 fases)
                    html.Div("Corrientes (A)", className="text-center mt-4 mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Fase 1", className="text-center small"),
                            html.Div(
                                corriente_f1,
                                id=f"corriente-f1-{maquina_id}",
                                className="h4 text-center",
                                style={'color': '#FF9800'}
                            )
                        ], width=4),
                        dbc.Col([
                            html.Div("Fase 2", className="text-center small"),
                            html.Div(
                                corriente_f2,
                                id=f"corriente-f2-{maquina_id}",
                                className="h4 text-center",
                                style={'color': '#FF9800'}
                            )
                        ], width=4),
                        dbc.Col([
                            html.Div("Fase 3", className="text-center small"),
                            html.Div(
                                corriente_f3,
                                id=f"corriente-f3-{maquina_id}",
                                className="h4 text-center",
                                style={'color': '#FF9800'}
                            )
                        ], width=4)
                    ]),
                    
                    # Indicadores de voltaje de controladores
                    html.Div("Voltaje Controladores (VDC)", className="text-center mt-4 mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Izquierdo", className="text-center small"),
                            html.Div(
                                voltaje_izq,
                                id=f"voltaje-ctrl-izq-{maquina_id}",
                                className="h4 text-center",
                                style={'color': '#2196F3'}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Div("Derecho", className="text-center small"),
                            html.Div(
                                voltaje_der,
                                id=f"voltaje-ctrl-der-{maquina_id}",
                                className="h4 text-center",
                                style={'color': '#2196F3'}
                            )
                        ], width=6)
                    ]),
                    
                    # Sección para la tarjeta de salud
                    html.Div(
                        id=f"health-card-{maquina_id}",
                        className="mt-4"
                    )
                ], width=4),
               
                # Columna 2: Gráficos y predicciones
                dbc.Col([
                    # Modo de operación
                    html.Div(
                        id=f"modo-operacion-{maquina_id}",
                        className="text-center mb-3"
                    ),
                   
                    # Gráficos de corrientes y voltajes
                    html.Div([
                        dbc.Tabs([
                            dbc.Tab([
                                dcc.Graph(id=f"corrientes-graph-{maquina_id}")
                            ], label="Corrientes", tab_id=f"tab-corrientes-{maquina_id}"),
                            dbc.Tab([
                                dcc.Graph(id=f"voltajes-graph-{maquina_id}")
                            ], label="Voltajes", tab_id=f"tab-voltajes-{maquina_id}")
                        ], id=f"tabs-graficos-{maquina_id}")
                    ]),
                   
                    # Sección de predicciones
                    html.Div([
                        html.H4("Predicciones y Análisis", className="mb-3"),
                        html.Div(id=f"predicciones-{maquina_id}")
                    ], className="mt-4")
                ], width=8)
            ]),
           
            # Sección de históricos
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H4("Histórico de Mediciones", className="mb-3"),
                    dbc.Table(
                        id=f"tabla-historico-{maquina_id}",
                        striped=True,
                        bordered=True,
                        hover=True
                    )
                ])
            ])
        ]),
       
        dbc.CardFooter([
            html.Div("Últimas alertas:", className="text-danger"),
            html.Ul([
                html.Li(alert, className="small") for alert in estado["alertas"]
            ], id=f"alertas-{maquina_id}", className="list-unstyled mb-0")
        ], className="py-2")
    ], className="mb-4")

# =========================================
# Funciones de actualización de UI
# =========================================

def actualizar_ui(n_intervals, replay_intervals, maquina_id):
    """Función de actualización de la interfaz de usuario con los nuevos parámetros"""
    try:
        with data_lock:
            # Añadir depuración para ver si se llama y con qué valores
            print(f"actualizar_ui llamada: n_intervals={n_intervals}, replay_intervals={replay_intervals}, maquina={maquina_id}")
            
            estado = estado_maquinas[maquina_id]
            if not estado["timestamp"]:
                print(f"No hay datos de timestamp para {maquina_id}, creando visualización vacía")
                # Crear visualizaciones vacías para evitar errores
                fig_corrientes = create_corrientes_graph(estado)
                fig_voltajes = create_voltajes_graph(estado)
                
                indicador_progreso = crear_indicador_progreso(0)
                maquina_viz = crear_svg_maquina(estado["posicion"], 0)
                health_card = dbc.Card(dbc.CardBody("Esperando datos para análisis"))
                
                return (
                    fig_corrientes,
                    fig_voltajes,
                    estado["posicion"],
                    {'color': '#4CAF50'} if estado["posicion"] == "Izquierda" else {'color': '#FF5722'},
                    "N/A",  # corriente_f1
                    "N/A",  # corriente_f2
                    "N/A",  # corriente_f3
                    "N/A",  # voltaje_ctrl_izq
                    "N/A",  # voltaje_ctrl_der
                    [],     # alertas
                    indicador_progreso,
                    maquina_viz,
                    html.P("Recopilando datos para predicciones..."),  # predicciones
                    html.P("Esperando datos históricos..."),  # tabla
                    html.Div([dbc.Badge("Modo: NORMAL", color="success", className="p-2")]),  # modo
                    health_card
                )
            
            # Verificar si estamos en modo reproducción
            if estado["modo_operacion"] == "replay":
                print(f"Actualizando UI en modo replay: {maquina_id}, Pos={estado['posicion']}, Prog={estado['ciclo_progreso']}")
            
            # Gráfico de corrientes usando la nueva función
            try:
                # Usar datos de demostración para garantizar que se muestren gráficos
                demo_data = fix_graphs_demo()
                # Actualizar el estado con estos datos si está vacío
                if len(estado.get("timestamp", [])) < 2:
                    estado.update(demo_data)
                
                estado["machine_id"] = maquina_id  # Agregar machine_id para referencia
                # Usar el generador de gráficos de demostración para garantizar visualización
                fig_corrientes = create_corrientes_graph_demo()
                print(f"Gráfico de corrientes creado correctamente para {maquina_id}")
            except Exception as e:
                print(f"Error creando gráfico de corrientes para {maquina_id}: {e}")
                fig_corrientes = go.Figure()
                fig_corrientes.add_annotation(
                    text=f"Error: {str(e)}",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
                fig_corrientes.update_layout(
                    title="Tendencia de Corrientes",
                    template="plotly_dark",
                    height=300
                )
            
            # Gráfico de voltajes usando la nueva función
            try:
                fig_voltajes = create_voltajes_graph(estado)
                print(f"Gráfico de voltajes creado correctamente para {maquina_id}")
            except Exception as e:
                print(f"Error creando gráfico de voltajes para {maquina_id}: {e}")
                fig_voltajes = go.Figure()
                fig_voltajes.add_annotation(
                    text=f"Error: {str(e)}",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
                fig_voltajes.update_layout(
                    title="Tendencia de Voltajes",
                    template="plotly_dark",
                    height=300
                )
           
            # Indicador de posición
            pos_color = {'color': '#4CAF50'} if estado["posicion"] == "Izquierda" else {'color': '#FF5722'}
           
            # Actualizar visualización de la máquina
            try:
                maquina_viz = crear_svg_maquina(
                    estado["posicion"],
                    estado["ciclo_progreso"]
                )
                print(f"Visualización de máquina creada correctamente para {maquina_id}")
            except Exception as e:
                print(f"Error creando visualización de máquina para {maquina_id}: {e}")
                # Crear visualización por defecto
                maquina_viz = html.Div("Error en visualización", style={"color": "red"})
           
            # Crear contenido de predicciones
            try:
                predicciones = estado.get("predicciones", {})
                if predicciones:
                    # Crear tarjetas para cada grupo de predicciones
                    pred_content = [
                        dbc.Card([
                            dbc.CardHeader("Predicciones de Corrientes", className="bg-primary text-white"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.P([
                                            "Fase 1: ",
                                            html.Strong(f"{predicciones.get('corriente_f1', {}).get('proximo_valor', 0):.1f} A")
                                        ]),
                                        html.Small(f"Tendencia: {predicciones.get('corriente_f1', {}).get('tendencia', 'N/A')}")
                                    ], width=4),
                                    dbc.Col([
                                        html.P([
                                            "Fase 2: ",
                                            html.Strong(f"{predicciones.get('corriente_f2', {}).get('proximo_valor', 0):.1f} A")
                                        ]),
                                        html.Small(f"Tendencia: {predicciones.get('corriente_f2', {}).get('tendencia', 'N/A')}")
                                    ], width=4),
                                    dbc.Col([
                                        html.P([
                                            "Fase 3: ",
                                            html.Strong(f"{predicciones.get('corriente_f3', {}).get('proximo_valor', 0):.1f} A")
                                        ]),
                                        html.Small(f"Tendencia: {predicciones.get('corriente_f3', {}).get('tendencia', 'N/A')}")
                                    ], width=4)
                                ])
                            ])
                        ], className="mb-3"),
                        
                        dbc.Card([
                            dbc.CardHeader("Predicciones de Voltajes", className="bg-info text-white"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.P([
                                            "Ctrl Izquierdo: ",
                                            html.Strong(f"{predicciones.get('voltaje_ctrl_izq', {}).get('proximo_valor', 0):.1f} V")
                                        ]),
                                        html.Small(f"Tendencia: {predicciones.get('voltaje_ctrl_izq', {}).get('tendencia', 'N/A')}")
                                    ], width=6),
                                    dbc.Col([
                                        html.P([
                                            "Ctrl Derecho: ",
                                            html.Strong(f"{predicciones.get('voltaje_ctrl_der', {}).get('proximo_valor', 0):.1f} V")
                                        ]),
                                        html.Small(f"Tendencia: {predicciones.get('voltaje_ctrl_der', {}).get('tendencia', 'N/A')}")
                                    ], width=6)
                                ])
                            ])
                        ], className="mb-3"),
                        
                        dbc.Card([
                            dbc.CardHeader("Mantenimiento", className="bg-secondary text-white"),
                            dbc.CardBody([
                                html.P([
                                    "Próximo mantenimiento en: ",
                                    html.Strong(f"{predicciones.get('proximo_mantenimiento', {}).get('ciclos_hasta_revision', 'N/A')} ciclos")
                                ]),
                                html.P([
                                    "Estado general: ",
                                    html.Strong(predicciones.get('proximo_mantenimiento', {}).get('estado_general', 'N/A'))
                                ])
                            ])
                        ])
                    ]
                else:
                    pred_content = html.P("Recopilando datos para predicciones...")
                print(f"Contenido de predicciones creado correctamente para {maquina_id}")
            except Exception as e:
                print(f"Error creando contenido de predicciones para {maquina_id}: {e}")
                pred_content = html.P("Error en predicciones")
           
            # Crear tarjeta de salud de mantenimiento predictivo
            try:
                health = estado.get("health_status", {})
                
                # Determinar color según estado
                status = health.get("status", "unknown")
                if status == "good":
                    color = "success"
                    icon = "✓"
                elif status == "warning":
                    color = "warning"
                    icon = "⚠️"
                elif status == "critical":
                    color = "danger"
                    icon = "⛔"
                else:
                    color = "secondary"
                    icon = "?"
                
                # Crear componente
                health_card = dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            f"{icon} Estado de Salud ", 
                            dbc.Badge(
                                f"{health.get('health_score', 0):.1f}%", 
                                color=color,
                                className="ms-2"
                            )
                        ], className="d-flex justify-content-between align-items-center")
                    ], className=f"bg-{color} text-white"),
                    
                    dbc.CardBody([
                        html.Div([
                            html.Strong("Diagnóstico: "),
                            html.Span(health.get("message", "Sin datos suficientes"))
                        ], className="mb-2"),
                        
                        html.Div([
                            html.Strong("Próximo mantenimiento: "),
                            html.Span(health.get("maintenance_due", "Desconocido"))
                        ], className="mb-3"),
                        
                        html.H6("Recomendaciones:"),
                        html.Ul([
                            html.Li(rec) for rec in health.get("recommendations", ["Sin recomendaciones"])
                        ], className="small"),
                        
                        html.Div([
                            html.Small(f"Última actualización: {health.get('last_updated', 'Nunca')}")
                        ], className="text-muted mt-2 text-end")
                    ])
                ], className="mb-3")
                print(f"Tarjeta de salud creada correctamente para {maquina_id}")
            except Exception as e:
                print(f"Error creando tarjeta de salud para {maquina_id}: {e}")
                # Crear tarjeta por defecto
                health_card = dbc.Card(dbc.CardBody("Error cargando datos de salud"))
           
            # Crear tabla histórica
            try:
                df = ml_monitor.obtener_ultimas_mediciones(maquina_id)
                if df.empty:
                    tabla = html.P("No hay datos históricos disponibles")
                else:
                    tabla = [
                        html.Thead([
                            html.Tr([
                                html.Th("Timestamp"),
                                html.Th("F1 (A)"),
                                html.Th("F2 (A)"),
                                html.Th("F3 (A)"),
                                html.Th("Ctrl Izq (V)"),
                                html.Th("Ctrl Der (V)"),
                                html.Th("Posición"),
                                html.Th("Progreso")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(row['timestamp']),
                                html.Td(f"{row.get('corriente_f1', row.get('corriente', 0)):.1f}"),
                                html.Td(f"{row.get('corriente_f2', row.get('corriente', 0)):.1f}"),
                                html.Td(f"{row.get('corriente_f3', row.get('corriente', 0)):.1f}"),
                                html.Td(f"{row.get('voltaje_ctrl_izq', row.get('voltaje', 0)/10):.1f}"),
                                html.Td(f"{row.get('voltaje_ctrl_der', row.get('voltaje', 0)/10):.1f}"),
                                html.Td(row['posicion']),
                                html.Td(f"{row['ciclo_progreso']:.1f}%")
                            ]) for _, row in df.iterrows()
                        ])
                    ]
                print(f"Tabla histórica creada correctamente para {maquina_id}")
            except Exception as e:
                print(f"Error creando tabla histórica para {maquina_id}: {e}")
                tabla = html.P("Error cargando datos históricos")
           
            # Indicador de modo de operación
            try:
                modo_operacion = html.Div([
                    dbc.Badge(
                        f"Modo: {estado['modo_operacion'].upper()}",
                        color="warning" if estado['modo_operacion'] == "replay" else "success",
                        className="p-2"
                    )
                ])
                print(f"Indicador de modo creado correctamente para {maquina_id}")
            except Exception as e:
                print(f"Error creando indicador de modo para {maquina_id}: {e}")
                modo_operacion = html.Div("Modo: DESCONOCIDO")
           
            # Actualizar indicador de progreso
            try:
                indicador_progreso = crear_indicador_progreso(estado["ciclo_progreso"])
                print(f"Indicador de progreso creado correctamente para {maquina_id}")
            except Exception as e:
                print(f"Error creando indicador de progreso para {maquina_id}: {e}")
                # Crear un indicador por defecto
                indicador_progreso = go.Figure()
            
            print(f"Actualización UI completada para {maquina_id}")
            return (
                fig_corrientes,
                fig_voltajes,
                estado["posicion"],
                pos_color,
                f"{estado['corriente_f1'][-1]:.1f}" if estado['corriente_f1'] else "N/A",
                f"{estado['corriente_f2'][-1]:.1f}" if estado['corriente_f2'] else "N/A",
                f"{estado['corriente_f3'][-1]:.1f}" if estado['corriente_f3'] else "N/A",
                f"{estado['voltaje_ctrl_izq'][-1]:.1f}" if estado['voltaje_ctrl_izq'] else "N/A",
                f"{estado['voltaje_ctrl_der'][-1]:.1f}" if estado['voltaje_ctrl_der'] else "N/A",
                [html.Li(alert, className="small") for alert in estado["alertas"]],
                indicador_progreso,
                maquina_viz,
                pred_content,
                tabla,
                modo_operacion,
                health_card
            )
            
    except Exception as e:
        import traceback
        print(f"Error completo al actualizar UI para {maquina_id}: {str(e)}")
        print(traceback.format_exc())
        return dash.no_update

# Función de actualización para cada máquina
# Función de actualización para cada máquina
def update_maquina(n_intervals, replay_intervals, maquina_id):
    """Callback de actualización para una máquina específica con los nuevos parámetros"""
    print(f"update_maquina llamada para {maquina_id}: interval={n_intervals}, replay={replay_intervals}")
    
    # Verificar si hay datos antes de llamar a actualizar_ui
    with data_lock:
        estado = estado_maquinas[maquina_id]
        timestamp_len = len(estado["timestamp"]) if "timestamp" in estado and estado["timestamp"] is not None else 0
        print(f"Estado actual de {maquina_id}: {timestamp_len} puntos de datos")
        
        # Si no hay datos, generar algunos valores iniciales para mostrar algo
        if timestamp_len == 0:
            print(f"Sin datos para {maquina_id}, generando valores iniciales")
            now = datetime.now()
            for i in range(5):
                tiempo_simulado = now - timedelta(seconds=i*0.5)
                
                # Valores predeterminados para mostrar algo
                corriente_base = 0.5
                voltaje_base = MAQUINAS[maquina_id]["voltaje_ctrl_nominal"]
                
                # Inicializar las listas si no existen
                if "timestamp" not in estado or estado["timestamp"] is None:
                    estado["timestamp"] = []
                if "corriente_f1" not in estado or estado["corriente_f1"] is None:
                    estado["corriente_f1"] = []
                if "corriente_f2" not in estado or estado["corriente_f2"] is None:
                    estado["corriente_f2"] = []
                if "corriente_f3" not in estado or estado["corriente_f3"] is None:
                    estado["corriente_f3"] = []
                if "voltaje_ctrl_izq" not in estado or estado["voltaje_ctrl_izq"] is None:
                    estado["voltaje_ctrl_izq"] = []
                if "voltaje_ctrl_der" not in estado or estado["voltaje_ctrl_der"] is None:
                    estado["voltaje_ctrl_der"] = []
                
                # Agregar valores
                estado["timestamp"].append(tiempo_simulado)
                estado["corriente_f1"].append(corriente_base * 1.0)
                estado["corriente_f2"].append(corriente_base * 0.95)
                estado["corriente_f3"].append(corriente_base * 1.05)
                
                if estado["posicion"] == "Izquierda":
                    estado["voltaje_ctrl_izq"].append(voltaje_base + 0.1)
                    estado["voltaje_ctrl_der"].append(voltaje_base * 0.05)
                else:
                    estado["voltaje_ctrl_izq"].append(voltaje_base * 0.05)
                    estado["voltaje_ctrl_der"].append(voltaje_base + 0.1)
                
            print(f"Datos iniciales generados para {maquina_id}: {len(estado['timestamp'])} puntos")
    
    return actualizar_ui(n_intervals, replay_intervals, maquina_id)

# =========================================
# Layout principal reorganizado
# =========================================

# Pestaña de monitoreo en tiempo real
def crear_pestaña_monitoreo():
    return html.Div([
        html.H3("Monitoreo en Tiempo Real", className="text-center my-3"),
        dbc.Tabs([
            dbc.Tab(crear_tarjeta_maquina("VIM-11/21"), label="VIM-11/21"),
            dbc.Tab(crear_tarjeta_maquina("TSE-54B"), label="TSE-54B")
        ])
    ])

# Pestaña de sistema de replay
def crear_pestaña_replay():
    return html.Div([
        html.H3("Sistema de Replay de Eventos", className="text-center my-3"),
        create_replay_controls_basculacion(),
        dbc.Row([
            dbc.Col([
                # Elemento para mostrar máquina seleccionada en modo replay
                dbc.Card([
                    dbc.CardHeader("Visualización de Replay"),
                    dbc.CardBody([
                        dbc.Alert(
                            "Selecciona una fecha, máquina y evento de basculación para iniciar la reproducción.",
                            color="info"
                        ),
                        html.Div(id="replay-visualization", className="mt-3")
                    ])
                ])
            ])
        ])
    ])

# Layout principal con pestañas
app.layout = dbc.Container([
    html.H1("Monitor de Máquinas de Cambio", className="text-center my-4"),
    
    # Pestañas principales para separar monitoreo y replay
    dbc.Tabs([
        dbc.Tab(crear_pestaña_monitoreo(), label="Monitoreo en Tiempo Real", tab_id="tab-monitoreo"),
        dbc.Tab(crear_pestaña_replay(), label="Sistema de Replay", tab_id="tab-replay"),
    ], id="tabs-principal"),
    
    # Componentes de intervalo para actualización
    dcc.Interval(
        id='interval-component',
        interval=1000,  # en milisegundos
        n_intervals=0
    ),
    dcc.Interval(
        id='replay-interval-component',
        interval=1000,  # en milisegundos
        n_intervals=0
    )
], fluid=True)

# =========================================
# Callbacks
# =========================================

# Primero define la función fuera del contexto del callback
def update_replay_visualization(selected_machine, n_intervals=None):
    """
    Actualiza la visualización de la máquina en modo replay de manera similar a tiempo real
    """
    if not selected_machine:
        return dbc.Alert("Selecciona una máquina para visualizar", color="warning")
    
    try:
        with data_lock:
            # Verificar si hay una reproducción en curso
            if not replay_system.current_replay or 'machine_id' not in replay_system.current_replay:
                return dbc.Alert("La máquina no está en modo replay. Inicia la reproducción.", color="warning")
            
            # Comprobar si la máquina seleccionada coincide
            if replay_system.current_replay['machine_id'] != selected_machine:
                return dbc.Alert(
                    f"La máquina en reproducción ({replay_system.current_replay['machine_id']}) no coincide con la seleccionada ({selected_machine}).", 
                    color="warning"
                )
            
            # Obtener datos del estado actual
            estado = estado_maquinas[selected_machine]
            current_replay = replay_system.current_replay
            
            # Datos para la visualización
            datos_replay = current_replay['data']
            current_idx = current_replay.get('current_index', 0)
            total_frames = current_replay.get('total_frames', 1)
            
            # Obtener el frame actual
            frame_actual = datos_replay.iloc[current_idx]
            
            # Preparar datos históricos
            datos_historicos = datos_replay.iloc[:current_idx+1]
            
            # Gráficos históricos
            fig_voltaje = go.Figure(
                go.Scatter(
                    x=datos_historicos['timestamp'],
                    y=datos_historicos['voltaje'],
                    mode='lines',
                    line=dict(color='#2196F3'),
                    name='Voltaje'
                )
            )
            fig_voltaje.update_layout(
                title="Tendencia de Voltaje",
                yaxis_title="Voltaje (V)",
                margin=dict(l=30, r=30, t=40, b=30),
                height=200,
                template="plotly_dark",
                xaxis_range=[datos_replay['timestamp'].min(), datos_replay['timestamp'].max()]
            )
            fig_voltaje.add_hline(y=242, line_dash="dot", line_color="red")
            fig_voltaje.add_hline(y=220, line_color="green")

            fig_corriente = go.Figure(
                go.Scatter(
                    x=datos_historicos['timestamp'],
                    y=datos_historicos['corriente'],
                    mode='lines',
                    line=dict(color='#FF9800'),
                    name='Corriente'
                )
            )
            fig_corriente.update_layout(
                title="Tendencia de Corriente",
                yaxis_title="Corriente (A)",
                margin=dict(l=30, r=30, t=40, b=30),
                height=200,
                template="plotly_dark",
                xaxis_range=[datos_replay['timestamp'].min(), datos_replay['timestamp'].max()]
            )
            fig_corriente.add_hline(y=5.5, line_dash="dot", line_color="red")

            return dbc.Card([
                dbc.CardHeader([
                    html.H4(f"Máquina: {selected_machine}", className="card-title"),
                    html.Small(f"Frame {current_idx + 1} de {total_frames}")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        # Columna 1: Indicadores principales y visualización
                        dbc.Col([
                            dbc.Row([
                                dbc.Col([
                                    html.Div("Última posición", className="text-center mb-2"),
                                    html.Div(
                                        frame_actual['posicion'],
                                        className="h3 text-center",
                                        style={'color': '#4CAF50' if frame_actual['posicion'] == "Izquierda" else '#FF5722'}
                                    )
                                ], width=6),
                                
                                dbc.Col([
                                    dcc.Graph(
                                        figure=crear_indicador_progreso(frame_actual['ciclo_progreso']),
                                        config={'displayModeBar': False},
                                        style={'height': '120px'}
                                    )
                                ], width=6)
                            ]),
                            
                            # Visualización de la máquina de cambio
                            html.Div(
                                crear_svg_maquina(frame_actual['posicion'], frame_actual['ciclo_progreso']),
                                className="mt-4"
                            ),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Div("Voltaje (V)", className="text-center"),
                                    html.Div(
                                        f"{frame_actual['voltaje']:.1f}",
                                        className="h2 text-center",
                                        style={'color': '#2196F3'}
                                    )
                                ], width=6),
                                
                                dbc.Col([
                                    html.Div("Corriente (A)", className="text-center"),
                                    html.Div(
                                        f"{frame_actual['corriente']:.1f}",
                                        className="h2 text-center",
                                        style={'color': '#FF9800'}
                                    )
                                ], width=6)
                            ], className="mt-4")
                        ], width=4),
                        
                        # Columna 2: Gráficos 
                        dbc.Col([
                            dcc.Graph(
                                figure=fig_voltaje
                            ),
                            dcc.Graph(
                                figure=fig_corriente
                            )
                        ], width=8)
                    ])
                ])
            ])
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"Error al visualizar el replay: {str(e)}", color="danger")

# Modificar el callback para quitar n_intervals
@app.callback(
    Output("replay-visualization", "children"),
    [Input('replay-maquina', 'value'),
     Input('replay-interval-component', 'n_intervals')]
)
def update_replay_visualization_callback(selected_machine, n_intervals):
    """Callback wrapper para actualizar la visualización de replay"""
    return update_replay_visualization(selected_machine, n_intervals)


# De manera similar, define primero la función sync_replay_on_play y luego registra el callback
def sync_replay_on_play(n_clicks, selected_machine):
    """
    Actualiza la visualización inmediatamente después de iniciar la reproducción
    """
    if not n_clicks or not selected_machine:
        return dash.no_update
    
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    # Solo responder al botón de reproducción
    if ctx.triggered[0]['prop_id'] == 'replay-play-button.n_clicks':
        try:
            # Usar data_lock para asegurar acceso exclusivo a los datos
            with data_lock:
                # Verificar si la máquina está en modo replay
                if selected_machine in estado_maquinas:
                    estado = estado_maquinas[selected_machine]
                    
                    # Dar un poco de tiempo para que se establezca el estado
                    if estado["modo_operacion"] == "replay":
                        print(f"Forzando actualización de visualización para {selected_machine} después de iniciar reproducción")
                        
                        # Llamar directamente a la función de visualización
                        return update_replay_visualization(selected_machine, 0)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
    return dash.no_update

# Luego, registra el callback usando la función definida
@app.callback(
    Output("replay-visualization", "children", allow_duplicate=True),
    [Input('replay-play-button', 'n_clicks')],
    [State('replay-maquina', 'value')],
    prevent_initial_call=True
)
def sync_replay_on_play_callback(n_clicks, selected_machine):
    """Callback wrapper para sincronizar la visualización de replay cuando se inicia"""
    return sync_replay_on_play(n_clicks, selected_machine)


def registrar_callbacks():
    """Registra todos los callbacks necesarios para el dashboard con los nuevos parámetros"""
    # Callbacks para actualizar cada máquina
    for maquina_id in MAQUINAS:
        # Callback principal que actualiza la interfaz periódicamente
        app.callback(
            [Output(f"corrientes-graph-{maquina_id}", "figure", allow_duplicate=True),
             Output(f"voltajes-graph-{maquina_id}", "figure", allow_duplicate=True),
             Output(f"posicion-{maquina_id}", "children", allow_duplicate=True),
             Output(f"posicion-{maquina_id}", "style", allow_duplicate=True),
             Output(f"corriente-f1-{maquina_id}", "children", allow_duplicate=True),
             Output(f"corriente-f2-{maquina_id}", "children", allow_duplicate=True),
             Output(f"corriente-f3-{maquina_id}", "children", allow_duplicate=True),
             Output(f"voltaje-ctrl-izq-{maquina_id}", "children", allow_duplicate=True),
             Output(f"voltaje-ctrl-der-{maquina_id}", "children", allow_duplicate=True),
             Output(f"alertas-{maquina_id}", "children", allow_duplicate=True),
             Output(f"ciclo-progreso-{maquina_id}", "figure", allow_duplicate=True),
             Output(f"maquina-cambio-{maquina_id}", "children", allow_duplicate=True),
             Output(f"predicciones-{maquina_id}", "children", allow_duplicate=True),
             Output(f"tabla-historico-{maquina_id}", "children", allow_duplicate=True),
             Output(f"modo-operacion-{maquina_id}", "children", allow_duplicate=True),
             Output(f"health-card-{maquina_id}", "children", allow_duplicate=True)],
            [Input('interval-component', 'n_intervals'),
             Input('replay-interval-component', 'n_intervals')],
            prevent_initial_call=True
        )(lambda n_intervals, replay_intervals, mid=maquina_id: update_maquina(n_intervals, replay_intervals, mid))
        
        # NUEVO: Callback para la actualización inicial forzada - se dispara al cargar
        app.callback(
            [Output(f"corriente-f1-{maquina_id}", "children"),
             Output(f"corriente-f2-{maquina_id}", "children"),
             Output(f"corriente-f3-{maquina_id}", "children"),
             Output(f"voltaje-ctrl-izq-{maquina_id}", "children"),
             Output(f"voltaje-ctrl-der-{maquina_id}", "children")],
            [Input('tabs-principal', 'value')],  # Se dispara al cargar las pestañas
            prevent_initial_call=False  # Importante: permitir llamada inicial
        )(lambda tabs_value, mid=maquina_id: datos_iniciales_valores(mid))
        
def datos_iniciales_valores(maquina_id):
    """Proporciona los valores iniciales para la visualización"""
    with data_lock:
        estado = estado_maquinas[maquina_id]
        
        # Si hay datos disponibles, usar los últimos valores
        if len(estado.get("corriente_f1", [])) > 0:
            return (
                f"{estado['corriente_f1'][-1]:.1f}",
                f"{estado['corriente_f2'][-1]:.1f}",
                f"{estado['corriente_f3'][-1]:.1f}",
                f"{estado['voltaje_ctrl_izq'][-1]:.1f}",
                f"{estado['voltaje_ctrl_der'][-1]:.1f}"
            )
        else:
            # En caso extremo, proporcionar valores predeterminados
            return ("0.5", "0.5", "0.5", "24.0", "1.2")
# Registrar callbacks del sistema de replay
register_replay_callbacks_basculacion(app, replay_system)

# =========================================
# Iniciar la aplicación
# =========================================

def inicializar_datos_prueba():
    """Inicializa datos de prueba para asegurar visualización inmediata"""
    print("Inicializando datos de prueba para visualización inmediata...")
    
    for maquina_id, estado in estado_maquinas.items():
        now = datetime.now()
        config = MAQUINAS[maquina_id]
        
        # Verificar si ya hay datos
        if len(estado.get("timestamp", [])) > 0:
            print(f"Máquina {maquina_id} ya tiene datos, omitiendo inicialización")
            continue
            
        # Generar 20 puntos de datos históricos
        for i in range(20):
            tiempo = now - timedelta(seconds=i*1)
            
            # Valores realistas para simular el comportamiento
            if i % 6 < 3:  # Alternar entre espera y movimiento
                corriente_base = 0.3  # Menor corriente en espera
                prog = 0
            else:
                corriente_base = 4.0  # Mayor corriente en movimiento
                prog = 50 if i % 12 < 6 else 80
            
            # Aplicar factores para crear variaciones entre fases
            estado["timestamp"].append(tiempo)
            estado["corriente_f1"].append(corriente_base * (1.0 + random.uniform(-0.1, 0.1)))
            estado["corriente_f2"].append(corriente_base * (0.95 + random.uniform(-0.1, 0.1)))
            estado["corriente_f3"].append(corriente_base * (1.05 + random.uniform(-0.1, 0.1)))
            
            # Voltajes de los controladores
            if estado["posicion"] == "Izquierda":
                estado["voltaje_ctrl_izq"].append(24.0 + random.uniform(-0.5, 0.5))
                estado["voltaje_ctrl_der"].append(1.2 + random.uniform(-0.1, 0.1))
            else:
                estado["voltaje_ctrl_izq"].append(1.2 + random.uniform(-0.1, 0.1))
                estado["voltaje_ctrl_der"].append(24.0 + random.uniform(-0.5, 0.5))
            
            # Guardar en la base de datos para consistencia
            try:
                ml_monitor.guardar_medicion(maquina_id, {
                    "timestamp": tiempo,
                    "corriente_f1": estado["corriente_f1"][-1],
                    "corriente_f2": estado["corriente_f2"][-1],
                    "corriente_f3": estado["corriente_f3"][-1],
                    "voltaje_ctrl_izq": estado["voltaje_ctrl_izq"][-1],
                    "voltaje_ctrl_der": estado["voltaje_ctrl_der"][-1],
                    "posicion": estado["posicion"],
                    "ciclo_progreso": prog
                })
            except Exception as e:
                print(f"Error al guardar dato inicial: {e}")
        
        # También actualizar el estado actual con el último valor
        estado["ciclo_progreso"] = prog
        
        print(f"Datos de prueba generados para {maquina_id}: {len(estado['timestamp'])} puntos")
        print(f"Valores actuales - F1: {estado['corriente_f1'][-1]:.2f}A, "
              f"F2: {estado['corriente_f2'][-1]:.2f}A, F3: {estado['corriente_f3'][-1]:.2f}A")
        print(f"Voltajes - Izq: {estado['voltaje_ctrl_izq'][-1]:.2f}V, "
              f"Der: {estado['voltaje_ctrl_der'][-1]:.2f}V")

# Modifica la parte final del código para llamar a esta función
if __name__ == '__main__':
    # Inicializar datos de prueba antes de arrancar los hilos
    inicializar_datos_prueba()
    
    # Iniciar hilos de simulación
    for maquina_id in MAQUINAS:
        Thread(target=simular_maquina, args=(maquina_id,), daemon=True).start()
    
    # Iniciar servidor
    app.run_server(debug=True, port=8050)