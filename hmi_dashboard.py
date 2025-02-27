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
#from replay_system import ReplaySystem, create_replay_controls, register_replay_callbacks
from replay_system_basculacion import ReplaySystemBasculacion, create_replay_controls_basculacion, register_replay_callbacks_basculacion

# Inicializar el monitor ML y sistema de replay
ml_monitor = MonitoringML()
# replay_system = ReplaySystem(ml_monitor.db_path)
replay_system = ReplaySystemBasculacion(ml_monitor.db_path)

# Mutex para sincronización de datos
data_lock = Lock()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Monitor de Máquinas de Cambio - Metro de Santiago"

# =========================================
# Configuración de simulación realista
# =========================================

MAQUINAS = {
    "VIM-11/21": {
        "ubicacion": "Línea 4A, Estación Vicuña Mackena",
        "voltaje_nominal": 220,
        "corriente_maxima": 5.0,
        "ciclo_duracion": 6  # segundos
    },
    "TSE-54B": {
        "ubicacion": "Talleres San Eugenio",
        "voltaje_nominal": 220,
        "corriente_maxima": 5.0,
        "ciclo_duracion": 6
    }
}

# Estado inicial de las máquinas con campos adicionales para ML
estado_maquinas = {
    "VIM-11/21": {
        "timestamp": [],
        "voltaje": [],
        "corriente": [],
        "posicion": "Izquierda",
        "ciclo_progreso": 0,
        "alertas": [],
        "predicciones": {},
        "estadisticas": {},
        "modo_operacion": "normal"  # normal/replay
    },
    "TSE-54B": {
        "timestamp": [],
        "voltaje": [],
        "corriente": [],
        "posicion": "Derecha",
        "ciclo_progreso": 0,
        "alertas": [],
        "predicciones": {},
        "estadisticas": {},
        "modo_operacion": "normal"  # normal/replay
    }
}

# =========================================
# Simulación de comportamiento realista
# =========================================

def simular_maquina(maquina_id):
    config = MAQUINAS[maquina_id]
    estado = estado_maquinas[maquina_id]
    tiempo_espera = 3  # segundos de espera antes del movimiento
   
    while True:
        with data_lock:
            # Verificar si estamos en modo replay
            if replay_system.current_replay:
                # Marcar modo de operación como replay
                estado["modo_operacion"] = "replay"
                print(f"Modo replay activo para {maquina_id}, índice: {replay_system.current_replay.get('current_index')}")
                
                # Verificar si replay_system tiene los datos necesarios
                if 'data' not in replay_system.current_replay:
                    print(f"Error: replay_system.current_replay no tiene 'data' para {maquina_id}")
                    time.sleep(0.5)
                    continue
                
                # Obtener datos del frame actual (si existe)
                if replay_system.current_replay.get('current_index') is not None:
                    current_idx = replay_system.current_replay['current_index']
                    
                    # Verificar que el índice está dentro del rango
                    if current_idx < len(replay_system.current_replay['data']):
                        # Obtener la fila actual
                        row = replay_system.current_replay['data'].iloc[current_idx]
                        
                        # Actualizar estado con datos del registro actual
                        # IMPORTANTE: sobrescribir los valores si están en modo replay
                        if len(estado["timestamp"]) > 0:
                            estado["timestamp"][-1] = row['timestamp']  # Sobrescribir último timestamp
                            estado["voltaje"][-1] = row['voltaje']      # Sobrescribir último voltaje
                            estado["corriente"][-1] = row['corriente']  # Sobrescribir última corriente
                        else:
                            estado["timestamp"].append(row['timestamp'])
                            estado["voltaje"].append(row['voltaje'])
                            estado["corriente"].append(row['corriente'])
                        
                        # Actualizar posición y progreso directamente
                        estado["posicion"] = row['posicion']
                        estado["ciclo_progreso"] = row['ciclo_progreso']
                        
                        print(f"Actualizado estado {maquina_id}: Pos={estado['posicion']}, Prog={estado['ciclo_progreso']}")
                
                # Si la reproducción está pausada, no avanzar
                if not replay_system.current_replay.get('paused', False):
                    # La velocidad y el avance se manejan en el callback del intervalo
                    pass
                
                # Tiempo de espera para el bucle de simulación en modo replay
                # Ajustado según la velocidad de reproducción
                speed = replay_system.current_replay.get('speed', 1.0)
                wait_time = 1.0 / max(0.1, speed)  # Evitar división por cero o valores negativos
                time.sleep(wait_time)
                continue
           
            # Modo normal de simulación
            estado["modo_operacion"] = "normal"
            now = datetime.now()
            tiempo_total = config["ciclo_duracion"] + tiempo_espera
            tiempo_actual = time.time() % tiempo_total
           
            # Fase de espera
            if tiempo_actual < tiempo_espera:
                fase = 0
            else:
                # Fase de movimiento
                fase = (tiempo_actual - tiempo_espera) / config["ciclo_duracion"]
           
            # Simular voltaje con variación normal
            voltaje = config["voltaje_nominal"] + random.uniform(-3, 3)
            voltaje += 2 * (1 - abs(fase - 0.5))
           
            # Simular corriente durante el movimiento
            if fase == 0:  # En espera
                corriente = random.uniform(0.2, 0.5)
            elif fase < 0.2 or fase > 0.8:  # Inicio/fin de movimiento
                corriente = random.uniform(0.5, 1.5)
            else:  # Durante el movimiento
                corriente = config["corriente_maxima"] * (0.8 + random.uniform(-0.1, 0.3))
           
            # Determinar posición
            ciclo_completo = (time.time() // tiempo_total) % 2
            estado["posicion"] = "Derecha" if ciclo_completo else "Izquierda"
           
            # Generar alertas básicas
            alerta = ""
            if voltaje > 242:
                alerta = f"ALERTA: Sobretensión ({voltaje:.1f}V)"
            elif corriente > 5.5:
                alerta = f"ALERTA: Sobrecorriente ({corriente:.1f}A)"
           
            # Actualizar estado
            estado["timestamp"].append(now)
            estado["voltaje"].append(voltaje)
            estado["corriente"].append(corriente)
            estado["ciclo_progreso"] = fase * 100 if fase > 0 else 0
           
            if alerta:
                estado["alertas"].append(f"{now.strftime('%H:%M:%S')} - {alerta}")
           
            # Guardar datos en la base de datos
            ml_monitor.guardar_medicion(maquina_id, {
                "timestamp": now,
                "voltaje": voltaje,
                "corriente": corriente,
                "posicion": estado["posicion"],
                "ciclo_progreso": estado["ciclo_progreso"]
            })
           
            # Actualizar predicciones y estadísticas cada 10 segundos
            if len(estado["timestamp"]) % 20 == 0:
                estado["predicciones"] = ml_monitor.predecir_tendencias(maquina_id)
                estado["estadisticas"] = ml_monitor.obtener_estadisticas(maquina_id)
           
            # Mantener solo los últimos N puntos para la visualización
            max_points = 100
            if len(estado["timestamp"]) > max_points:
                estado["timestamp"] = estado["timestamp"][-max_points:]
                estado["voltaje"] = estado["voltaje"][-max_points:]
                estado["corriente"] = estado["corriente"][-max_points:]
                estado["alertas"] = estado["alertas"][-10:]  # Mantener solo las últimas 10 alertas
       
        time.sleep(0.5)

# =========================================
# Interfaz de usuario profesional
# =========================================

def crear_tarjeta_maquina(maquina_id):
    config = MAQUINAS[maquina_id]
    estado = estado_maquinas[maquina_id]
   
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
                   
                    dbc.Row([
                        dbc.Col([
                            html.Div("Voltaje (V)", className="text-center"),
                            html.Div(
                                f"{estado['voltaje'][-1]:.1f}" if estado['voltaje'] else "N/A",
                                id=f"voltaje-actual-{maquina_id}",
                                className="h2 text-center",
                                style={'color': '#2196F3'}
                            )
                        ], width=6),
                       
                        dbc.Col([
                            html.Div("Corriente (A)", className="text-center"),
                            html.Div(
                                f"{estado['corriente'][-1]:.1f}" if estado['corriente'] else "N/A",
                                id=f"corriente-actual-{maquina_id}",
                                className="h2 text-center",
                                style={'color': '#FF9800'}
                            )
                        ], width=6)
                    ], className="mt-4")
                ], width=4),
               
                # Columna 2: Gráficos y predicciones
                dbc.Col([
                    # Modo de operación
                    html.Div(
                        id=f"modo-operacion-{maquina_id}",
                        className="text-center mb-3"
                    ),
                   
                    dcc.Graph(id=f"voltaje-graph-{maquina_id}"),
                    dcc.Graph(id=f"corriente-graph-{maquina_id}"),
                   
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

# Layout principal
app.layout = dbc.Container([
    html.H1("Monitor de Máquinas de Cambio", className="text-center my-4"),
   
    # Sistema de Replay
    #create_replay_controls()
    create_replay_controls_basculacion()
    ,
   
    # Contenido principal
    dbc.Tabs([
        dbc.Tab(crear_tarjeta_maquina("VIM-11/21"), label="VIM-11/21"),
        dbc.Tab(crear_tarjeta_maquina("TSE-54B"), label="TSE-54B")
    ])
], fluid=True)

# =========================================
# Callbacks
# =========================================


def actualizar_ui(n_intervals, replay_intervals, maquina_id):
    """Función de actualización de la interfaz de usuario"""
    with data_lock:
        try:
            # Añadir depuración para ver si se llama y con qué valores
            print(f"actualizar_ui llamada: n_intervals={n_intervals}, replay_intervals={replay_intervals}, maquina={maquina_id}")
            
            estado = estado_maquinas[maquina_id]
            if not estado["timestamp"]:
                return dash.no_update
            
            # Verificar si estamos en modo reproducción
            if estado["modo_operacion"] == "replay":
                print(f"Actualizando UI en modo replay: {maquina_id}, Pos={estado['posicion']}, Prog={estado['ciclo_progreso']}")
                # Añadir lógica específica para forzar actualización en modo replay
            
            # Gráfico de voltaje
            fig_voltaje = go.Figure(
                go.Scatter(
                    x=estado["timestamp"],
                    y=estado["voltaje"],
                    line=dict(color='#2196F3'),
                    name='Voltaje'
                )
            )
            fig_voltaje.update_layout(
                title="Tendencia de Voltaje",
                yaxis_title="Voltaje (V)",
                margin=dict(l=30, r=30, t=40, b=30),
                height=200,
                template="plotly_dark"
            )
            fig_voltaje.add_hline(y=242, line_dash="dot", line_color="red")
            fig_voltaje.add_hline(y=220, line_color="green")
           
            # Gráfico de corriente
            fig_corriente = go.Figure(
                go.Scatter(
                    x=estado["timestamp"],
                    y=estado["corriente"],
                    line=dict(color='#FF9800'),
                    name='Corriente'
                )
            )
            fig_corriente.update_layout(
                title="Tendencia de Corriente",
                yaxis_title="Corriente (A)",
                margin=dict(l=30, r=30, t=40, b=30),
                height=200,
                template="plotly_dark"
            )
            fig_corriente.add_hline(y=5.5, line_dash="dot", line_color="red")
           
            # Indicador de posición
            pos_color = {'color': '#4CAF50'} if estado["posicion"] == "Izquierda" else {'color': '#FF5722'}
           
            # Actualizar visualización de la máquina
            maquina_viz = crear_svg_maquina(
                estado["posicion"],
                estado["ciclo_progreso"]
            )
           
            # Crear contenido de predicciones
            predicciones = estado.get("predicciones", {})
            if predicciones:
                pred_content = [
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Predicciones", className="mb-3"),
                            html.P([
                                "Próximo voltaje esperado: ",
                                html.Strong(f"{predicciones['voltaje']['proximo_valor']:.1f} V")
                            ]),
                            html.P([
                                "Próxima corriente esperada: ",
                                html.Strong(f"{predicciones['corriente']['proximo_valor']:.1f} A")
                            ]),
                            html.Hr(),
                            html.H6("Estado del Sistema"),
                            html.P([
                                "Tendencia voltaje: ",
                                html.Strong(predicciones['voltaje']['tendencia'])
                            ]),
                            html.P([
                                "Tendencia corriente: ",
                                html.Strong(predicciones['corriente']['tendencia'])
                            ]),
                            html.P([
                                "Próximo mantenimiento en: ",
                                html.Strong(f"{predicciones.get('proximo_mantenimiento', {}).get('ciclos_hasta_revision', 'N/A')} ciclos")
                            ])
                        ])
                    ])
                ]
            else:
                pred_content = html.P("Recopilando datos para predicciones...")
           
            # Crear tabla histórica
            df = ml_monitor.obtener_ultimas_mediciones(maquina_id)
            tabla = [
                html.Thead([
                    html.Tr([
                        html.Th("Timestamp"),
                        html.Th("Voltaje (V)"),
                        html.Th("Corriente (A)"),
                        html.Th("Posición"),
                        html.Th("Progreso")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(row['timestamp']),
                        html.Td(f"{row['voltaje']:.1f}"),
                        html.Td(f"{row['corriente']:.1f}"),
                        html.Td(row['posicion']),
                        html.Td(f"{row['ciclo_progreso']:.1f}%")
                    ]) for _, row in df.iterrows()
                ])
            ]
           
            # Indicador de modo de operación
            modo_operacion = html.Div([
                dbc.Badge(
                    f"Modo: {estado['modo_operacion'].upper()}",
                    color="warning" if estado['modo_operacion'] == "replay" else "success",
                    className="p-2"
                )
            ])
           
            return (
                fig_voltaje,
                fig_corriente,
                estado["posicion"],
                pos_color,
                f"{estado['voltaje'][-1]:.1f}" if estado['voltaje'] else "N/A",
                f"{estado['corriente'][-1]:.1f}" if estado['corriente'] else "N/A",
                [html.Li(alert, className="small") for alert in estado["alertas"]],
                crear_indicador_progreso(estado["ciclo_progreso"]),
                maquina_viz,
                pred_content,
                tabla,
                modo_operacion
            )
        except Exception as e:
            print(f"Error actualizando UI para {maquina_id}: {str(e)}")
            return dash.no_update
       
# Función de actualización para cada máquina
def update_maquina(n_intervals, replay_intervals, maquina_id):
    """Callback de actualización para una máquina específica"""
    return actualizar_ui(n_intervals, replay_intervals, maquina_id)

def registrar_callbacks():
    """Registra todos los callbacks necesarios para el dashboard"""
    for maquina_id in MAQUINAS:
        app.callback(
            [Output(f"voltaje-graph-{maquina_id}", "figure", allow_duplicate=True),
             Output(f"corriente-graph-{maquina_id}", "figure", allow_duplicate=True),
             Output(f"posicion-{maquina_id}", "children", allow_duplicate=True),
             Output(f"posicion-{maquina_id}", "style", allow_duplicate=True),
             Output(f"voltaje-actual-{maquina_id}", "children", allow_duplicate=True),
             Output(f"corriente-actual-{maquina_id}", "children", allow_duplicate=True),
             Output(f"alertas-{maquina_id}", "children", allow_duplicate=True),
             Output(f"ciclo-progreso-{maquina_id}", "figure", allow_duplicate=True),
             Output(f"maquina-cambio-{maquina_id}", "children", allow_duplicate=True),
             Output(f"predicciones-{maquina_id}", "children", allow_duplicate=True),
             Output(f"tabla-historico-{maquina_id}", "children", allow_duplicate=True),
             Output(f"modo-operacion-{maquina_id}", "children", allow_duplicate=True)],
            [Input('interval-component', 'n_intervals'),
             Input('replay-interval-component', 'n_intervals')],
            prevent_initial_call=True
        )(lambda n_intervals, replay_intervals, mid=maquina_id: update_maquina(n_intervals, replay_intervals, mid))

# Registrar todos los callbacks
registrar_callbacks()



# =========================================

# Registrar callbacks del sistema de replay
#register_replay_callbacks(app, replay_system)
register_replay_callbacks_basculacion(app, replay_system)

# Agregar el componente de intervalo para actualización periódica
if not any(isinstance(child, dcc.Interval) for child in app.layout.children):
    app.layout.children.append(
        dcc.Interval(
            id='interval-component',
            interval=1000,  # en milisegundos
            n_intervals=0
        )
    )

# Iniciar hilos de simulación
for maquina_id in MAQUINAS:
    Thread(target=simular_maquina, args=(maquina_id,), daemon=True).start()

# =========================================
# Iniciar la aplicación
# =========================================

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
