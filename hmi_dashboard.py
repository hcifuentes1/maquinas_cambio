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

# Inicializar el monitor ML y sistema de replay
ml_monitor = MonitoringML()
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
    """Simula el comportamiento de una máquina de cambio"""
    config = MAQUINAS[maquina_id]
    estado = estado_maquinas[maquina_id]
    tiempo_espera = 3  # segundos de espera antes del movimiento
   
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
                    
        except Exception as e:
            import traceback
            print(f"Error en simulación de {maquina_id}: {str(e)}")
            print(traceback.format_exc())
            
        time.sleep(0.5)

# =========================================
# Interfaz de usuario profesional
# =========================================

def crear_tarjeta_maquina(maquina_id):
    """Crea una tarjeta con la información de una máquina de cambio"""
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

# =========================================
# Funciones de actualización de UI
# =========================================

def actualizar_ui(n_intervals, replay_intervals, maquina_id):
    """Función de actualización de la interfaz de usuario"""
    try:
        with data_lock:
            # Añadir depuración para ver si se llama y con qué valores
            print(f"actualizar_ui llamada: n_intervals={n_intervals}, replay_intervals={replay_intervals}, maquina={maquina_id}")
            
            estado = estado_maquinas[maquina_id]
            if not estado["timestamp"]:
                print(f"No hay datos de timestamp para {maquina_id}, omitiendo actualización")
                return dash.no_update
            
            # Verificar si estamos en modo reproducción
            if estado["modo_operacion"] == "replay":
                print(f"Actualizando UI en modo replay: {maquina_id}, Pos={estado['posicion']}, Prog={estado['ciclo_progreso']}")
            
            # Gráfico de voltaje
            try:
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
                print(f"Gráfico de voltaje creado correctamente para {maquina_id}")
            except Exception as e:
                print(f"Error creando gráfico de voltaje para {maquina_id}: {e}")
                fig_voltaje = go.Figure()
           
            # Gráfico de corriente
            try:
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
                print(f"Gráfico de corriente creado correctamente para {maquina_id}")
            except Exception as e:
                print(f"Error creando gráfico de corriente para {maquina_id}: {e}")
                fig_corriente = go.Figure()
           
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
                print(f"Contenido de predicciones creado correctamente para {maquina_id}")
            except Exception as e:
                print(f"Error creando contenido de predicciones para {maquina_id}: {e}")
                pred_content = html.P("Error en predicciones")
           
            # Crear tabla histórica
            try:
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
                fig_voltaje,
                fig_corriente,
                estado["posicion"],
                pos_color,
                f"{estado['voltaje'][-1]:.1f}" if estado['voltaje'] else "N/A",
                f"{estado['corriente'][-1]:.1f}" if estado['corriente'] else "N/A",
                [html.Li(alert, className="small") for alert in estado["alertas"]],
                indicador_progreso,
                maquina_viz,
                pred_content,
                tabla,
                modo_operacion
            )
            
    except Exception as e:
        import traceback
        print(f"Error completo al actualizar UI para {maquina_id}: {str(e)}")
        print(traceback.format_exc())
        return dash.no_update

# Función de actualización para cada máquina
def update_maquina(n_intervals, replay_intervals, maquina_id):
    """Callback de actualización para una máquina específica"""
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
    """Registra todos los callbacks necesarios para el dashboard"""
    # Callbacks para actualizar cada máquina
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

# Registrar callbacks del sistema de replay
register_replay_callbacks_basculacion(app, replay_system)

# =========================================
# Iniciar la aplicación
# =========================================

if __name__ == '__main__':
    # Iniciar hilos de simulación
    for maquina_id in MAQUINAS:
        Thread(target=simular_maquina, args=(maquina_id,), daemon=True).start()
    
    app.run_server(debug=True, port=8050)