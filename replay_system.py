# replay_system.py

from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import pandas as pd
import sqlite3

class ReplaySystem:
    def __init__(self, db_path):
        self.db_path = db_path
        self.current_replay = None
       
    def get_available_dates(self):
        """Obtiene las fechas disponibles en la base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT DISTINCT date(timestamp) as fecha
                FROM mediciones
                ORDER BY fecha DESC
            """
            dates = pd.read_sql_query(query, conn)
            conn.close()
            return dates['fecha'].tolist()
        except Exception as e:
            print(f"Error obteniendo fechas: {e}")
            return []

    def get_data_range(self, start_datetime, end_datetime, maquina_id=None):
        """Obtiene los datos para un rango de tiempo específico"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT *
                FROM mediciones
                WHERE timestamp BETWEEN ? AND ?
            """
            params = [start_datetime, end_datetime]
           
            if maquina_id:
                query += " AND maquina_id = ?"
                params.append(maquina_id)
           
            query += " ORDER BY timestamp ASC"
           
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
           
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error obteniendo datos: {e}")
            return pd.DataFrame()

def create_replay_controls():
    """Crea los controles para el sistema de replay"""
    controls = dbc.Card([
        dbc.CardHeader(html.H4("Control de Replay", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                # Selector de fecha inicial
                dbc.Col([
                    html.Label("Fecha Inicial"),
                    dcc.DatePickerSingle(
                        id='replay-start-date',
                        display_format='DD/MM/YYYY',
                        className="mb-2"
                    ),
                    html.Label("Hora Inicial (HH:MM)"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Input(
                                id='replay-start-hour',
                                type='number',
                                min=0,
                                max=23,
                                value=0,
                                className="form-control",
                                placeholder="Hora"
                            )
                        ], width=6),
                        dbc.Col([
                            dcc.Input(
                                id='replay-start-minute',
                                type='number',
                                min=0,
                                max=59,
                                value=0,
                                className="form-control",
                                placeholder="Minuto"
                            )
                        ], width=6)
                    ])
                ], md=3),
               
                # Selector de fecha final
                dbc.Col([
                    html.Label("Fecha Final"),
                    dcc.DatePickerSingle(
                        id='replay-end-date',
                        display_format='DD/MM/YYYY',
                        className="mb-2"
                    ),
                    html.Label("Hora Final (HH:MM)"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Input(
                                id='replay-end-hour',
                                type='number',
                                min=0,
                                max=23,
                                value=23,
                                className="form-control",
                                placeholder="Hora"
                            )
                        ], width=6),
                        dbc.Col([
                            dcc.Input(
                                id='replay-end-minute',
                                type='number',
                                min=0,
                                max=59,
                                value=59,
                                className="form-control",
                                placeholder="Minuto"
                            )
                        ], width=6)
                    ])
                ], md=3),
               
                # Control de velocidad
                dbc.Col([
                    html.Label("Velocidad de Reproducción"),
                    dcc.Slider(
                        id='replay-speed',
                        min=0.1,
                        max=10,
                        step=0.1,
                        value=1,
                        marks={
                            0.1: '0.1x',
                            1: '1x',
                            2: '2x',
                            5: '5x',
                            10: '10x'
                        }
                    )
                ], md=4),
               
                # Controles de reproducción
                dbc.Col([
                    html.Label("Controles"),
                    html.Div([
                        dbc.Button(
                            "▶️ Reproducir",
                            id="replay-play-button",
                            n_clicks=0,
                            color="success",
                            className="me-2"
                        ),
                        dbc.Button(
                            "⏸️ Pausar",
                            id="replay-pause-button",
                            n_clicks=0,
                            color="warning",
                            className="me-2"
                        ),
                        dbc.Button(
                            "⏹️ Detener",
                            id="replay-stop-button",
                            n_clicks=0,
                            color="danger"
                        )
                    ], className="d-flex")
                ], md=2)
            ]),
           
            # Barra de progreso
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Progress(
                            id="replay-progress",
                            value=0,
                            striped=True,
                            animated=True,
                            className="mt-3"
                        ),
                        html.Div(
                            id="replay-time-display",
                            className="text-center mt-2"
                        )
                    ])
                ])
            ])
        ])
    ], className="mb-4")
   
    # Agregar el componente de intervalo específico para el replay
    controls.children.append(
        dcc.Interval(
            id='replay-interval-component',
            interval=1000,  # en milisegundos
            n_intervals=0
        )
    )
   
    return controls

def register_replay_callbacks(app, replay_system):
    """Registra los callbacks del sistema de replay"""
   
    @app.callback(
        [Output('replay-time-display', 'children', allow_duplicate=True),
         Output('replay-progress', 'value', allow_duplicate=True),
         Output('replay-start-date', 'min_date_allowed', allow_duplicate=True),
         Output('replay-start-date', 'max_date_allowed', allow_duplicate=True),
         Output('replay-end-date', 'min_date_allowed', allow_duplicate=True),
         Output('replay-end-date', 'max_date_allowed', allow_duplicate=True),
         Output('replay-start-date', 'initial_visible_month', allow_duplicate=True),
         Output('replay-end-date', 'initial_visible_month', allow_duplicate=True)],
        [Input('replay-play-button', 'n_clicks'),
         Input('replay-pause-button', 'n_clicks'),
         Input('replay-stop-button', 'n_clicks'),
         Input('replay-interval-component', 'n_intervals')],
        [State('replay-start-date', 'date'),
         State('replay-start-hour', 'value'),
         State('replay-start-minute', 'value'),
         State('replay-end-date', 'date'),
         State('replay-end-hour', 'value'),
         State('replay-end-minute', 'value'),
         State('replay-speed', 'value')],
        prevent_initial_call=True
    )
    def update_replay_status(n_play, n_pause, n_stop, n_intervals,
                           start_date, start_hour, start_minute,
                           end_date, end_hour, end_minute, speed):
        ctx = callback_context
        available_dates = replay_system.get_available_dates()
       
        # Valores por defecto para las fechas del DatePicker
        if available_dates:
            min_date = datetime.strptime(min(available_dates), '%Y-%m-%d').date()
            max_date = datetime.strptime(max(available_dates), '%Y-%m-%d').date()
            date_values = [min_date, max_date, min_date, max_date, max_date, max_date]
        else:
            date_values = [None] * 6

        # Si no hay trigger, retorna valores iniciales
        if not ctx.triggered:
            return ["Esperando inicio...", 0] + date_values

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
       
        try:
            if trigger_id == 'replay-play-button' and start_date and end_date:
                # Convertir los valores a int con manejo de None
                start_hour = int(start_hour) if start_hour is not None else 0
                start_minute = int(start_minute) if start_minute is not None else 0
                end_hour = int(end_hour) if end_hour is not None else 23
                end_minute = int(end_minute) if end_minute is not None else 59
               
                # Crear objetos datetime
                start_dt = datetime.strptime(
                    f"{start_date} {start_hour:02d}:{start_minute:02d}:00",
                    "%Y-%m-%d %H:%M:%S"
                )
                end_dt = datetime.strptime(
                    f"{end_date} {end_hour:02d}:{end_minute:02d}:59",
                    "%Y-%m-%d %H:%M:%S"
                )
               
                if end_dt <= start_dt:
                    return ["Error: La fecha final debe ser posterior a la inicial", 0] + date_values
               
                replay_system.current_replay = {
                    'start_time': start_dt,
                    'end_time': end_dt,
                    'current_time': start_dt,
                    'speed': speed if speed is not None else 1.0
                }
               
                return [f"Reproduciendo desde: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}", 0] + date_values
               
            elif trigger_id == 'replay-pause-button' and replay_system.current_replay:
                current_time = replay_system.current_replay['current_time']
                return [f"Pausado en: {current_time.strftime('%Y-%m-%d %H:%M:%S')}", 50] + date_values
               
            elif trigger_id == 'replay-stop-button':
                replay_system.current_replay = None
                return ["Reproducción detenida", 0] + date_values
               
        except Exception as e:
            print(f"Error en replay: {str(e)}")
            return [f"Error: {str(e)}", 0] + date_values
           
        return ["Esperando inicio...", 0] + date_values
