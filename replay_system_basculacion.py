# replay_system_basculacion.py

from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta, time
import pandas as pd
import sqlite3
import numpy as np
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ReplaySystem')

class ReplaySystemBasculacion:
    """Sistema de replay basado en eventos de basculación para máquinas de cambio"""
    
    def __init__(self, db_path):
        """
        Inicializa el sistema de replay
        
        Args:
            db_path: Ruta a la base de datos SQLite
        """
        self.db_path = db_path
        self.current_replay = None
        # Definir horarios límite para filtrado
        self.hora_inicio = time(5, 0)  # 05:00
        self.hora_fin = time(23, 30)   # 23:30
        
    def get_available_dates(self):
        """
        Obtiene las fechas disponibles con eventos de basculación en la base de datos
        
        Returns:
            list: Lista de fechas en formato YYYY-MM-DD
        """
        try:
            conn = sqlite3.connect(self.db_path)
            # Consulta para encontrar días con eventos de basculación entre 05:00 y 23:30
            query = """
                SELECT DISTINCT date(timestamp) as fecha
                FROM mediciones
                WHERE time(timestamp) BETWEEN time('05:00:00') AND time('23:30:00')
                ORDER BY fecha DESC
            """
            dates = pd.read_sql_query(query, conn)
            conn.close()
            return dates['fecha'].tolist()
        except Exception as e:
            logger.error(f"Error obteniendo fechas: {e}")
            return []
    
    def find_basculacion_events(self, fecha, maquina_id=None):
        """
        Encuentra eventos de basculación (cambios de posición) para una fecha específica
        
        Args:
            fecha: Fecha en formato YYYY-MM-DD
            maquina_id: ID de la máquina (opcional, None para todas)
            
        Returns:
            DataFrame: Eventos de basculación con timestamps de inicio
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Base de la consulta para detectar cambios de posición
            # Un evento de basculación ocurre cuando la posición cambia respecto al registro anterior
            base_query = f"""
                WITH posiciones_ordenadas AS (
                    SELECT 
                        timestamp, 
                        maquina_id, 
                        posicion,
                        LAG(posicion) OVER (PARTITION BY maquina_id ORDER BY timestamp) as posicion_anterior
                    FROM mediciones
                    WHERE date(timestamp) = '{fecha}'
                    AND time(timestamp) BETWEEN time('05:00:00') AND time('23:30:00')
                )
                SELECT 
                    timestamp as inicio_evento, 
                    maquina_id,
                    posicion as nueva_posicion,
                    posicion_anterior
                FROM posiciones_ordenadas
                WHERE posicion <> posicion_anterior OR posicion_anterior IS NULL
            """
            
            # Filtrar por máquina si se especifica
            if maquina_id:
                query = base_query + f" AND maquina_id = '{maquina_id}'"
            else:
                query = base_query
                
            query += " ORDER BY timestamp ASC"
            
            # Ejecutar consulta
            eventos = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convertir tipos de datos
            if not eventos.empty:
                eventos['inicio_evento'] = pd.to_datetime(eventos['inicio_evento'])
            
            return eventos
            
        except Exception as e:
            logger.error(f"Error buscando eventos de basculación: {e}")
            return pd.DataFrame()
    
    def get_basculacion_data(self, basculacion_id, maquina_id):
        """
        Obtiene los datos completos de un evento de basculación
        
        Args:
            basculacion_id: Timestamp de inicio del evento de basculación
            maquina_id: ID de la máquina
            
        Returns:
            DataFrame: Datos del evento completo (varios registros secuenciales)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Primero obtener el timestamp de inicio exacto
            inicio_dt = pd.to_datetime(basculacion_id)
            
            # Calcular tiempo estimado para incluir todo el evento (duración típica + margen)
            # Asumimos que una basculación típica toma menos de 30 segundos
            fin_dt = inicio_dt + timedelta(seconds=30)
            
            # Consultar todos los datos durante este periodo
            query = """
                SELECT *
                FROM mediciones
                WHERE maquina_id = ?
                  AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(
                    maquina_id, 
                    inicio_dt.strftime('%Y-%m-%d %H:%M:%S'), 
                    fin_dt.strftime('%Y-%m-%d %H:%M:%S')
                )
            )
            
            conn.close()
            
            # Convertir tipos de datos
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de basculación: {e}")
            return pd.DataFrame()
    
    def get_all_basculacion_data(self, fecha, maquina_id):
        """
        Obtiene todos los datos de basculación para una fecha y máquina
        
        Args:
            fecha: Fecha en formato YYYY-MM-DD
            maquina_id: ID de la máquina
            
        Returns:
            DataFrame: Datos concatenados de todos los eventos de basculación
        """
        try:
            # Encontrar todos los eventos de basculación para esta fecha/máquina
            eventos = self.find_basculacion_events(fecha, maquina_id)
            
            if eventos.empty:
                logger.info(f"No se encontraron eventos de basculación para {fecha}, máquina {maquina_id}")
                return pd.DataFrame()
            
            # Crear dataframe para almacenar todos los datos
            all_data = []
            
            # Para cada evento, obtener sus datos completos
            for idx, evento in eventos.iterrows():
                datos_evento = self.get_basculacion_data(
                    evento['inicio_evento'],
                    evento['maquina_id']
                )
                
                if not datos_evento.empty:
                    # Añadir identificador de evento para referencia
                    datos_evento['evento_id'] = idx
                    all_data.append(datos_evento)
            
            # Concatenar todos los eventos
            if all_data:
                return pd.concat(all_data, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error obteniendo datos completos de basculación: {e}")
            return pd.DataFrame()
    
    def get_data_range(self, start_datetime, end_datetime, maquina_id=None):
        """
        Obtiene los datos para un rango de tiempo específico (mantiene compatibilidad con sistema original)
        
        Args:
            start_datetime: Inicio del rango
            end_datetime: Fin del rango
            maquina_id: ID de máquina específica (opcional)
            
        Returns:
            DataFrame: Datos en el rango especificado
        """
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
            logger.error(f"Error obteniendo datos: {e}")
            return pd.DataFrame()


def create_replay_controls_basculacion():
    """
    Crea los controles para el sistema de replay basado en basculación
    """
    controls = dbc.Card([
        dbc.CardHeader(html.H4("Control de Replay de Basculación", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                # Selector de fecha 
                dbc.Col([
                    html.Label("Fecha de Operación"),
                    dcc.DatePickerSingle(
                        id='replay-date',
                        display_format='DD/MM/YYYY',
                        className="mb-2"
                    ),
                ], md=3),
                
                # Selector de máquina
                dbc.Col([
                    html.Label("Máquina"),
                    dcc.Dropdown(
                        id='replay-maquina',
                        options=[],  # Se llenará dinámicamente
                        className="mb-2"
                    ),
                ], md=3),
                
                # Selector de evento de basculación
                dbc.Col([
                    html.Label("Evento de Basculación"),
                    dcc.Dropdown(
                        id='replay-basculacion',
                        options=[],  # Se llenará dinámicamente
                        className="mb-2"
                    ),
                ], md=4),
                
                # Control de velocidad
                dbc.Col([
                    html.Label("Velocidad"),
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
                ], md=2)
            ]),
            
            # Controles de reproducción
            dbc.Row([
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
                ], md=6),
                
                # Estadísticas del día
                dbc.Col([
                    html.Div([
                        html.H6("Resumen del día:", className="mb-2"),
                        html.Div(id="replay-day-stats", className="small")
                    ])
                ], md=6)
            ], className="mt-3"),
            
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
    
    # Agregar el componente de intervalo para el replay
    controls.children.append(
        dcc.Interval(
            id='replay-interval-component',
            interval=1000,  # en milisegundos
            n_intervals=0
        )
    )
    
    return controls


# Funciones auxiliares para los callbacks

def machine_callback_wrapper(selected_date, replay_system):
    """Obtiene opciones de máquinas para una fecha específica"""
    if not selected_date:
        return []
    
    try:
        conn = sqlite3.connect(replay_system.db_path)
        query = f"""
            SELECT DISTINCT maquina_id
            FROM mediciones
            WHERE date(timestamp) = '{selected_date}'
            AND time(timestamp) BETWEEN time('05:00:00') AND time('23:30:00')
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        options = [{'label': mid, 'value': mid} for mid in df['maquina_id']]
        return options
    except Exception as e:
        logger.error(f"Error obteniendo máquinas: {e}")
        return []


def basculacion_callback_wrapper(selected_date, selected_machine, replay_system):
    """Obtiene opciones de eventos de basculación"""
    if not selected_date or not selected_machine:
        return [], "Seleccione fecha y máquina para ver estadísticas"
    
    try:
        # Obtener eventos de basculación
        eventos = replay_system.find_basculacion_events(selected_date, selected_machine)
        
        if eventos.empty:
            return [], "No se encontraron eventos de basculación en este día"
        
        # Crear opciones para dropdown
        options = []
        for idx, row in eventos.iterrows():
            timestamp = row['inicio_evento']
            from_pos = row['posicion_anterior'] if not pd.isna(row['posicion_anterior']) else "Inicio"
            to_pos = row['nueva_posicion']
            
            # Formatear hora para mostrar
            hora_str = timestamp.strftime('%H:%M:%S')
            label = f"{hora_str}: {from_pos} → {to_pos}"
            
            options.append({
                'label': label,
                'value': timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Generar estadísticas del día
        total_events = len(eventos)
        first_event = eventos['inicio_evento'].min().strftime('%H:%M:%S')
        last_event = eventos['inicio_evento'].max().strftime('%H:%M:%S')
        
        stats_html = [
            html.Div(f"Total eventos: {total_events}"),
            html.Div(f"Primer cambio: {first_event}"),
            html.Div(f"Último cambio: {last_event}")
        ]
        
        return options, stats_html
    except Exception as e:
        logger.error(f"Error actualizando opciones de basculación: {e}")
        return [], f"Error: {str(e)}"
    

def replay_status_callback_wrapper(n_play, n_pause, n_stop, n_intervals,
                                   selected_date, selected_machine, selected_basculacion, 
                                   speed, replay_system):
    """
    Wrapper para el callback de estado del replay.
    
    Args:
        n_play: Clicks del botón reproducir
        n_pause: Clicks del botón pausar
        n_stop: Clicks del botón detener
        n_intervals: Número de intervalos
        selected_date: Fecha seleccionada
        selected_machine: ID de máquina seleccionada
        selected_basculacion: Evento de basculación seleccionado
        speed: Valor de velocidad de reproducción
        replay_system: Sistema de replay
    """
    ctx = callback_context
    available_dates = replay_system.get_available_dates()
   
    # Valores por defecto para las fechas
    if available_dates:
        min_date = datetime.strptime(min(available_dates), '%Y-%m-%d').date()
        max_date = datetime.strptime(max(available_dates), '%Y-%m-%d').date()
        date_values = [min_date, max_date, max_date]
    else:
        date_values = [None, None, None]
   
    # Si no hay trigger, retorna valores iniciales
    if not ctx.triggered:
        return ["Esperando selección...", 0] + date_values
   
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"Callback desencadenado por: {trigger_id}")
    print(f"Valores: n_play={n_play}, n_pause={n_pause}, n_stop={n_stop}, n_intervals={n_intervals}")
   
    try:
        if trigger_id == 'replay-play-button' and n_play and selected_date and selected_machine and selected_basculacion:
            # Verificación robusta de los datos de entrada
            print(f"Botón Reproducir presionado ({n_play} veces)")
            print(f"Iniciar reproducción: Fecha={selected_date}, Máquina={selected_machine}, Evento={selected_basculacion}")
            
            # Detener cualquier reproducción anterior
            if replay_system.current_replay and 'machine_id' in replay_system.current_replay:
                old_machine = replay_system.current_replay.get('machine_id')
                if old_machine:
                    try:
                        import sys
                        if 'hmi_dashboard' in sys.modules:
                            hmi_dashboard = sys.modules['hmi_dashboard']
                            with hmi_dashboard.data_lock:
                                if old_machine in hmi_dashboard.estado_maquinas:
                                    hmi_dashboard.estado_maquinas[old_machine]["modo_operacion"] = "normal"
                    except Exception as e:
                        print(f"Error al resetear modo de operación anterior: {e}")
            
            # Obtener datos del evento de basculación seleccionado
            basculacion_timestamp = pd.to_datetime(selected_basculacion)
           
            # Obtener los datos de este evento específico
            datos_evento = replay_system.get_basculacion_data(
                basculacion_timestamp,
                selected_machine
            )
           
            if datos_evento.empty:
                print(f"Error: No se encontraron datos para el evento {selected_basculacion} de la máquina {selected_machine}")
                return ["Error: No se encontraron datos para este evento", 0] + date_values
           
            # Configurar replay con estructura mejorada
            replay_system.current_replay = {
                'data': datos_evento,
                'current_index': 0,
                'speed': speed if speed is not None else 1.0,
                'paused': False,
                'total_frames': len(datos_evento),
                'current_time': basculacion_timestamp,
                'start_time': basculacion_timestamp,
                'end_time': basculacion_timestamp + timedelta(seconds=30),
                'machine_id': selected_machine  # Guardar la máquina seleccionada
            }
           
            # Información de depuración
            print(f"INICIANDO REPRODUCCIÓN: {selected_machine}")
            print(f"Datos encontrados: {len(datos_evento)} registros")
            if not datos_evento.empty:
                print(f"Primer registro: {datos_evento.iloc[0]}")
                print(f"Último registro: {datos_evento.iloc[-1]}")
           
            # Actualizar directamente el estado de la máquina seleccionada
            try:
                import sys
                if 'hmi_dashboard' in sys.modules:
                    hmi_dashboard = sys.modules['hmi_dashboard']
                    with hmi_dashboard.data_lock:
                        if selected_machine in hmi_dashboard.estado_maquinas:
                            estado = hmi_dashboard.estado_maquinas[selected_machine]
                            estado["modo_operacion"] = "replay"
                           
                            if len(datos_evento) > 0:
                                row = datos_evento.iloc[0]
                                if len(estado["timestamp"]) > 0:
                                    estado["timestamp"][-1] = row['timestamp']
                                    estado["voltaje"][-1] = row['voltaje']
                                    estado["corriente"][-1] = row['corriente']
                                else:
                                    estado["timestamp"].append(row['timestamp'])
                                    estado["voltaje"].append(row['voltaje'])
                                    estado["corriente"].append(row['corriente'])
                               
                                estado["posicion"] = row['posicion']
                                estado["ciclo_progreso"] = row['ciclo_progreso']
                               
                                print(f"Estado inicial forzado: Posición={estado['posicion']}, Progreso={estado['ciclo_progreso']}")
            except Exception as e:
                import traceback
                print(f"No se pudo forzar estado inicial: {str(e)}")
                print(traceback.format_exc())
           
            return [f"Reproduciendo evento de {basculacion_timestamp.strftime('%H:%M:%S')}", 0] + date_values
       
        elif trigger_id == 'replay-pause-button' and n_pause and replay_system.current_replay:
            print(f"Botón Pausar presionado ({n_pause} veces)")
            # Verificación robusta de la estructura
            if 'paused' not in replay_system.current_replay or 'current_index' not in replay_system.current_replay or 'total_frames' not in replay_system.current_replay:
                print("Error: replay_system.current_replay no tiene la estructura esperada para pausar")
                return ["Error: estructura de datos inválida", 0] + date_values
                
            # IMPORTANTE: Alternar el estado de pausa (toggle)
            replay_system.current_replay['paused'] = not replay_system.current_replay.get('paused', False)
            estado = "Pausado" if replay_system.current_replay['paused'] else "Reproduciendo"
            
            print(f"Estado de pausa cambiado a: {replay_system.current_replay['paused']}")
           
            current_idx = replay_system.current_replay['current_index']
            progress = (current_idx / replay_system.current_replay['total_frames']) * 100
           
            return [f"{estado} en posición {current_idx+1}/{replay_system.current_replay['total_frames']}", progress] + date_values
       
        elif trigger_id == 'replay-stop-button' and n_stop:
            print(f"Botón Detener presionado ({n_stop} veces)")
            # Guardar información antes de limpiar
            if replay_system.current_replay and 'machine_id' in replay_system.current_replay:
                machine_id = replay_system.current_replay['machine_id']
                print(f"Deteniendo reproducción para {machine_id}")
                try:
                    import sys
                    if 'hmi_dashboard' in sys.modules:
                        hmi_dashboard = sys.modules['hmi_dashboard']
                        with hmi_dashboard.data_lock:
                            if machine_id in hmi_dashboard.estado_maquinas:
                                # IMPORTANTE: Resetear el estado explícitamente
                                estado = hmi_dashboard.estado_maquinas[machine_id]
                                estado["modo_operacion"] = "normal"
                                print(f"Estado de máquina {machine_id} restablecido a 'normal'")
                except Exception as e:
                    print(f"Error al resetear modo de operación: {e}")
            
            # IMPORTANTE: Limpiar explícitamente el objeto de reproducción
            replay_system.current_replay = None
            print("Objeto de reproducción limpiado")
            return ["Reproducción detenida", 0] + date_values
       
        elif trigger_id == 'replay-interval-component' and replay_system.current_replay:
            # Verificación robusta de la estructura de datos
            if 'data' not in replay_system.current_replay or 'current_index' not in replay_system.current_replay:
                print("Error: replay_system.current_replay no tiene la estructura esperada")
                return ["Error en estructura de datos", 0] + date_values
                
            # Si está pausado, no hacer nada
            if replay_system.current_replay.get('paused', False):
                current_idx = replay_system.current_replay['current_index']
                total_frames = replay_system.current_replay.get('total_frames', 1)  # Evitar división por cero
                progress = (current_idx / total_frames) * 100
                return [f"Pausado en posición {current_idx+1}/{total_frames}", progress] + date_values
           
            # Avanzar reproducción según velocidad
            speed = replay_system.current_replay.get('speed', 1.0)
            frames_to_advance = max(1, int(speed))
           
            old_idx = replay_system.current_replay['current_index']
            replay_system.current_replay['current_index'] += frames_to_advance
            current_idx = replay_system.current_replay['current_index']
           
            print(f"Avanzando reproducción: {old_idx} -> {current_idx} (de {replay_system.current_replay['total_frames']})")
           
            # Verificar si llegamos al final
            if current_idx >= replay_system.current_replay['total_frames']:
                machine_id = replay_system.current_replay.get('machine_id')
                replay_system.current_replay = None
                # Resetear modo de operación
                if machine_id:
                    try:
                        import sys
                        if 'hmi_dashboard' in sys.modules:
                            hmi_dashboard = sys.modules['hmi_dashboard']
                            with hmi_dashboard.data_lock:
                                if machine_id in hmi_dashboard.estado_maquinas:
                                    hmi_dashboard.estado_maquinas[machine_id]["modo_operacion"] = "normal"
                                    print(f"Reproducción completada, máquina {machine_id} restablecida a modo normal")
                    except Exception as e:
                        print(f"Error al resetear modo de operación al finalizar: {e}")
                return ["Reproducción completada", 100] + date_values
           
            # Calcular progreso
            progress = (current_idx / replay_system.current_replay['total_frames']) * 100
           
            # Verificar que el índice es válido
            if current_idx >= len(replay_system.current_replay['data']):
                print(f"Error: índice fuera de rango: {current_idx} >= {len(replay_system.current_replay['data'])}")
                return [f"Error: índice fuera de rango", progress] + date_values
                
            # Mostrar información del frame actual
            current_time = replay_system.current_replay['data'].iloc[current_idx]['timestamp']
            time_str = current_time.strftime('%H:%M:%S.%f')[:-3]
           
            # Actualizar directamente el estado de la máquina seleccionada
            machine_id = replay_system.current_replay.get('machine_id')
            if not machine_id:
                print("Error: No hay machine_id en replay_system.current_replay")
                return [f"Error: Sin ID de máquina", progress] + date_values
                
            try:
                import sys
                if 'hmi_dashboard' in sys.modules:
                    hmi_dashboard = sys.modules['hmi_dashboard']
                    with hmi_dashboard.data_lock:
                        if machine_id in hmi_dashboard.estado_maquinas:
                            estado = hmi_dashboard.estado_maquinas[machine_id]
                            row = replay_system.current_replay['data'].iloc[current_idx]
                           
                            # Actualizar estado directamente
                            if len(estado["timestamp"]) > 0:
                                estado["timestamp"][-1] = row['timestamp']
                                estado["voltaje"][-1] = row['voltaje']
                                estado["corriente"][-1] = row['corriente']
                            else:
                                estado["timestamp"].append(row['timestamp'])
                                estado["voltaje"].append(row['voltaje'])
                                estado["corriente"].append(row['corriente'])
                           
                            estado["posicion"] = row['posicion']
                            estado["ciclo_progreso"] = row['ciclo_progreso']
                            estado["modo_operacion"] = "replay"
                           
                            print(f"FORZADA actualización directa para {machine_id}: Pos={estado['posicion']}, Prog={estado['ciclo_progreso']}")
            except Exception as e:
                import traceback
                print(f"Error actualizando estado directo: {str(e)}")
                print(traceback.format_exc())
           
            return [f"Reproduciendo: {time_str} ({current_idx+1}/{replay_system.current_replay['total_frames']})", progress] + date_values
           
    except Exception as e:
        import traceback
        print(f"Error en reproducción: {str(e)}")
        print(traceback.format_exc())
        return [f"Error: {str(e)}", 0] + date_values
   
    return ["Esperando selección...", 0] + date_values


def register_replay_callbacks_basculacion(app, replay_system):
    """
    Registra los callbacks para el sistema de replay basado en basculación
    
    Args:
        app: Aplicación Dash
        replay_system: Instancia de ReplaySystemBasculacion
    """
    
    # Callback para máquinas
    app.callback(
        Output('replay-maquina', 'options'),
        Input('replay-date', 'date')
    )(lambda selected_date: machine_callback_wrapper(selected_date, replay_system))
    
    # Callback para eventos de basculación
    app.callback(
        [Output('replay-basculacion', 'options'),
         Output('replay-day-stats', 'children')],
        [Input('replay-date', 'date'),
         Input('replay-maquina', 'value')]
    )(lambda selected_date, selected_machine: 
      basculacion_callback_wrapper(selected_date, selected_machine, replay_system))
    
    # Callback para control de replay
    app.callback(
        [Output('replay-time-display', 'children'),
         Output('replay-progress', 'value'),
         Output('replay-date', 'min_date_allowed'),
         Output('replay-date', 'max_date_allowed'),
         Output('replay-date', 'initial_visible_month')],
        [Input('replay-play-button', 'n_clicks'),
         Input('replay-pause-button', 'n_clicks'),
         Input('replay-stop-button', 'n_clicks'),
         Input('replay-interval-component', 'n_intervals')],
        [State('replay-date', 'date'),
         State('replay-maquina', 'value'),
         State('replay-basculacion', 'value'),
         State('replay-speed', 'value')]
    )(lambda n_play, n_pause, n_stop, n_intervals,
            selected_date, selected_machine, selected_basculacion, speed:
      replay_status_callback_wrapper(
          n_play, n_pause, n_stop, n_intervals,
          selected_date, selected_machine, selected_basculacion, 
          speed, replay_system))