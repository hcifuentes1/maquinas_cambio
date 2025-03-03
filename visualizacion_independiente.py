import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import sqlite3
import os
import time

# Configuración de la base de datos
user_dir = os.path.expanduser('~')
db_path = os.path.join(user_dir, 'metro_data', 'historico_maquinas.db')

# Función para obtener datos recientes
def get_latest_data(maquina_id, limit=100):
    try:
        conn = sqlite3.connect(db_path)
        
        # Primero verificar si estamos usando la nueva estructura
        c = conn.cursor()
        c.execute("PRAGMA table_info(mediciones)")
        columns = [col[1] for col in c.fetchall()]
        
        if 'corriente_f1' in columns:
            # Nueva estructura
            query = """
                SELECT timestamp, corriente_f1, corriente_f2, corriente_f3, 
                voltaje_ctrl_izq, voltaje_ctrl_der, posicion, ciclo_progreso
                FROM mediciones
                WHERE maquina_id = ?
                ORDER BY timestamp DESC LIMIT ?
            """
        else:
            # Estructura antigua
            query = """
                SELECT timestamp, voltaje, corriente, posicion, ciclo_progreso
                FROM mediciones
                WHERE maquina_id = ?
                ORDER BY timestamp DESC LIMIT ?
            """
        
        df = pd.read_sql_query(query, conn, params=(maquina_id, limit))
        conn.close()
        
        # Convertir datos si es necesario
        if 'corriente_f1' not in df.columns and not df.empty:
            df['corriente_f1'] = df['corriente']
            df['corriente_f2'] = df['corriente']
            df['corriente_f3'] = df['corriente']
            df['voltaje_ctrl_izq'] = df['voltaje'] / 10
            df['voltaje_ctrl_der'] = df['voltaje'] / 10
        
        # Convertir timestamps a datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ordenar por timestamp ascendente para la visualización
        df = df.sort_values('timestamp')
        
        return df
    except Exception as e:
        print(f"Error al obtener datos: {e}")
        return pd.DataFrame()

# Generar datos de demostración si no hay datos reales
def generate_demo_data(num_points=100):
    now = datetime.now()
    data = {
        'timestamp': [now - timedelta(seconds=i) for i in range(num_points-1, -1, -1)],
        'corriente_f1': [0.5 + 3.5 * (i % 10 < 5) + 0.2 * (i % 3) for i in range(num_points)],
        'corriente_f2': [0.48 + 3.4 * (i % 10 < 5) + 0.19 * (i % 3) for i in range(num_points)],
        'corriente_f3': [0.52 + 3.6 * (i % 10 < 5) + 0.21 * (i % 3) for i in range(num_points)],
        'voltaje_ctrl_izq': [24.0 + 0.2 * ((i % 10) - 5) for i in range(num_points)],
        'voltaje_ctrl_der': [1.2 + 0.05 * ((i % 10) - 5) for i in range(num_points)],
        'posicion': ['Izquierda' if i % 20 < 10 else 'Derecha' for i in range(num_points)],
        'ciclo_progreso': [(i % 10) * 10 if i % 10 < 5 else 100 - (i % 10) * 10 for i in range(num_points)]
    }
    return pd.DataFrame(data)

# Funciones para crear gráficos
def create_corrientes_graph(df):
    fig = go.Figure()
    
    if len(df) > 1:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['corriente_f1'],
            line=dict(color='#FF9800'),
            name='Fase 1'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['corriente_f2'],
            line=dict(color='#FFA726'),
            name='Fase 2'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['corriente_f3'],
            line=dict(color='#FFB74D'),
            name='Fase 3'
        ))
    else:
        # Usar datos de demostración si no hay datos reales
        demo_df = generate_demo_data(50)
        
        fig.add_trace(go.Scatter(
            x=demo_df['timestamp'],
            y=demo_df['corriente_f1'],
            line=dict(color='#FF9800'),
            name='Fase 1'
        ))
        
        fig.add_trace(go.Scatter(
            x=demo_df['timestamp'],
            y=demo_df['corriente_f2'],
            line=dict(color='#FFA726'),
            name='Fase 2'
        ))
        
        fig.add_trace(go.Scatter(
            x=demo_df['timestamp'],
            y=demo_df['corriente_f3'],
            line=dict(color='#FFB74D'),
            name='Fase 3'
        ))
    
    fig.update_layout(
        title="Tendencia de Corrientes",
        yaxis_title="Corriente (A)",
        margin=dict(l=30, r=30, t=40, b=30),
        height=300,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.add_hline(y=5.5, line_dash="dot", line_color="red")
    
    return fig

def create_voltajes_graph(df):
    fig = go.Figure()
    
    if len(df) > 1:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['voltaje_ctrl_izq'],
            line=dict(color='#2196F3'),
            name='Ctrl Izquierdo'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['voltaje_ctrl_der'],
            line=dict(color='#64B5F6'),
            name='Ctrl Derecho'
        ))
    else:
        # Usar datos de demostración si no hay datos reales
        demo_df = generate_demo_data(50)
        
        fig.add_trace(go.Scatter(
            x=demo_df['timestamp'],
            y=demo_df['voltaje_ctrl_izq'],
            line=dict(color='#2196F3'),
            name='Ctrl Izquierdo'
        ))
        
        fig.add_trace(go.Scatter(
            x=demo_df['timestamp'],
            y=demo_df['voltaje_ctrl_der'],
            line=dict(color='#64B5F6'),
            name='Ctrl Derecho'
        ))
    
    fig.update_layout(
        title="Tendencia de Voltajes de Controladores",
        yaxis_title="Voltaje (VDC)",
        margin=dict(l=30, r=30, t=40, b=30),
        height=300,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    voltaje_nominal = 24.0        
    fig.add_hline(y=voltaje_nominal, line_color="green")
    fig.add_hline(y=voltaje_nominal * 1.15, line_dash="dot", line_color="red")
    
    return fig

# Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Visualización Independiente - Máquinas de Cambio"

app.layout = dbc.Container([
    html.H2("Visualización Independiente - Máquinas de Cambio", className="text-center my-4"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Seleccionar Máquina:"),
            dcc.Dropdown(
                id='maquina-dropdown',
                options=[
                    {'label': 'VIM-11/21', 'value': 'VIM-11/21'},
                    {'label': 'TSE-54B', 'value': 'TSE-54B'}
                ],
                value='VIM-11/21',
                className="mb-4"
            )
        ], width=4)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Tendencias de Corrientes"),
                dbc.CardBody([
                    dcc.Graph(id='corrientes-graph')
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Tendencias de Voltajes"),
                dbc.CardBody([
                    dcc.Graph(id='voltajes-graph')
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Datos Actuales"),
                dbc.CardBody(id='datos-actuales')
            ])
        ], width=12)
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=500,  # Actualizar cada 500 ms
        n_intervals=0
    )
], fluid=True)

# Callbacks
@app.callback(
    [Output('corrientes-graph', 'figure'),
     Output('voltajes-graph', 'figure'),
     Output('datos-actuales', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('maquina-dropdown', 'value')]
)
def update_graphs(n_intervals, maquina_id):
    # Obtener datos recientes
    df = get_latest_data(maquina_id)
    
    if df.empty:
        print(f"No hay datos para {maquina_id}, usando datos de demostración")
        df = generate_demo_data(50)
    
    # Crear gráficos
    fig_corrientes = create_corrientes_graph(df)
    fig_voltajes = create_voltajes_graph(df)
    
    # Obtener últimos valores
    if not df.empty:
        last_row = df.iloc[-1]
        datos_actuales = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H5("Corrientes de Fase (A)"),
                    dbc.Row([
                        dbc.Col([
                            html.Strong("F1:"),
                            html.Span(f" {last_row.get('corriente_f1', 0):.2f}A")
                        ], width=4),
                        dbc.Col([
                            html.Strong("F2:"),
                            html.Span(f" {last_row.get('corriente_f2', 0):.2f}A")
                        ], width=4),
                        dbc.Col([
                            html.Strong("F3:"),
                            html.Span(f" {last_row.get('corriente_f3', 0):.2f}A")
                        ], width=4)
                    ])
                ], width=6),
                dbc.Col([
                    html.H5("Voltajes de Controladores (VDC)"),
                    dbc.Row([
                        dbc.Col([
                            html.Strong("Izquierdo:"),
                            html.Span(f" {last_row.get('voltaje_ctrl_izq', 0):.2f}V")
                        ], width=6),
                        dbc.Col([
                            html.Strong("Derecho:"),
                            html.Span(f" {last_row.get('voltaje_ctrl_der', 0):.2f}V")
                        ], width=6)
                    ])
                ], width=6)
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Strong("Posición:"),
                    html.Span(f" {last_row.get('posicion', 'Desconocida')}")
                ], width=6),
                dbc.Col([
                    html.Strong("Progreso:"),
                    html.Span(f" {last_row.get('ciclo_progreso', 0):.1f}%")
                ], width=6)
            ]),
            html.Hr(),
            html.Div(f"Última actualización: {last_row.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}", 
                    className="text-muted text-end small")
        ])
    else:
        datos_actuales = html.P("No hay datos disponibles")
    
    return fig_corrientes, fig_voltajes, datos_actuales

if __name__ == '__main__':
    print(f"Iniciando visualización independiente...")
    print(f"Usando base de datos: {db_path}")
    
    # Verificar si podemos acceder a la base de datos
    if os.path.exists(db_path):
        print(f"Base de datos encontrada, verificando contenido...")
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM mediciones")
            count = c.fetchone()[0]
            print(f"La base de datos contiene {count} registros")
            
            # Verificar los últimos datos
            c.execute("SELECT * FROM mediciones ORDER BY timestamp DESC LIMIT 1")
            last_record = c.fetchone()
            if last_record:
                print(f"Último registro: {last_record}")
            conn.close()
        except Exception as e:
            print(f"Error accediendo a la base de datos: {e}")
    else:
        print(f"¡ALERTA! Base de datos no encontrada en {db_path}")
        
    app.run_server(debug=True, port=8051)