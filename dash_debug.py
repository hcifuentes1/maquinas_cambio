import plotly.graph_objs as go
import pandas as pd
import time
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import random

# Función para generar datos de simulación directamente
def generate_demo_data(num_points=100):
    now = datetime.now()
    timestamps = [now - timedelta(seconds=i) for i in range(num_points-1, -1, -1)]
    
    # Generar datos de corriente con patrones realistas
    corriente_f1 = []
    corriente_f2 = []
    corriente_f3 = []
    voltaje_izq = []
    voltaje_der = []
    
    for i in range(num_points):
        # Alternar entre ciclos de actividad y reposo
        cycle_position = i % 12
        
        if cycle_position < 6:  # En ciclo activo
            base_current = 4.0 + random.uniform(-0.2, 0.3)
            voltaje_i = 24.0 + random.uniform(-0.5, 0.5)
            voltaje_d = 1.2 + random.uniform(-0.1, 0.1)
        else:  # En reposo
            base_current = 0.5 + random.uniform(-0.1, 0.1)
            voltaje_i = 24.0 + random.uniform(-0.3, 0.3)
            voltaje_d = 1.2 + random.uniform(-0.05, 0.05)
            
        # Añadir variación entre fases
        corriente_f1.append(base_current * (1.0 + random.uniform(-0.05, 0.05)))
        corriente_f2.append(base_current * (0.95 + random.uniform(-0.05, 0.05)))
        corriente_f3.append(base_current * (1.05 + random.uniform(-0.05, 0.05)))
        voltaje_izq.append(voltaje_i)
        voltaje_der.append(voltaje_d)
    
    return {
        'timestamp': timestamps,
        'corriente_f1': corriente_f1,
        'corriente_f2': corriente_f2,
        'corriente_f3': corriente_f3,
        'voltaje_ctrl_izq': voltaje_izq,
        'voltaje_ctrl_der': voltaje_der
    }

# Crear gráficos independientes para corrientes
def create_corrientes_graph_demo():
    data = generate_demo_data()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['corriente_f1'],
        line=dict(color='#FF9800'),
        name='Fase 1'
    ))
    
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['corriente_f2'],
        line=dict(color='#FFA726'),
        name='Fase 2'
    ))
    
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['corriente_f3'],
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

# Crear gráficos independientes para voltajes
def create_voltajes_graph_demo():
    data = generate_demo_data()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['voltaje_ctrl_izq'],
        line=dict(color='#2196F3'),
        name='Ctrl Izquierdo'
    ))
    
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['voltaje_ctrl_der'],
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

# Función para integrar en hmi_dashboard.py
def fix_graphs_demo():
    # Generar valores de ejemplo
    data = generate_demo_data(50)
    corriente_f1 = data['corriente_f1'][-1]
    corriente_f2 = data['corriente_f2'][-1]
    corriente_f3 = data['corriente_f3'][-1]
    voltaje_izq = data['voltaje_ctrl_izq'][-1]
    voltaje_der = data['voltaje_ctrl_der'][-1]
    
    print(f"Datos generados: F1={corriente_f1:.2f}A, F2={corriente_f2:.2f}A, F3={corriente_f3:.2f}A")
    print(f"Voltajes: Izq={voltaje_izq:.2f}V, Der={voltaje_der:.2f}V")
    
    return {
        'timestamp': data['timestamp'],
        'corriente_f1': data['corriente_f1'],
        'corriente_f2': data['corriente_f2'],
        'corriente_f3': data['corriente_f3'],
        'voltaje_ctrl_izq': data['voltaje_ctrl_izq'],
        'voltaje_ctrl_der': data['voltaje_ctrl_der'],
    }