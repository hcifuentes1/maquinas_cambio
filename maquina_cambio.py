# maquina_cambio.py
from dash import html
import plotly.graph_objs as go
import math

def crear_svg_maquina(posicion, progreso):
    """
    Crea un SVG de la máquina de cambio con movimiento continuo
   
    Args:
        posicion (str): 'Izquierda' o 'Derecha'
        progreso (float): Valor entre 0 y 100
    """
    # Asegurar que el progreso sea un número
    progreso = float(progreso) if progreso is not None else 0
   
    # Configuración de la aguja
    longitud_aguja = 100
    x_inicio = 175
    y_centro = 100
    angulo_max = 25
   
    # Determinar ángulos inicial y final basados en la dirección del movimiento
    if posicion == "Derecha":
        angulo_inicial = angulo_max  # Comienza desde la posición izquierda
        angulo_final = -angulo_max   # Termina en la posición derecha
    else:
        angulo_inicial = -angulo_max  # Comienza desde la posición derecha
        angulo_final = angulo_max     # Termina en la posición izquierda
   
    # Calcular el ángulo actual basado en el progreso
    if progreso == 0:
        # En espera: mantener la última posición
        angulo = angulo_inicial
        estado_movimiento = "En espera"
    elif progreso >= 100:
        # Posición final
        angulo = angulo_final
        estado_movimiento = posicion
    else:
        # En movimiento: interpolar entre la posición inicial y final
        angulo = angulo_inicial + (angulo_final - angulo_inicial) * (progreso / 100)
        estado_movimiento = "En movimiento"
   
    # Calcular punto final
    angulo_rad = math.radians(angulo)
    x_final = x_inicio + longitud_aguja * math.cos(angulo_rad)
    y_final = y_centro + longitud_aguja * math.sin(angulo_rad)
   
    # Color del texto de estado
    color_estado = "#FFA500" if estado_movimiento == "En movimiento" else \
                  "#4CAF50" if posicion == "Izquierda" else \
                  "#FF5722"
   
    # Texto de posición actual
    texto_posicion = "Sin control" if estado_movimiento == "En movimiento" else posicion
    color_posicion = "#FFA500" if estado_movimiento == "En movimiento" else \
                    "#4CAF50" if posicion == "Izquierda" else \
                    "#FF5722"
   
    svg_content = f'''
    <svg viewBox="0 0 400 200" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="400" height="200" fill="#1a1a1a"/>
       
        <!-- Rieles fijos -->
        <path d="M 100,50 L 350,50" stroke="#666" stroke-width="8" fill="none"/>
        <path d="M 100,150 L 350,150" stroke="#666" stroke-width="8" fill="none"/>
       
        <!-- Motor/mecanismo -->
        <rect x="155" y="85" width="40" height="30"
              fill="#444" stroke="#666"/>
       
        <!-- Aguja móvil -->
        <path d="M {x_inicio},{y_centro} L {x_final},{y_final}"
              stroke="#FFA500" stroke-width="10" fill="none"/>
       
        <!-- Punto de pivote -->
        <circle cx="{x_inicio}" cy="{y_centro}" r="6" fill="#FFA500"/>
       
        <!-- Punto final -->
        <circle cx="{x_final}" cy="{y_final}" r="6" fill="#FFA500"/>
       
        <!-- Indicadores de dirección -->
        <text x="100" y="30" fill="{('#4CAF50' if posicion == 'Izquierda' else '#666')}"
              style="font-size: 14px">Vía Principal</text>
        <text x="300" y="30" fill="{('#FF5722' if posicion == 'Derecha' else '#666')}"
              style="font-size: 14px">Desviador</text>
       
        <!-- Estado de movimiento -->
        <text x="200" y="190" fill="{color_estado}" text-anchor="middle"
              style="font-size: 16px; font-weight: bold">
            {estado_movimiento}
        </text>
       
        <!-- Posición actual -->
        <text x="200" y="140" fill="{color_posicion}" text-anchor="middle"
              style="font-size: 16px; font-weight: bold">
            Posición: {texto_posicion}
        </text>
       
        <!-- Indicador de progreso -->
        <text x="200" y="170" fill="white" text-anchor="middle"
              style="font-size: 14px">
            Progreso: {progreso:.1f}%
        </text>
    </svg>
    '''
   
    # Convertir el SVG a una data URI
    import base64
    svg_bytes = svg_content.encode()
    encoded = base64.b64encode(svg_bytes).decode()
    data_uri = f"data:image/svg+xml;base64,{encoded}"
   
    return html.Div([
        html.Img(
            src=data_uri,
            style={
                "width": "100%",
                "height": "100%"
            }
        )
    ], style={
        "width": "100%",
        "height": "250px",
        "backgroundColor": "#1a1a1a",
        "borderRadius": "8px",
        "padding": "16px"
    })

def crear_indicador_progreso(progreso):
    """Crea un indicador de progreso circular"""
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=progreso,
        number={
    'font': {'color': 'white'},
    'suffix': '%'
},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'visible': False},
            'bar': {'color': "#9C27B0"},
            'steps': [{'range': [0, 100], 'color': "#E1BEE7"}]
        }
    )).update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=120,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
