# maintenance_dashboard.py
import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import json
import os
from dash import html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go


class MaintenanceDashboard:
    def __init__(self, db_path, config_path='maintenance_config.json'):
        """
        Inicializa el dashboard de mantenimiento predictivo
        
        Args:
            db_path (str): Ruta a la base de datos SQLite
            config_path (str): Ruta al archivo de configuración
        """
        self.db_path = db_path
        self.config = self._load_config(config_path)
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='maintenance_dashboard.log'
        )
        self.logger = logging.getLogger('MaintenanceDashboard')
    
    def _load_config(self, config_path):
        """
        Carga la configuración de notificaciones y umbrales
        
        Returns:
            dict: Configuración de mantenimiento
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                # Configuración por defecto
                return {
                    "email_notifications": {
                        "enabled": False,
                        "smtp_server": "",
                        "smtp_port": 587,
                        "sender_email": "",
                        "sender_password": "",
                        "recipients": []
                    },
                    "alert_thresholds": {
                        "critical_health_score": 40,
                        "warning_health_score": 70,
                        "maintenance_interval_days": 30
                    }
                }
        except Exception as e:
            self.logger.error(f"Error cargando configuración: {e}")
            return {}

    def get_machine_maintenance_history(self, machine_id=None):
        """
        Obtiene el historial de mantenimiento de máquinas
        
        Args:
            machine_id (str, optional): ID de máquina específica
        
        Returns:
            pd.DataFrame: Historial de mantenimiento
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Consulta para obtener historial de mantenimiento
            query = """
                SELECT 
                    maquina_id,
                    timestamp,
                    health_status,
                    health_score,
                    recommendations,
                    maintenance_due
                FROM maintenance_logs
                {where_clause}
                ORDER BY timestamp DESC
            """
            
            # Preparar cláusula WHERE si se especifica máquina
            where_clause = "WHERE maquina_id = ?" if machine_id else ""
            
            # Ejecutar consulta
            if machine_id:
                df = pd.read_sql_query(query.format(where_clause=where_clause), 
                                       conn, 
                                       params=(machine_id,))
            else:
                df = pd.read_sql_query(query.format(where_clause=where_clause), 
                                       conn)
            
            conn.close()
            return df
        except Exception as e:
            self.logger.error(f"Error obteniendo historial de mantenimiento: {e}")
            return pd.DataFrame()

    def update_machine_health_summary(self, n_intervals=None):
        """Actualiza el resumen de estado de máquinas"""
        try:
            # Obtener datos recientes de cada máquina
            summary = []
            for machine_id in ['VIM-11/21', 'TSE-54B']:  # Ajustar según tus máquinas
                # Obtener último estado de salud
                health_data = self.get_machine_maintenance_history(machine_id)
                if not health_data.empty:
                    latest = health_data.iloc[0]
                    summary.append(
                        html.Div([
                            html.H5(f"Máquina {machine_id}"),
                            html.P(f"Estado: {latest['health_status'].upper()}"),
                            html.P(f"Puntuación: {latest['health_score']:.2f}%"),
                            html.P(f"Próximo Mantenimiento: {latest['maintenance_due']}")
                        ])
                    )
            
            return summary
        except Exception as e:
            self.logger.error(f"Error actualizando resumen de salud: {e}")
            return [html.P("Error cargando datos")]

    def update_health_status_chart(self, n):
        """Crea un gráfico de estado de salud"""
        try:
            # Obtener historial de mantenimiento
            history = self.get_machine_maintenance_history()
            
            # Crear gráfico de líneas
            fig = go.Figure()
            
            for machine in history['maquina_id'].unique():
                machine_data = history[history['maquina_id'] == machine]
                
                fig.add_trace(go.Scatter(
                    x=machine_data['timestamp'],
                    y=machine_data['health_score'],
                    mode='lines+markers',
                    name=machine,
                    line=dict(
                        width=3,
                        color='green' if machine == 'VIM-11/21' else 'blue'
                    )
                ))
            
            fig.update_layout(
                title='Puntuación de Salud por Máquina',
                xaxis_title='Fecha',
                yaxis_title='Puntuación de Salud (%)',
                height=400
            )
            
            return fig
        except Exception as e:
            self.logger.error(f"Error creando gráfico de estado: {e}")
            return go.Figure()

    def update_maintenance_schedule(self, n):
        """Genera tabla de próximo mantenimiento"""
        try:
            # Obtener historial de mantenimiento
            history = self.get_machine_maintenance_history()
            
            # Preparar datos para la tabla
            schedule_data = []
            for machine in history['maquina_id'].unique():
                machine_data = history[history['maquina_id'] == machine].iloc[0]
                schedule_data.append({
                    'machine': machine,
                    'maintenance_due': machine_data['maintenance_due'],
                    'status': machine_data['health_status']
                })
            
            return schedule_data
        except Exception as e:
            self.logger.error(f"Error actualizando tabla de mantenimiento: {e}")
            return []
        
    def create_maintenance_dashboard_layout(self):
        """
        Crea el layout para el dashboard de mantenimiento
        
        Returns:
            dash.html: Layout del dashboard de mantenimiento
        """
        return dbc.Container([
            html.H1("Panel de Mantenimiento Predictivo", className="text-center my-4"),
            
            # Fila de métricas principales
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Resumen de Estado de Máquinas"),
                        dbc.CardBody(id='machine-health-summary')
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Fila de gráficos y tablas
            dbc.Row([
                # Gráfico de estado de salud
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Estado de Salud por Máquina"),
                        dbc.CardBody(dcc.Graph(id='health-status-chart'))
                    ])
                ], width=6),
                
                # Tabla de mantenimiento
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Próximo Mantenimiento"),
                        dbc.CardBody(
                            dash_table.DataTable(
                                id='maintenance-schedule-table',
                                columns=[
                                    {'name': 'Máquina', 'id': 'machine'},
                                    {'name': 'Próximo Mantenimiento', 'id': 'maintenance_due'},
                                    {'name': 'Estado', 'id': 'status'}
                                ],
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '10px'
                                }
                            )
                        )
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Fila de historial y configuración
            dbc.Row([
                # Historial de mantenimiento
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Historial de Mantenimiento"),
                        dbc.CardBody(
                            dash_table.DataTable(
                                id='maintenance-history-table',
                                columns=[
                                    {'name': 'Máquina', 'id': 'maquina_id'},
                                    {'name': 'Fecha', 'id': 'timestamp'},
                                    {'name': 'Estado', 'id': 'health_status'},
                                    {'name': 'Puntuación', 'id': 'health_score'}
                                ],
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '10px'
                                }
                            )
                        )
                    ])
                ], width=12)
            ])
        ], fluid=True)

    def update_maintenance_history(self, n):
        """Actualiza la tabla de historial de mantenimiento"""
        try:
            # Obtener historial completo de mantenimiento
            history = self.get_machine_maintenance_history()
            
            # Convertir a formato para tabla
            return history.to_dict('records')
        except Exception as e:
            self.logger.error(f"Error actualizando historial de mantenimiento: {e}")
            return []

    def register_dashboard_callbacks(self, app):
        """
        Registra los callbacks para el dashboard de mantenimiento
        
        Args:
            app (dash.Dash): Aplicación Dash
        """
        @app.callback(
            Output('machine-health-summary', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def machine_health_summary_callback(n):
            return self.update_machine_health_summary(n)
        
        @app.callback(
            Output('health-status-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def health_status_chart_callback(n):
            return self.update_health_status_chart(n)
        
        @app.callback(
            Output('maintenance-schedule-table', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def maintenance_schedule_callback(n):
            return self.update_maintenance_schedule(n)
        
        @app.callback(
            Output('maintenance-history-table', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def maintenance_history_callback(n):
            return self.update_maintenance_history(n)

    def log_maintenance_event(self, machine_id, health_status, health_score, recommendations, maintenance_due):
        """
        Registra un evento de mantenimiento en la base de datos
        
        Args:
            machine_id (str): ID de la máquina
            health_status (str): Estado de salud
            health_score (float): Puntuación de salud
            recommendations (list): Recomendaciones de mantenimiento
            maintenance_due (str): Próximo mantenimiento
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Crear tabla de logs si no existe
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS maintenance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    maquina_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    health_status TEXT,
                    health_score REAL,
                    recommendations TEXT,
                    maintenance_due TEXT
                )
            ''')
            
            # Convertir recomendaciones a cadena JSON
            recommendations_json = json.dumps(recommendations)
            
            # Insertar evento de mantenimiento
            cursor.execute('''
                INSERT INTO maintenance_logs 
                (maquina_id, health_status, health_score, recommendations, maintenance_due)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                machine_id, 
                health_status, 
                health_score, 
                recommendations_json, 
                maintenance_due
            ))
            
            conn.commit()
            conn.close()
            
            # Verificar umbrales para envío de alertas
            alert_thresholds = self.config.get('alert_thresholds', {})
            critical_threshold = alert_thresholds.get('critical_health_score', 40)
            warning_threshold = alert_thresholds.get('warning_health_score', 70)
            
            # Enviar alerta si es crítico o advertencia
            if health_score <= critical_threshold:
                self.send_maintenance_alert(
                    machine_id, 
                    'critical', 
                    health_score, 
                    recommendations
                )
            elif health_score <= warning_threshold:
                self.send_maintenance_alert(
                    machine_id, 
                    'warning', 
                    health_score, 
                    recommendations
                )
            
            self.logger.info(f"Evento de mantenimiento registrado para {machine_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error registrando evento de mantenimiento: {e}")
            return False
    
    def export_maintenance_report(self, start_date=None, end_date=None, format='csv'):
        """
        Exporta un informe de mantenimiento
        
        Args:
            start_date (datetime, optional): Fecha de inicio
            end_date (datetime, optional): Fecha de fin
            format (str, optional): Formato de exportación (csv, xlsx, pdf)
        
        Returns:
            str: Ruta del archivo exportado
        """
        try:
            # Generar informe
            report_data = self.generate_maintenance_report(start_date, end_date)
            
            # Crear directorio de informes si no existe
            reports_dir = 'maintenance_reports'
            os.makedirs(reports_dir, exist_ok=True)
            
            # Nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{reports_dir}/maintenance_report_{timestamp}"
            
            if format == 'csv':
                # Exportar a CSV
                filename = f"{base_filename}.csv"
                pd.DataFrame(report_data['machines']).to_csv(filename, index=False)
                self.logger.info(f"Informe exportado: {filename}")
                return filename
            
            elif format == 'xlsx':
                # Exportar a Excel
                filename = f"{base_filename}.xlsx"
                with pd.ExcelWriter(filename) as writer:
                    # Hoja de resumen
                    pd.DataFrame([report_data['summary']]).to_excel(writer, sheet_name='Resumen', index=False)
                    # Hoja de detalles de máquinas
                    pd.DataFrame(report_data['machines']).to_excel(writer, sheet_name='Máquinas', index=False)
                self.logger.info(f"Informe exportado: {filename}")
                return filename
            
            elif format == 'pdf':
                # Exportar a PDF (requiere instalación de reportlab)
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
                from reportlab.lib import colors
                from reportlab.lib.styles import getSampleStyleSheet
                
                filename = f"{base_filename}.pdf"
                doc = SimpleDocTemplate(filename, pagesize=letter)
                elements = []
                
                # Estilos
                styles = getSampleStyleSheet()
                
                # Título
                title = Paragraph("Informe de Mantenimiento Predictivo", styles['Title'])
                elements.append(title)
                
                # Resumen
                summary_data = [
                    ['Total Máquinas', str(report_data['summary']['total_machines'])],
                    ['Total Chequeos', str(report_data['summary']['total_checks'])],
                    ['Alertas Críticas', str(report_data['summary']['total_critical_alerts'])],
                    ['Alertas de Advertencia', str(report_data['summary']['total_warning_alerts'])]
                ]
                summary_table = Table(summary_data)
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 12),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                elements.append(summary_table)
                
                # Detalles de máquinas
                machine_data = [['Máquina', 'Total Chequeos', 'Críticos', 'Advertencias', 'Puntuación Promedio']]
                for machine in report_data['machines']:
                    machine_data.append([
                        machine['maquina_id'],
                        str(machine['total_checks']),
                        str(machine['critical_count']),
                        str(machine['warning_count']),
                        f"{machine['avg_health_score']:.2f}"
                    ])
                
                machine_table = Table(machine_data)
                machine_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 12),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                elements.append(machine_table)
                
                # Generar PDF
                doc.build(elements)
                
                self.logger.info(f"Informe exportado: {filename}")
                return filename
            
            else:
                raise ValueError("Formato no soportado. Use 'csv', 'xlsx' o 'pdf'.")
        
        except Exception as e:
            self.logger.error(f"Error exportando informe de mantenimiento: {e}")
            return None
        


# Las funciones de integración también permanecen igual
def integrar_dashboard_mantenimiento(app, ml_monitor):
    """
    Integra el dashboard de mantenimiento en la aplicación principal
    
    Args:
        app (dash.Dash): Aplicación Dash principal
        ml_monitor (MonitoringML): Instancia del monitor de ML
    
    Returns:
        MaintenanceDashboard: Instancia del dashboard de mantenimiento
    """
    # Crear instancia del dashboard de mantenimiento
    maintenance_dashboard = MaintenanceDashboard(ml_monitor.db_path)
    
    # Agregar pestaña de mantenimiento al layout existente
    def crear_pestaña_mantenimiento():
        return html.Div([
            html.H3("Mantenimiento Predictivo", className="text-center my-3"),
            maintenance_dashboard.create_maintenance_dashboard_layout()
        ])
    
    # Modificar layout existente para incluir nueva pestaña
    def modificar_layout(original_layout):
        # Buscar el primer elemento de Tabs
        for i, child in enumerate(original_layout.children):
            if isinstance(child, dbc.Tabs):
                # Agregar nueva pestaña de mantenimiento
                child.children.append(
                    dbc.Tab(
                        crear_pestaña_mantenimiento(), 
                        label="Mantenimiento", 
                        tab_id="tab-mantenimiento"
                    )
                )
                break
        return original_layout
    
    # Registrar callbacks específicos del dashboard de mantenimiento
    maintenance_dashboard.register_dashboard_callbacks(app)
    
    return maintenance_dashboard

def setup_predictive_maintenance(app, ml_monitor, predictive_system):
    """
    Configura el sistema de mantenimiento predictivo
    
    Args:
        app (dash.Dash): Aplicación Dash principal
        ml_monitor (MonitoringML): Monitor de ML
        predictive_system (PredictiveMaintenanceSystem): Sistema de mantenimiento predictivo
    
    Returns:
        MaintenanceDashboard: Instancia del dashboard de mantenimiento
    """
    # Crear dashboard de mantenimiento
    maintenance_dashboard = MaintenanceDashboard(ml_monitor.db_path)
    
    # Método para registrar eventos de mantenimiento
    def registrar_estado_maquina(maquina_id, estado):
        """
        Registra el estado de una máquina en el sistema de mantenimiento
        
        Args:
            maquina_id (str): ID de la máquina
            estado (dict): Estado de la máquina
        """
        try:
            health_status = estado.get('health_status', {})
            
            # Registrar evento de mantenimiento
            maintenance_dashboard.log_maintenance_event(
                machine_id=maquina_id,
                health_status=health_status.get('status', 'unknown'),
                health_score=health_status.get('health_score', 0),
                recommendations=health_status.get('recommendations', []),
                maintenance_due=health_status.get('maintenance_due', 'Unknown')
            )
        except Exception as e:
            print(f"Error registrando estado de máquina: {e}")
    
    return maintenance_dashboard, registrar_estado_maquina