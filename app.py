"""
MICROFINANCE RISK ANALYZER
Application moderne d'analyse du risque crédit
"""

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc



app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "/assets/app_styles.css"  # ← Ajoute cette ligne
    ],
    suppress_callback_exceptions=True,
    use_pages=True,
    pages_folder="pages"
)

server = app.server 

app.title = "Microfinance Risk Analyzer"

app.layout = dbc.Container([
    # Header moderne
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H1("Projet Data Visualisation", className="app-title"),
                    html.P("Système d'analyse et prédiction du risque crédit microfinance", 
                           className="app-subtitle"),
                ], className="header-content"),
            ], className="app-header")
        ])
    ]),
    
    # Navigation moderne
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Nav([
                    dbc.NavLink([
                        html.Span("Exploration des Données", className="nav-text"),
                        html.Span(className="nav-indicator")
                    ], href="/", active="exact", className="modern-nav-link"),
                    
                    dbc.NavLink([
                        html.Span("Modélisation Prédictive", className="nav-text"),
                        html.Span(className="nav-indicator")
                    ], href="/modelisation", active="exact", className="modern-nav-link"),
                ], className="modern-nav")
            ], className="nav-container")
        ])
    ]),
    
    # Contenu des pages
    html.Div([
        dash.page_container
    ], className="page-content"),
    
    # Footer minimaliste
    html.Footer([
        html.Div([
            html.Span("Wilfred Rod TCHAPDA KOUADJO", className="footer-text"),
            html.Span(" · ", className="footer-separator"),
            html.Span("ENSAE Dakar", className="footer-text"),
            html.Span(" · ", className="footer-separator"),
            html.Span("2025-2026", className="footer-text"),
        ], className="footer-content")
    ], className="app-footer")
    
], fluid=True, className="app-container")

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)