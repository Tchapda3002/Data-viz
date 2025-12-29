"""
PAGE 1 - EXPLORATION DES DONNÉES
Interface moderne pour l'analyse exploratoire
"""

import dash
from dash import html, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

dash.register_page(__name__, path='/', name='Exploration')

# Chargement des données
df = pd.read_excel('/Users/Apple/Desktop/Projets/Projet_Data_viz/data/microfinance_credit_risk.xlsx')

# Préparation des listes
regions = ['Toutes'] + sorted(df['region'].unique().tolist())
secteurs = ['Tous'] + sorted(df['secteur_activite'].unique().tolist())
canaux = ['Tous'] + sorted(df['canal_octroi'].unique().tolist())
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Fonctions utilitaires
def calculate_default_rate(data, group_by=None):
    if group_by is None:
        return (data['defaut_90j'].sum() / len(data)) * 100
    else:
        return data.groupby(group_by)['defaut_90j'].apply(
            lambda x: (x.sum() / len(x)) * 100
        ).reset_index(name='taux_defaut')

def filter_data(data, region=None, secteur=None, canal=None, montant_range=None, dsti_range=None):
    filtered = data.copy()
    if region and region != 'Toutes':
        filtered = filtered[filtered['region'] == region]
    if secteur and secteur != 'Tous':
        filtered = filtered[filtered['secteur_activite'] == secteur]
    if canal and canal != 'Tous':
        filtered = filtered[filtered['canal_octroi'] == canal]
    if montant_range:
        filtered = filtered[
            (filtered['montant_pret_xof'] >= montant_range[0]) &
            (filtered['montant_pret_xof'] <= montant_range[1])
        ]
    if dsti_range:
        filtered = filtered[
            (filtered['dsti_pct'] >= dsti_range[0]) &
            (filtered['dsti_pct'] <= dsti_range[1])
        ]
    return filtered

# Layout
layout = html.Div([
    
    # Section Filtres
    html.Div([
        html.Div([
            html.H3("Filtres d'Analyse", className="section-title"),
            html.P("Segmentez vos données selon différents critères", className="section-subtitle"),
        ], className="section-header"),
        
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Région", className="filter-label"),
                    dcc.Dropdown(
                        id='region-filter',
                        options=[{'label': r, 'value': r} for r in regions],
                        value='Toutes',
                        clearable=False,
                        className="modern-dropdown"
                    )
                ], md=3),
                
                dbc.Col([
                    html.Label("Secteur d'Activité", className="filter-label"),
                    dcc.Dropdown(
                        id='secteur-filter',
                        options=[{'label': s, 'value': s} for s in secteurs],
                        value='Tous',
                        clearable=False,
                        className="modern-dropdown"
                    )
                ], md=3),
                
                dbc.Col([
                    html.Label("Canal d'Octroi", className="filter-label"),
                    dcc.Dropdown(
                        id='canal-filter',
                        options=[{'label': c, 'value': c} for c in canaux],
                        value='Tous',
                        clearable=False,
                        className="modern-dropdown"
                    )
                ], md=3),
                
                dbc.Col([
                    html.Label("Type de Filtre", className="filter-label"),
                    dcc.RadioItems(
                        id='slider-type',
                        options=[
                            {'label': 'Montant', 'value': 'montant'},
                            {'label': 'DSTI', 'value': 'dsti'}
                        ],
                        value='montant',
                        inline=True,
                        className="modern-radio"
                    )
                ], md=3)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Div(id='slider-container')
                ])
            ])
        ], className="filters-grid")
    ], className="section-card"),
    
    # Section KPIs
    html.Div([
        html.Div([
            html.H3("Indicateurs Clés", className="section-title"),
            html.P("Vue d'ensemble des taux de défaut", className="section-subtitle"),
        ], className="section-header"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Taux Global", className="kpi-label"),
                    html.Div(id='kpi-global', className="kpi-value"),
                    html.Div(className="kpi-bar kpi-bar-danger")
                ], className="kpi-card")
            ], md=3),
            
            dbc.Col([
                html.Div([
                    html.Div("Par Région", className="kpi-label"),
                    html.Div(id='kpi-region', className="kpi-value"),
                    html.Div(className="kpi-bar kpi-bar-primary")
                ], className="kpi-card")
            ], md=3),
            
            dbc.Col([
                html.Div([
                    html.Div("Par Secteur", className="kpi-label"),
                    html.Div(id='kpi-secteur', className="kpi-value"),
                    html.Div(className="kpi-bar kpi-bar-success")
                ], className="kpi-card")
            ], md=3),
            
            dbc.Col([
                html.Div([
                    html.Div("Par Canal", className="kpi-label"),
                    html.Div(id='kpi-canal', className="kpi-value"),
                    html.Div(className="kpi-bar kpi-bar-warning")
                ], className="kpi-card")
            ], md=3)
        ])
    ], className="section-card"),
    
    # Section Visualisations
    html.Div([
        html.Div([
            html.H3("Visualisations Interactives", className="section-title"),
            html.P("Explorez les relations et distributions", className="section-subtitle"),
        ], className="section-header"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("Distribution DSTI", className="chart-title"),
                        dcc.RadioItems(
                            id='dsti-chart-type',
                            options=[
                                {'label': 'Histogramme', 'value': 'hist'},
                                {'label': 'Boxplot', 'value': 'box'}
                            ],
                            value='hist',
                            inline=True,
                            className="chart-toggle"
                        )
                    ], className="chart-header"),
                    dcc.Graph(id='dsti-graph', config={'displayModeBar': False})
                ], className="chart-card")
            ], md=6),
            
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("Analyse Bivariée", className="chart-title"),
                        html.Div([
                            dcc.Dropdown(
                                id='scatter-x',
                                options=[{'label': col, 'value': col} for col in numeric_cols if col != 'defaut_90j'],
                                value='revenu_mensuel_xof',
                                className="axis-selector",
                                placeholder="Axe X"
                            ),
                            dcc.Dropdown(
                                id='scatter-y',
                                options=[{'label': col, 'value': col} for col in numeric_cols if col != 'defaut_90j'],
                                value='montant_pret_xof',
                                className="axis-selector",
                                placeholder="Axe Y"
                            )
                        ], className="axis-selectors")
                    ], className="chart-header"),
                    dcc.Graph(id='scatter-graph', config={'displayModeBar': False})
                ], className="chart-card")
            ], md=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Matrice de Corrélation", className="chart-title"),
                    dcc.Graph(id='correlation-graph', config={'displayModeBar': False})
                ], className="chart-card")
            ])
        ])
    ], className="section-card"),
    
    # Section Table
    html.Div([
        html.Div([
            html.H3("Données Détaillées", className="section-title"),
            html.P("Vue tabulaire des données filtrées", className="section-subtitle"),
        ], className="section-header"),
        
        html.Div(id='datatable-container')
    ], className="section-card")
    
], className="page-container")

# Callbacks
@callback(
    Output('slider-container', 'children'),
    Input('slider-type', 'value')
)
def update_slider(slider_type):
    if slider_type == 'montant':
        min_val = df['montant_pret_xof'].min()
        max_val = df['montant_pret_xof'].max()
        return html.Div([
            html.Label(f"Montant du Prêt: {min_val:,.0f} - {max_val:,.0f} XOF", className="slider-label"),
            dcc.RangeSlider(
                id='range-slider',
                min=min_val,
                max=max_val,
                value=[min_val, max_val],
                marks={int(min_val): f'{min_val/1e6:.0f}M', int(max_val): f'{max_val/1e6:.0f}M'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="modern-slider"
            )
        ])
    else:
        min_val = df['dsti_pct'].min()
        max_val = df['dsti_pct'].max()
        return html.Div([
            html.Label(f"Taux DSTI: {min_val:.1f}% - {max_val:.1f}%", className="slider-label"),
            dcc.RangeSlider(
                id='range-slider',
                min=min_val,
                max=max_val,
                value=[min_val, max_val],
                marks={int(min_val): f'{min_val:.0f}%', int(max_val): f'{max_val:.0f}%'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="modern-slider"
            )
        ])

@callback(
    [Output('kpi-global', 'children'),
     Output('kpi-region', 'children'),
     Output('kpi-secteur', 'children'),
     Output('kpi-canal', 'children'),
     Output('dsti-graph', 'figure'),
     Output('scatter-graph', 'figure'),
     Output('correlation-graph', 'figure'),
     Output('datatable-container', 'children')],
    [Input('region-filter', 'value'),
     Input('secteur-filter', 'value'),
     Input('canal-filter', 'value'),
     Input('range-slider', 'value'),
     Input('slider-type', 'value'),
     Input('dsti-chart-type', 'value'),
     Input('scatter-x', 'value'),
     Input('scatter-y', 'value')]
)
def update_all(region, secteur, canal, slider_range, slider_type, chart_type, scatter_x, scatter_y):
    # Protection contre les valeurs None au premier chargement
    if slider_range is None:
        if slider_type == 'montant':
            slider_range = [df['montant_pret_xof'].min(), df['montant_pret_xof'].max()]
        else:
            slider_range = [df['dsti_pct'].min(), df['dsti_pct'].max()]
    
    # Filtrage
    if slider_type == 'montant':
        filtered_df = filter_data(df, region, secteur, canal, montant_range=slider_range)
    else:
        filtered_df = filter_data(df, region, secteur, canal, dsti_range=slider_range)
    
    # KPIs
    kpi_global = f"{calculate_default_rate(filtered_df):.2f}%"
    
    if region != 'Toutes':
        kpi_region = f"{calculate_default_rate(filtered_df):.2f}%"
    else:
        region_stats = calculate_default_rate(filtered_df, 'region')
        max_region = region_stats.loc[region_stats['taux_defaut'].idxmax()]
        kpi_region = f"{max_region['taux_defaut']:.2f}% ({max_region['region']})"
    
    if secteur != 'Tous':
        kpi_secteur = f"{calculate_default_rate(filtered_df):.2f}%"
    else:
        secteur_stats = calculate_default_rate(filtered_df, 'secteur_activite')
        max_secteur = secteur_stats.loc[secteur_stats['taux_defaut'].idxmax()]
        kpi_secteur = f"{max_secteur['taux_defaut']:.2f}% ({max_secteur['secteur_activite']})"
    
    if canal != 'Tous':
        kpi_canal = f"{calculate_default_rate(filtered_df):.2f}%"
    else:
        canal_stats = calculate_default_rate(filtered_df, 'canal_octroi')
        max_canal = canal_stats.loc[canal_stats['taux_defaut'].idxmax()]
        kpi_canal = f"{max_canal['taux_defaut']:.2f}% ({max_canal['canal_octroi']})"
    
    # Graphiques avec style moderne
    colors = {'0': '#10b981', '1': '#ef4444'}
    
    if chart_type == 'hist':
        fig_dsti = px.histogram(
            filtered_df, x='dsti_pct', color='defaut_90j',
            nbins=30, color_discrete_map={0: '#10b981', 1: '#ef4444'},
            labels={'dsti_pct': 'DSTI (%)', 'defaut_90j': 'Statut'}
        )
    else:
        fig_dsti = px.box(
            filtered_df, x='defaut_90j', y='dsti_pct',
            color='defaut_90j', color_discrete_map={0: '#10b981', 1: '#ef4444'},
            labels={'dsti_pct': 'DSTI (%)', 'defaut_90j': 'Statut'}
        )
    
    fig_dsti.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif', 'color': '#1e293b'},
        margin=dict(t=20, b=40, l=40, r=20),
        showlegend=False
    )
    
    fig_scatter = px.scatter(
        filtered_df, x=scatter_x, y=scatter_y, color='defaut_90j',
        color_discrete_map={0: '#10b981', 1: '#ef4444'},
        labels={'defaut_90j': 'Statut'}, opacity=0.6
    )
    fig_scatter.update_traces(marker=dict(size=10))
    fig_scatter.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif', 'color': '#1e293b'},
        margin=dict(t=20, b=40, l=40, r=20),
        showlegend=False
    )
    
    corr_cols = [col for col in numeric_cols if col in filtered_df.columns]
    corr_matrix = filtered_df[corr_cols].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 9, "color": "#1e293b"},
        colorbar=dict(title="Corrélation", titlefont={"size": 12})
    ))
    
    fig_corr.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif', 'color': '#1e293b'},
        margin=dict(t=20, b=60, l=60, r=20),
        height=500
    )
    
    # Table moderne
    table = dash_table.DataTable(
        data=filtered_df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in filtered_df.columns],
        page_size=15,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '14px',
            'fontFamily': 'Inter, sans-serif',
            'fontSize': '13px',
            'color': '#334155',
            'border': 'none'
        },
        style_header={
            'fontWeight': '600',
            'backgroundColor': '#f8fafc',
            'color': '#1e293b',
            'borderBottom': '2px solid #e2e8f0',
            'textTransform': 'uppercase',
            'fontSize': '11px',
            'letterSpacing': '0.5px'
        },
        style_data={
            'borderBottom': '1px solid #f1f5f9'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8fafc'
            },
            {
                'if': {'filter_query': '{defaut_90j} = 1'},
                'backgroundColor': '#fef2f2',
                'color': '#dc2626'
            }
        ],
        filter_action='native',
        sort_action='native'
    )
    
    return kpi_global, kpi_region, kpi_secteur, kpi_canal, fig_dsti, fig_scatter, fig_corr, table