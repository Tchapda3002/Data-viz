"""
PAGE 1 - EXPLORATION DES DONNÉES (VERSION MINIMALISTE)
Interface compacte pour l'analyse exploratoire avec téléchargements
"""

import dash
from dash import html, dcc, callback, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import geopandas as gpd
import json
import base64
import io

dash.register_page(__name__, path='/', name='Exploration')

# Chargement des données
df = pd.read_excel('./data/microfinance_credit_risk.xlsx')

# Chargement du shapefile du Sénégal
gdf_senegal = gpd.read_file('SEN_adm/SEN_adm1.shp')

# Préparation des listes
regions = ['Toutes'] + sorted(df['region'].unique().tolist())
secteurs = ['Tous'] + sorted(df['secteur_activite'].unique().tolist())
canaux = ['Tous'] + sorted(df['canal_octroi'].unique().tolist())
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Fonctions utilitaires
def filter_data(data, region=None, secteur=None, canal=None, montant_range=None):
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
    return filtered

def format_number(value, is_currency=False, is_pct=False):
    if pd.isna(value):
        return "N/A"
    if is_pct:
        return f"{value:.2f}%"
    if is_currency:
        if value >= 1e9:
            return f"{value/1e9:.2f}B XOF"
        elif value >= 1e6:
            return f"{value/1e6:.2f}M XOF"
        elif value >= 1e3:
            return f"{value/1e3:.2f}K XOF"
        return f"{value:,.0f} XOF"
    return f"{value:,.2f}"

# Layout
layout = html.Div([
    
    # Filtres compacts
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Région", className="filter-label-mini"),
                dcc.Dropdown(
                    id='region-filter',
                    options=[{'label': r, 'value': r} for r in regions],
                    value='Toutes',
                    placeholder="Région",
                    className="compact-dropdown"
                )
            ], md=2),
            dbc.Col([
                html.Label("Secteur d'activité", className="filter-label-mini"),
                dcc.Dropdown(
                    id='secteur-filter',
                    options=[{'label': s, 'value': s} for s in secteurs],
                    value='Tous',
                    placeholder="Secteur",
                    className="compact-dropdown"
                )
            ], md=2),
            dbc.Col([
                html.Label("Canal d'octroi", className="filter-label-mini"),
                dcc.Dropdown(
                    id='canal-filter',
                    options=[{'label': c, 'value': c} for c in canaux],
                    value='Tous',
                    placeholder="Canal",
                    className="compact-dropdown"
                )
            ], md=2),
            dbc.Col([
                html.Div([
                    html.Label("Montant du prêt (XOF)", className="filter-label-mini"),
                    dcc.RangeSlider(
                        id='montant-slider',
                        min=df['montant_pret_xof'].min(),
                        max=df['montant_pret_xof'].max(),
                        value=[df['montant_pret_xof'].min(), df['montant_pret_xof'].max()],
                        marks={
                            int(df['montant_pret_xof'].min()): f'{df["montant_pret_xof"].min()/1e6:.0f}M',
                            int(df['montant_pret_xof'].max()): f'{df["montant_pret_xof"].max()/1e6:.0f}M'
                        },
                        tooltip={"placement": "bottom", "always_visible": False},
                        className="compact-slider"
                    )
                ])
            ], md=6)
        ], className="mb-3")
    ], className="filters-compact"),
    
    # KPIs Grid (8 indicateurs)
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("👥", className="kpi-icon"),
                    html.Div([
                        html.Div(id='kpi-clients', className="kpi-value-mini"),
                        html.Div("Clients actifs", className="kpi-label-mini")
                    ])
                ], className="kpi-mini")
            ], md=3, sm=6, xs=12),
            
            dbc.Col([
                html.Div([
                    html.Div("💰", className="kpi-icon"),
                    html.Div([
                        html.Div(id='kpi-revenu', className="kpi-value-mini"),
                        html.Div("Revenu moyen", className="kpi-label-mini")
                    ])
                ], className="kpi-mini")
            ], md=3, sm=6, xs=12),
            
            dbc.Col([
                html.Div([
                    html.Div("⏱️", className="kpi-icon"),
                    html.Div([
                        html.Div(id='kpi-retard', className="kpi-value-mini"),
                        html.Div("Jours retard moy.", className="kpi-label-mini")
                    ])
                ], className="kpi-mini")
            ], md=3, sm=6, xs=12),
            
            dbc.Col([
                html.Div([
                    html.Div("💳", className="kpi-icon"),
                    html.Div([
                        html.Div(id='kpi-montant', className="kpi-value-mini"),
                        html.Div("Montant prêt moy.", className="kpi-label-mini")
                    ])
                ], className="kpi-mini")
            ], md=3, sm=6, xs=12)
        ], className="mb-2"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("📊", className="kpi-icon"),
                    html.Div([
                        html.Div(id='kpi-taux-interet', className="kpi-value-mini"),
                        html.Div("Taux intérêt moy.", className="kpi-label-mini")
                    ])
                ], className="kpi-mini")
            ], md=3, sm=6, xs=12),
            
            dbc.Col([
                html.Div([
                    html.Div("⚠️", className="kpi-icon"),
                    html.Div([
                        html.Div(id='kpi-defaut', className="kpi-value-mini"),
                        html.Div("Taux défaut 90j", className="kpi-label-mini")
                    ])
                ], className="kpi-mini kpi-danger")
            ], md=3, sm=6, xs=12),
            
            dbc.Col([
                html.Div([
                    html.Div("🏦", className="kpi-icon"),
                    html.Div([
                        html.Div(id='kpi-epargne', className="kpi-value-mini"),
                        html.Div("Épargne moyenne", className="kpi-label-mini")
                    ])
                ], className="kpi-mini")
            ], md=3, sm=6, xs=12),
            
            dbc.Col([
                html.Div([
                    html.Div("📅", className="kpi-icon"),
                    html.Div([
                        html.Div(id='kpi-duree', className="kpi-value-mini"),
                        html.Div("Durée moy. prêt", className="kpi-label-mini")
                    ])
                ], className="kpi-mini")
            ], md=3, sm=6, xs=12)
        ])
    ], className="kpi-grid-mini mb-3"),
    
    # Répartitions géographique et sectorielle
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("🗺️ Répartition par Région", className="chart-title-mini"),
                        html.Button("📥", id="btn-download-region", className="btn-download-mini", n_clicks=0)
                    ], className="chart-header-mini"),
                    dcc.Graph(id='region-chart', config={'displayModeBar': False})
                ], className="chart-card-mini")
            ], md=6),
            
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("🏢 Répartition par Secteur", className="chart-title-mini"),
                        html.Button("📥", id="btn-download-secteur", className="btn-download-mini", n_clicks=0)
                    ], className="chart-header-mini"),
                    dcc.Graph(id='secteur-chart', config={'displayModeBar': False})
                ], className="chart-card-mini")
            ], md=6)
        ], className="mb-3")
    ]),
    
    # Graphiques d'analyse
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("Distribution DSTI", className="chart-title-mini"),
                        html.Div([
                            dcc.RadioItems(
                                id='dsti-chart-type',
                                options=[
                                    {'label': 'Hist', 'value': 'hist'},
                                    {'label': 'Box', 'value': 'box'}
                                ],
                                value='hist',
                                inline=True,
                                className="chart-toggle-mini"
                            ),
                            html.Button("📥", id="btn-download-dsti", className="btn-download-mini", n_clicks=0)
                        ], style={'display': 'flex', 'gap': '10px', 'alignItems': 'center'})
                    ], className="chart-header-mini"),
                    dcc.Graph(id='dsti-graph', config={'displayModeBar': False})
                ], className="chart-card-mini")
            ], md=6),
            
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("Analyse Bivariée", className="chart-title-mini"),
                        html.Div([
                            dcc.Dropdown(
                                id='scatter-x',
                                options=[{'label': col, 'value': col} for col in numeric_cols if col != 'defaut_90j'],
                                value='revenu_mensuel_xof',
                                className="axis-selector-mini",
                                placeholder="X"
                            ),
                            dcc.Dropdown(
                                id='scatter-y',
                                options=[{'label': col, 'value': col} for col in numeric_cols if col != 'defaut_90j'],
                                value='montant_pret_xof',
                                className="axis-selector-mini",
                                placeholder="Y"
                            ),
                            html.Button("📥", id="btn-download-scatter", className="btn-download-mini", n_clicks=0)
                        ], style={'display': 'flex', 'gap': '5px', 'alignItems': 'center'})
                    ], className="chart-header-mini"),
                    dcc.Graph(id='scatter-graph', config={'displayModeBar': False})
                ], className="chart-card-mini")
            ], md=6)
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("Matrice de Corrélation", className="chart-title-mini"),
                        html.Button("📥", id="btn-download-corr", className="btn-download-mini", n_clicks=0)
                    ], className="chart-header-mini"),
                    dcc.Graph(id='correlation-graph', config={'displayModeBar': False})
                ], className="chart-card-mini")
            ])
        ])
    ]),
    
    # Légende du tableau
    html.Div([
        html.Div("Légende du tableau:", className="legend-title"),
        html.Div([
            html.Div([
                html.Span([html.Span("", className="legend-dot legend-red"), "Défaut (90j) - ligne complète"]),
                html.Span([html.Span("", className="legend-dot legend-red-cell"), "Retard > moyenne ou DSTI > 40% ou Taux > moyenne"]),
            ], className="legend-row"),
            html.Div([
                html.Span([html.Span("", className="legend-dot legend-orange"), "DSTI 30-40% ou Revenu/Épargne < moyenne"]),
                html.Span([html.Span("", className="legend-dot legend-orange-cell"), "Âge < 25 ou > 60 ou Dépendants > 5"]),
            ], className="legend-row"),
            html.Div([
                html.Span([html.Span("", className="legend-dot legend-green"), "DSTI 20-30% ou Pas de retard"]),
                html.Span([html.Span("", className="legend-dot legend-green-dark"), "DSTI < 20% ou Revenu/Épargne > moyenne×1.5 ou Score MM > 70"]),
            ], className="legend-row")
        ], className="legend-items")
    ], className="table-legend mb-2"),
    
    # Table des données
    html.Div([
        html.Div([
            html.Span("📋 Données Détaillées", className="chart-title-mini"),
            html.Button("📥 Télécharger CSV", id="btn-download-table", className="btn-download-table", n_clicks=0)
        ], className="chart-header-mini mb-2"),
        html.Div(id='datatable-container')
    ], className="table-section"),
    
    # Downloads invisibles
    dcc.Download(id="download-table"),
    dcc.Download(id="download-region"),
    dcc.Download(id="download-secteur"),
    dcc.Download(id="download-dsti"),
    dcc.Download(id="download-scatter"),
    dcc.Download(id="download-corr")
    
], className="page-container-mini")

# Callbacks
@callback(
    [Output('kpi-clients', 'children'),
     Output('kpi-revenu', 'children'),
     Output('kpi-retard', 'children'),
     Output('kpi-montant', 'children'),
     Output('kpi-taux-interet', 'children'),
     Output('kpi-defaut', 'children'),
     Output('kpi-epargne', 'children'),
     Output('kpi-duree', 'children'),
     Output('region-chart', 'figure'),
     Output('secteur-chart', 'figure'),
     Output('dsti-graph', 'figure'),
     Output('scatter-graph', 'figure'),
     Output('correlation-graph', 'figure'),
     Output('datatable-container', 'children')],
    [Input('region-filter', 'value'),
     Input('secteur-filter', 'value'),
     Input('canal-filter', 'value'),
     Input('montant-slider', 'value'),
     Input('dsti-chart-type', 'value'),
     Input('scatter-x', 'value'),
     Input('scatter-y', 'value')]
)
def update_all(region, secteur, canal, montant_range, chart_type, scatter_x, scatter_y):
    # Filtrage
    filtered_df = filter_data(df, region, secteur, canal, montant_range=montant_range)
    
    # Calcul des KPIs
    kpi_clients = f"{len(filtered_df):,}"
    kpi_revenu = format_number(filtered_df['revenu_mensuel_xof'].mean(), is_currency=True)
    kpi_retard = f"{filtered_df['jours_retard_12m'].mean():.1f}j"
    kpi_montant = format_number(filtered_df['montant_pret_xof'].mean(), is_currency=True)
    kpi_taux_interet = format_number(filtered_df['taux_interet_annuel_pct'].mean(), is_pct=True)
    kpi_defaut = format_number((filtered_df['defaut_90j'].sum() / len(filtered_df)) * 100, is_pct=True)
    kpi_epargne = format_number(filtered_df['epargne_xof'].mean(), is_currency=True)
    kpi_duree = f"{filtered_df['duree_mois'].mean():.1f} mois"
    
    # Graphique région (carte choroplèthe du Sénégal)
    region_data = filtered_df.groupby('region').agg({
        'client_id': 'count',
        'defaut_90j': 'sum'
    }).reset_index()
    region_data.columns = ['region', 'count', 'defauts']
    region_data['taux_defaut'] = (region_data['defauts'] / region_data['count'] * 100).round(2)
    
    # Fusion avec le shapefile
    gdf_plot = gdf_senegal.merge(region_data, left_on='NAME_1', right_on='region', how='left')
    gdf_plot['count'] = gdf_plot['count'].fillna(0)
    gdf_plot['taux_defaut'] = gdf_plot['taux_defaut'].fillna(0)
    
    # Création de la carte choroplèthe
    fig_region = px.choropleth(
        gdf_plot,
        geojson=json.loads(gdf_plot.to_json()),
        locations=gdf_plot.index,
        color='count',
        hover_name='NAME_1',
        hover_data={'count': True, 'taux_defaut': ':.2f'},
        color_continuous_scale='Blues',
        labels={'count': 'Nombre de clients', 'taux_defaut': 'Taux défaut (%)'}
    )
    
    fig_region.update_geos(
        fitbounds="locations",
        visible=False
    )
    
    fig_region.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif', 'size': 11},
        margin=dict(t=10, b=10, l=10, r=10),
        height=280,
        coloraxis_colorbar=dict(
            title="Clients",
            titlefont={"size": 10},
            len=0.7
        )
    )
    
    # Graphique secteur
    secteur_data = filtered_df.groupby('secteur_activite').size().reset_index(name='count')
    fig_secteur = px.pie(
        secteur_data, values='count', names='secteur_activite',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4
    )
    fig_secteur.update_traces(textposition='inside', textinfo='percent+label')
    fig_secteur.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif', 'size': 10},
        margin=dict(t=10, b=10, l=10, r=10),
        height=280,
        showlegend=False
    )
    
    # Graphique DSTI
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
        font={'family': 'Inter, sans-serif', 'size': 11},
        margin=dict(t=10, b=30, l=30, r=10),
        height=280,
        showlegend=False
    )
    
    # Scatter plot
    fig_scatter = px.scatter(
        filtered_df, x=scatter_x, y=scatter_y, color='defaut_90j',
        color_discrete_map={0: '#10b981', 1: '#ef4444'},
        labels={'defaut_90j': 'Statut'}, opacity=0.6
    )
    fig_scatter.update_traces(marker=dict(size=8))
    fig_scatter.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif', 'size': 11},
        margin=dict(t=10, b=30, l=30, r=10),
        height=280,
        showlegend=False
    )
    
    # Matrice de corrélation
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
        textfont={"size": 8},
        colorbar=dict(title="Corr", titlefont={"size": 10}, len=0.7)
    ))
    
    fig_corr.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif', 'size': 9},
        margin=dict(t=10, b=50, l=50, r=10),
        height=400
    )
    
    # Calcul des seuils statistiques pour la coloration
    retard_moy = filtered_df['jours_retard_12m'].mean()
    dsti_moy = filtered_df['dsti_pct'].mean()
    taux_moy = filtered_df['taux_interet_annuel_pct'].mean()
    revenu_moy = filtered_df['revenu_mensuel_xof'].mean()
    epargne_moy = filtered_df['epargne_xof'].mean()
    
    # Table avec mise en forme conditionnelle
    table = dash_table.DataTable(
        data=filtered_df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in filtered_df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '8px 12px',
            'fontFamily': 'Inter, sans-serif',
            'fontSize': '12px',
            'color': '#334155',
            'border': 'none'
        },
        style_header={
            'fontWeight': '600',
            'backgroundColor': '#f8fafc',
            'color': '#1e293b',
            'borderBottom': '2px solid #e2e8f0',
            'fontSize': '11px',
            'padding': '10px 12px'
        },
        style_data={
            'borderBottom': '1px solid #f1f5f9'
        },
        style_data_conditional=[
            # Lignes alternées
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8fafc'
            },
            
            # === COLONNES INDIVIDUELLES - ROUGE (Mauvais) ===
            
            # Défaut 90j = 1 (toute la ligne en rouge)
            {
                'if': {'filter_query': '{defaut_90j} = 1'},
                'backgroundColor': '#fee2e2',
                'color': '#991b1b',
                'fontWeight': '500'
            },
            
            # Jours de retard > moyenne (rouge léger)
            {
                'if': {
                    'filter_query': f'{{jours_retard_12m}} > {retard_moy}',
                    'column_id': 'jours_retard_12m'
                },
                'backgroundColor': '#fecaca',
                'color': '#991b1b',
                'fontWeight': '600'
            },
            
            # Jours de retard très élevés (> 30 jours) (rouge foncé)
            {
                'if': {
                    'filter_query': '{jours_retard_12m} > 30',
                    'column_id': 'jours_retard_12m'
                },
                'backgroundColor': '#fca5a5',
                'color': '#7f1d1d',
                'fontWeight': '700'
            },
            
            # DSTI > 40% (rouge)
            {
                'if': {
                    'filter_query': '{dsti_pct} > 40',
                    'column_id': 'dsti_pct'
                },
                'backgroundColor': '#fecaca',
                'color': '#991b1b',
                'fontWeight': '600'
            },
            
            # DSTI > 50% (rouge foncé)
            {
                'if': {
                    'filter_query': '{dsti_pct} > 50',
                    'column_id': 'dsti_pct'
                },
                'backgroundColor': '#fca5a5',
                'color': '#7f1d1d',
                'fontWeight': '700'
            },
            
            # Taux d'intérêt > moyenne (rouge léger)
            {
                'if': {
                    'filter_query': f'{{taux_interet_annuel_pct}} > {taux_moy}',
                    'column_id': 'taux_interet_annuel_pct'
                },
                'backgroundColor': '#fecaca',
                'color': '#991b1b',
                'fontWeight': '600'
            },
            
            # === COLONNES INDIVIDUELLES - ORANGE (Attention) ===
            
            # DSTI entre 30% et 40% (orange)
            {
                'if': {
                    'filter_query': '{dsti_pct} > 30 && {dsti_pct} <= 40',
                    'column_id': 'dsti_pct'
                },
                'backgroundColor': '#fed7aa',
                'color': '#92400e',
                'fontWeight': '500'
            },
            
            # Revenu < moyenne (orange)
            {
                'if': {
                    'filter_query': f'{{revenu_mensuel_xof}} < {revenu_moy}',
                    'column_id': 'revenu_mensuel_xof'
                },
                'backgroundColor': '#fed7aa',
                'color': '#92400e'
            },
            
            # Épargne < moyenne (orange)
            {
                'if': {
                    'filter_query': f'{{epargne_xof}} < {epargne_moy}',
                    'column_id': 'epargne_xof'
                },
                'backgroundColor': '#fed7aa',
                'color': '#92400e'
            },
            
            # Âge < 25 ou > 60 (orange - risque démographique)
            {
                'if': {
                    'filter_query': '{age} < 25 || {age} > 60',
                    'column_id': 'age'
                },
                'backgroundColor': '#fed7aa',
                'color': '#92400e'
            },
            
            # Nombre de dépendants > 5 (orange)
            {
                'if': {
                    'filter_query': '{nb_dependants} > 5',
                    'column_id': 'nb_dependants'
                },
                'backgroundColor': '#fed7aa',
                'color': '#92400e'
            },
            
            # === COLONNES INDIVIDUELLES - VERT (Bon) ===
            
            # DSTI < 20% (vert foncé - excellent)
            {
                'if': {
                    'filter_query': '{dsti_pct} < 20',
                    'column_id': 'dsti_pct'
                },
                'backgroundColor': '#86efac',
                'color': '#14532d',
                'fontWeight': '600'
            },
            
            # DSTI entre 20% et 30% (vert clair - bon)
            {
                'if': {
                    'filter_query': '{dsti_pct} >= 20 && {dsti_pct} < 30',
                    'column_id': 'dsti_pct'
                },
                'backgroundColor': '#bbf7d0',
                'color': '#166534'
            },
            
            # Pas de retard (vert)
            {
                'if': {
                    'filter_query': '{jours_retard_12m} = 0',
                    'column_id': 'jours_retard_12m'
                },
                'backgroundColor': '#bbf7d0',
                'color': '#166534'
            },
            
            # Revenu > moyenne * 1.5 (vert - bon revenu)
            {
                'if': {
                    'filter_query': f'{{revenu_mensuel_xof}} > {revenu_moy * 1.5}',
                    'column_id': 'revenu_mensuel_xof'
                },
                'backgroundColor': '#bbf7d0',
                'color': '#166534',
                'fontWeight': '500'
            },
            
            # Épargne > moyenne * 1.5 (vert)
            {
                'if': {
                    'filter_query': f'{{epargne_xof}} > {epargne_moy * 1.5}',
                    'column_id': 'epargne_xof'
                },
                'backgroundColor': '#bbf7d0',
                'color': '#166534',
                'fontWeight': '500'
            },
            
            # Ancienneté > 24 mois (vert - client fidèle)
            {
                'if': {
                    'filter_query': '{anciennete_relation_mois} > 24',
                    'column_id': 'anciennete_relation_mois'
                },
                'backgroundColor': '#bbf7d0',
                'color': '#166534'
            },
            
            # Score Mobile Money > 70 (vert)
            {
                'if': {
                    'filter_query': '{usage_mobile_money_score} > 70',
                    'column_id': 'usage_mobile_money_score'
                },
                'backgroundColor': '#bbf7d0',
                'color': '#166534'
            }
        ],
        filter_action='native',
        sort_action='native'
    )
    
    return (kpi_clients, kpi_revenu, kpi_retard, kpi_montant, kpi_taux_interet, 
            kpi_defaut, kpi_epargne, kpi_duree, fig_region, fig_secteur, 
            fig_dsti, fig_scatter, fig_corr, table)

# Callbacks de téléchargement
@callback(
    Output("download-table", "data"),
    Input("btn-download-table", "n_clicks"),
    [State('region-filter', 'value'),
     State('secteur-filter', 'value'),
     State('canal-filter', 'value'),
     State('montant-slider', 'value')],
    prevent_initial_call=True
)
def download_table(n_clicks, region, secteur, canal, montant_range):
    filtered_df = filter_data(df, region, secteur, canal, montant_range=montant_range)
    return dcc.send_data_frame(filtered_df.to_csv, "donnees_microfinance.csv", index=False)

@callback(
    Output("download-region", "data"),
    Input("btn-download-region", "n_clicks"),
    State('region-chart', 'figure'),
    prevent_initial_call=True
)
def download_region(n_clicks, fig):
    return dcc.send_bytes(
        lambda bytes_io: go.Figure(fig).write_image(bytes_io, format='png', width=800, height=600),
        "repartition_region.png"
    )

@callback(
    Output("download-secteur", "data"),
    Input("btn-download-secteur", "n_clicks"),
    State('secteur-chart', 'figure'),
    prevent_initial_call=True
)
def download_secteur(n_clicks, fig):
    return dcc.send_bytes(
        lambda bytes_io: go.Figure(fig).write_image(bytes_io, format='png', width=800, height=600),
        "repartition_secteur.png"
    )

@callback(
    Output("download-dsti", "data"),
    Input("btn-download-dsti", "n_clicks"),
    State('dsti-graph', 'figure'),
    prevent_initial_call=True
)
def download_dsti(n_clicks, fig):
    return dcc.send_bytes(
        lambda bytes_io: go.Figure(fig).write_image(bytes_io, format='png', width=800, height=600),
        "distribution_dsti.png"
    )

@callback(
    Output("download-scatter", "data"),
    Input("btn-download-scatter", "n_clicks"),
    State('scatter-graph', 'figure'),
    prevent_initial_call=True
)
def download_scatter(n_clicks, fig):
    return dcc.send_bytes(
        lambda bytes_io: go.Figure(fig).write_image(bytes_io, format='png', width=800, height=600),
        "analyse_bivariee.png"
    )

@callback(
    Output("download-corr", "data"),
    Input("btn-download-corr", "n_clicks"),
    State('correlation-graph', 'figure'),
    prevent_initial_call=True
)
def download_corr(n_clicks, fig):
    return dcc.send_bytes(
        lambda bytes_io: go.Figure(fig).write_image(bytes_io, format='png', width=1000, height=800),
        "matrice_correlation.png"
    )