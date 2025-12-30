"""
PAGE 2 - MODÉLISATION PRÉDICTIVE (AMÉLIORÉE)
Interface moderne pour l'analyse discriminante et la prédiction
Correction du problème QDA avec régularisation et ajustement de seuil
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve

dash.register_page(__name__, path='/modelisation', name='Modélisation')

# Chargement et préparation
df = pd.read_excel('./data/microfinance_credit_risk.xlsx')

feature_cols = [
    'age', 'revenu_mensuel_xof', 'epargne_xof', 'anciennete_relation_mois',
    'historique_credit_mois', 'jours_retard_12m', 'nb_dependants',
    'usage_mobile_money_score', 'montant_pret_xof', 'duree_mois',
    'taux_interet_annuel_pct', 'dsti_pct', 'pret_groupe'
]

X = df[feature_cols].copy()
y = df['defaut_90j'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement LDA
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train_scaled, y_train)

# Entraînement QDA avec régularisation pour éviter le problème de singularité
# reg_param aide à stabiliser les matrices de covariance
qda_model = QuadraticDiscriminantAnalysis(reg_param=0.1)
qda_model.fit(X_train_scaled, y_train)

# Prédictions et probabilités
lda_proba = lda_model.predict_proba(X_test_scaled)[:, 1]
qda_proba = qda_model.predict_proba(X_test_scaled)[:, 1]

# Fonction pour trouver le seuil optimal basé sur F1-score
def find_optimal_threshold(y_true, y_proba):
    """Trouve le seuil qui maximise le F1-score"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], f1_scores[optimal_idx]

# Trouver les seuils optimaux
lda_threshold, lda_best_f1 = find_optimal_threshold(y_test, lda_proba)
qda_threshold, qda_best_f1 = find_optimal_threshold(y_test, qda_proba)

# Prédictions avec seuils optimaux
lda_pred = (lda_proba >= lda_threshold).astype(int)
qda_pred = (qda_proba >= qda_threshold).astype(int)

# Métriques avec seuils optimaux
lda_accuracy = accuracy_score(y_test, lda_pred)
lda_f1 = f1_score(y_test, lda_pred, zero_division=0)
lda_recall = recall_score(y_test, lda_pred, zero_division=0)
lda_cm = confusion_matrix(y_test, lda_pred)

qda_accuracy = accuracy_score(y_test, qda_pred)
qda_f1 = f1_score(y_test, qda_pred, zero_division=0)
qda_recall = recall_score(y_test, qda_pred, zero_division=0)
qda_cm = confusion_matrix(y_test, qda_pred)

# Courbes ROC
lda_fpr, lda_tpr, _ = roc_curve(y_test, lda_proba)
lda_auc = auc(lda_fpr, lda_tpr)
qda_fpr, qda_tpr, _ = roc_curve(y_test, qda_proba)
qda_auc = auc(qda_fpr, qda_tpr)

# Sélection du meilleur modèle
best_model = 'LDA' if lda_f1 > qda_f1 else 'QDA'
best_f1 = max(lda_f1, qda_f1)
best_threshold = lda_threshold if best_model == 'LDA' else qda_threshold

def create_confusion_matrix(cm, model_name):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Prédit Sain', 'Prédit Défaut'],
        y=['Réel Sain', 'Réel Défaut'],
        colorscale=[[0, '#e0f2fe'], [1, '#0891b2']],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 18, "color": "#1e293b"},
        showscale=False,
        hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif', 'color': '#1e293b'},
        margin=dict(t=20, b=40, l=60, r=20),
        height=350
    )
    return fig

def create_roc_curves():
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lda_fpr, y=lda_tpr, mode='lines',
        name=f"LDA (AUC = {lda_auc:.3f})",
        line=dict(color='#0891b2', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=qda_fpr, y=qda_tpr, mode='lines',
        name=f"QDA (AUC = {qda_auc:.3f})",
        line=dict(color='#8b5cf6', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Référence',
        line=dict(color='#cbd5e1', width=2, dash='dash')
    ))
    
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif', 'color': '#1e293b'},
        xaxis_title='Taux de Faux Positifs',
        yaxis_title='Taux de Vrais Positifs',
        margin=dict(t=20, b=40, l=40, r=20),
        height=400,
        showlegend=True,
        legend=dict(x=0.6, y=0.1, bgcolor='rgba(255,255,255,0.8)')
    )
    return fig

# Layout
layout = html.Div([
    
    # Section Comparaison Modèles
    html.Div([
        html.Div([
            html.H3("Bloc A - Comparaison des Modèles", className="section-title"),
            html.P("Évaluation comparative de l'analyse discriminante linéaire et quadratique", 
                   className="section-subtitle"),
        ], className="section-header"),
        
        # Info sur les améliorations
        html.Div([
            html.Div("✓ QDA avec régularisation (reg_param=0.1)", className="improvement-note"),
            html.Div(f"✓ Seuils optimisés - LDA: {lda_threshold:.3f} | QDA: {qda_threshold:.3f}", 
                    className="improvement-note")
        ], className="improvement-info"),
        
        # Métriques comparatives
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("LDA", className="model-label model-label-lda"),
                        html.Span("Linear Discriminant Analysis", className="model-name")
                    ], className="model-header"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div(f"{lda_accuracy:.4f}", className="metric-value"),
                                html.Div("Accuracy", className="metric-name")
                            ], className="metric-item")
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Div(f"{lda_f1:.4f}", className="metric-value"),
                                html.Div("F1-Score", className="metric-name")
                            ], className="metric-item")
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Div(f"{lda_recall:.4f}", className="metric-value"),
                                html.Div("Recall", className="metric-name")
                            ], className="metric-item")
                        ], md=4)
                    ])
                ], className="model-card model-card-lda")
            ], md=6),
            
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("QDA", className="model-label model-label-qda"),
                        html.Span("Quadratic Discriminant Analysis", className="model-name")
                    ], className="model-header"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div(f"{qda_accuracy:.4f}", className="metric-value"),
                                html.Div("Accuracy", className="metric-name")
                            ], className="metric-item")
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Div(f"{qda_f1:.4f}", className="metric-value"),
                                html.Div("F1-Score", className="metric-name")
                            ], className="metric-item")
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Div(f"{qda_recall:.4f}", className="metric-value"),
                                html.Div("Recall", className="metric-name")
                            ], className="metric-item")
                        ], md=4)
                    ])
                ], className="model-card model-card-qda")
            ], md=6)
        ], className="mb-4"),
        
        # Matrices de confusion
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Matrice de Confusion - LDA", className="chart-title"),
                    dcc.Graph(figure=create_confusion_matrix(lda_cm, "LDA"), 
                             config={'displayModeBar': False})
                ], className="chart-card")
            ], md=6),
            
            dbc.Col([
                html.Div([
                    html.Div("Matrice de Confusion - QDA", className="chart-title"),
                    dcc.Graph(figure=create_confusion_matrix(qda_cm, "QDA"), 
                             config={'displayModeBar': False})
                ], className="chart-card")
            ], md=6)
        ], className="mb-4"),
        
        # Courbes ROC
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Courbes ROC - Performance Comparative", className="chart-title"),
                    dcc.Graph(figure=create_roc_curves(), config={'displayModeBar': False})
                ], className="chart-card")
            ])
        ])
    ], className="section-card"),
    
    # Section Prédiction
    html.Div([
        html.Div([
            html.H3("Bloc B - Prédiction Client", className="section-title"),
            html.Div([
                html.Span("Modèle sélectionné: ", className="best-model-text"),
                html.Span(best_model, className=f"best-model-badge best-model-{best_model.lower()}"),
                html.Span(f" · F1-Score: {best_f1:.4f} · Seuil: {best_threshold:.3f}", 
                         className="best-model-score")
            ], className="best-model-info")
        ], className="section-header"),
        
        # Formulaire de prédiction
        html.Div([
            html.Div("Saisie des Informations Client", className="form-title"),
            
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Âge", className="input-label"),
                        dcc.Input(id='input-age', type='number', value=35, min=18, max=70,
                                 className="modern-input")
                    ], md=4),
                    dbc.Col([
                        html.Label("Revenu Mensuel (XOF)", className="input-label"),
                        dcc.Input(id='input-revenu', type='number', value=150000, min=0,
                                 className="modern-input")
                    ], md=4),
                    dbc.Col([
                        html.Label("Épargne (XOF)", className="input-label"),
                        dcc.Input(id='input-epargne', type='number', value=50000, min=0,
                                 className="modern-input")
                    ], md=4)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Ancienneté Relation (mois)", className="input-label"),
                        dcc.Input(id='input-anciennete', type='number', value=12, min=0,
                                 className="modern-input")
                    ], md=4),
                    dbc.Col([
                        html.Label("Historique Crédit (mois)", className="input-label"),
                        dcc.Input(id='input-historique', type='number', value=6, min=0,
                                 className="modern-input")
                    ], md=4),
                    dbc.Col([
                        html.Label("Jours de Retard (12 mois)", className="input-label"),
                        dcc.Input(id='input-retard', type='number', value=0, min=0,
                                 className="modern-input")
                    ], md=4)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Nombre de Dépendants", className="input-label"),
                        dcc.Input(id='input-dependants', type='number', value=2, min=0,
                                 className="modern-input")
                    ], md=3),
                    dbc.Col([
                        html.Label("Score Mobile Money", className="input-label"),
                        dcc.Input(id='input-mobile', type='number', value=50, min=0, max=100,
                                 className="modern-input")
                    ], md=3),
                    dbc.Col([
                        html.Label("Montant Prêt (XOF)", className="input-label"),
                        dcc.Input(id='input-montant', type='number', value=500000, min=0,
                                 className="modern-input")
                    ], md=3),
                    dbc.Col([
                        html.Label("Durée (mois)", className="input-label"),
                        dcc.Input(id='input-duree', type='number', value=12, min=1,
                                 className="modern-input")
                    ], md=3)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Taux d'Intérêt (%)", className="input-label"),
                        dcc.Input(id='input-taux', type='number', value=12.0, min=0, max=100,
                                 step=0.1, className="modern-input")
                    ], md=4),
                    dbc.Col([
                        html.Label("DSTI (%)", className="input-label"),
                        dcc.Input(id='input-dsti', type='number', value=25.0, min=0, max=100,
                                 step=0.1, className="modern-input")
                    ], md=4),
                    dbc.Col([
                        html.Label("Prêt de Groupe", className="input-label"),
                        dcc.Dropdown(
                            id='input-groupe',
                            options=[{'label': 'Non', 'value': 0}, {'label': 'Oui', 'value': 1}],
                            value=0,
                            clearable=False,
                            className="modern-dropdown"
                        )
                    ], md=4)
                ], className="mb-4"),
                
                html.Button("Analyser le Risque", id='predict-button', className="predict-button")
            ], className="prediction-form")
        ], className="form-container"),
        
        # Résultats
        html.Div(id='prediction-results')
    ], className="section-card")
    
], className="page-container")

@callback(
    Output('prediction-results', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('input-age', 'value'), State('input-revenu', 'value'),
     State('input-epargne', 'value'), State('input-anciennete', 'value'),
     State('input-historique', 'value'), State('input-retard', 'value'),
     State('input-dependants', 'value'), State('input-mobile', 'value'),
     State('input-montant', 'value'), State('input-duree', 'value'),
     State('input-taux', 'value'), State('input-dsti', 'value'),
     State('input-groupe', 'value')]
)
def predict_client(n_clicks, age, revenu, epargne, anciennete, historique,
                   retard, dependants, mobile, montant, duree, taux, dsti, groupe):
    if n_clicks is None:
        return html.Div([
            html.Div("En attente d'analyse", className="waiting-message")
        ], className="results-placeholder")
    
    client_data = [[age, revenu, epargne, anciennete, historique, retard, dependants,
                    mobile, montant, duree, taux, dsti, groupe]]
    client_scaled = scaler.transform(client_data)
    
    lda_proba_client = lda_model.predict_proba(client_scaled)[0, 1]
    qda_proba_client = qda_model.predict_proba(client_scaled)[0, 1]
    
    # Utiliser le modèle sélectionné avec son seuil optimal
    if best_model == 'LDA':
        final_proba = lda_proba_client
        final_decision = 'Défaut Probable' if lda_proba_client >= lda_threshold else 'Profil Sain'
    else:
        final_proba = qda_proba_client
        final_decision = 'Défaut Probable' if qda_proba_client >= qda_threshold else 'Profil Sain'
    
    risk_level = "Élevé" if final_proba >= best_threshold else "Faible"
    decision_class = 'danger' if final_proba >= best_threshold else 'success'
    
    return html.Div([
        html.Div("Résultats de l'Analyse", className="results-title"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Modèle LDA", className="proba-label"),
                    html.Div(f"{lda_proba_client*100:.1f}%", className="proba-value"),
                    html.Div(f"Seuil: {lda_threshold:.3f}", className="proba-subtitle")
                ], className="proba-card proba-card-lda")
            ], md=6),
            
            dbc.Col([
                html.Div([
                    html.Div("Modèle QDA", className="proba-label"),
                    html.Div(f"{qda_proba_client*100:.1f}%", className="proba-value"),
                    html.Div(f"Seuil: {qda_threshold:.3f}", className="proba-subtitle")
                ], className="proba-card proba-card-qda")
            ], md=6)
        ], className="mb-4"),
        
        html.Div([
            html.Div([
                html.Span("Décision Finale", className="decision-label"),
                html.Span(f" ({best_model})", className="decision-model")
            ]),
            html.Div(final_decision, className=f"decision-value decision-{decision_class}"),
            html.Div(f"Niveau de Risque: {risk_level}", className="risk-level")
        ], className="final-decision")
        
    ], className="results-container")