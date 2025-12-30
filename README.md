# Guide d'Installation et de Lancement

## Installation des Dépendances

### 1. Créer l'environnement virtuel (si pas déjà fait)
```bash
python -m venv venv
```

### 2. Activer l'environnement virtuel

**Sur macOS/Linux:**
```bash
source venv/bin/activate
```

**Sur Windows:**
```bash
venv\Scripts\activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```


---

## Lancement de l'Application

### Mode Développement (Local)
```bash
python app.py
```

L'application sera accessible sur: [http://localhost:8050](http://localhost:8050)

### Mode Production (avec render)
L'application sera accessible sur: https://data-viz-wilfred-tchapda.onrender.com/

---

## Structure du Projet

```
Projet_Data_viz/
├── app.py                          # Application principale Dash
├── requirements.txt                # Dépendances (avec lightgbm)
├── assets/
│   └── style.css                   # Styles CSS 
├── pages/
│   ├── page1_exploration.py        # Page d'exploration (avec couleurs Sahel)
│   └── page2_modelisation.py       # Page de modélisation (LDA, QDA, LGBM)
├── data/
│   └── microfinance_credit_risk.xlsx
├── SEN_adm/
│   └── SEN_adm1.shp                # Shapefile du Sénégal            # 
└── GUIDE_INSTALLATION.md           # Ce fichier
```

---


## Test de Compilation
```bash
python -m py_compile app.py pages/*.py
echo "Compilation réussie!"
```

---

## Fonctionnalités Principales

### Page 1 - Exploration des Données
- Filtres interactifs (région, secteur, canal, montant)
- 8 KPIs dynamiques
- Carte choroplèthe du Sénégal (avec palette Sahel)
- Graphiques sectoriels et analyses bivariées
- Matrice de corrélation personnalisée
- Tableau de données avec coloration conditionnelle
- Export CSV et PNG

### Page 2 - Modélisation Prédictive
- **3 modèles comparés:**
  1. LDA (Linear Discriminant Analysis)
  2. QDA (Quadratic Discriminant Analysis)
  3. **LightGBM** (Light Gradient Boosting Machine) - NOUVEAU!

- Métriques pour chaque modèle: Accuracy, F1-Score, Recall
- 3 matrices de confusion
- Courbe ROC comparative
- **Sélection automatique du meilleur modèle** (basée sur F1-score)
- Prédiction client avec affichage des 3 probabilités




---

## Dépannage

### Erreur: "ModuleNotFoundError: No module named 'lightgbm'"
**Solution:**
```bash
pip install lightgbm>=4.0.0
```

### Erreur de chargement des données
Vérifiez que les fichiers suivants existent:
- `data/microfinance_credit_risk.xlsx`
- `SEN_adm/SEN_adm1.shp`

### Port 8050 déjà utilisé
Changez le port dans `app.py`:
```python
app.run_server(debug=True, host='0.0.0.0', port=8051)  # Autre port
```

---

## Prochaines Étapes

1. Installer les dépendances
2. Lancer l'application
3. Explorer les 2 pages
4. Comparer les 3 modèles
5. Tester des prédictions

---

## Support

- Contacter: Wilfred Rod TCHAPDA KOUADJO - ENSAE Dakar

**Bonne utilisation!**
