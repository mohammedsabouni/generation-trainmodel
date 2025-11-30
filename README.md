# Système de Diagnostic Multi-Maladies avec Random Forest

Projet de Fin d'Année - 2025

## Description

Ce projet implémente un système de diagnostic médical multi-label utilisant un algorithme Random Forest pour prédire simultanément plusieurs maladies à partir de paramètres cliniques.

Le système est capable de détecter :
- Diabète
- Hypertension
- Anémie
- Maladies cardiovasculaires
- Infections
- État sain

## Architecture du Projet

Le projet est divisé en deux scripts principaux :

### 1. generate_data.py
Script de génération et analyse de données synthétiques.

**Fonctionnalités :**
- Génération de 1800 échantillons de patients synthétiques
- Simulation de paramètres cliniques réalistes basés sur des standards médicaux
- Création de comorbidités (patients avec plusieurs maladies simultanées)
- Analyse statistique complète du dataset
- Visualisations des distributions de données

**Sorties :**
- `medical_dataset_synthetic.csv` : Dataset au format CSV
- `visualizations/` : Dossier contenant les graphiques d'analyse

### 2. train_model.py
Script d'entraînement et évaluation du modèle de machine learning.

**Fonctionnalités :**
- Chargement du dataset généré
- Entraînement d'un modèle Random Forest multi-label
- Évaluation des performances (précision, recall, F1-score)
- Génération de matrices de confusion
- Analyse de l'importance des features
- Prédiction sur de nouveaux patients

**Sorties :**
- `random_forest_medical_model.pkl` : Modèle entraîné sauvegardé
- `model_visualizations/` : Dossier contenant les graphiques d'évaluation

## Installation

### Prérequis
- Python 3.8 ou supérieur

### Dépendances

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

Ou via requirements.txt :

```bash
pip install -r requirements.txt
```

## Utilisation

### Étape 1 : Génération des Données

```bash
python generate_data.py
```

Ce script va :
1. Générer 1800 échantillons de patients avec paramètres cliniques réalistes
2. Sauvegarder le dataset dans `medical_dataset_synthetic.csv`
3. Créer des visualisations dans le dossier `visualizations/`
4. Afficher les statistiques du dataset dans le terminal

**Durée d'exécution :** ~30 secondes

### Étape 2 : Entraînement du Modèle

```bash
python train_model.py
```

Ce script va :
1. Charger le dataset généré
2. Entraîner un modèle Random Forest multi-label
3. Évaluer les performances sur un ensemble de test (20%)
4. Créer des visualisations des résultats dans `model_visualizations/`
5. Sauvegarder le modèle dans `random_forest_medical_model.pkl`
6. Faire une prédiction exemple sur un patient fictif

**Durée d'exécution :** ~1-2 minutes

## Paramètres Cliniques Utilisés

Le modèle utilise 12 features cliniques :

| Feature | Description | Unité |
|---------|-------------|-------|
| age | Âge du patient | années |
| sexe | Sexe (0=Homme, 1=Femme) | - |
| glycemie_jeun | Glycémie à jeun | mg/dL |
| tension_systolique | Tension artérielle systolique | mmHg |
| tension_diastolique | Tension artérielle diastolique | mmHg |
| frequence_cardiaque | Fréquence cardiaque | bpm |
| imc | Indice de Masse Corporelle | kg/m² |
| hemoglobine | Taux d'hémoglobine | g/dL |
| globules_blancs | Numération des globules blancs | ×10³/µL |
| cholesterol_total | Cholestérol total | mg/dL |
| creatinine | Créatinine sérique | mg/dL |
| temperature | Température corporelle | °C |

## Ranges Médicaux Normaux

Le système se base sur les standards cliniques internationaux :

- Glycémie à jeun : 70-100 mg/dL
- Tension systolique : 90-120 mmHg
- Tension diastolique : 60-80 mmHg
- Fréquence cardiaque : 60-100 bpm
- IMC : 18.5-24.9 kg/m²
- Hémoglobine (H) : 13-17 g/dL
- Hémoglobine (F) : 12-15 g/dL
- Globules blancs : 4-11 ×10³/µL
- Cholestérol total : 150-200 mg/dL
- Créatinine : 0.6-1.2 mg/dL
- Température : 36.5-37.5 °C

## Structure du Dataset

Le dataset généré contient :
- 1800 échantillons
- 12 features cliniques
- 6 labels (multi-label) :
  - diabete (0 ou 1)
  - hypertension (0 ou 1)
  - anemie (0 ou 1)
  - maladie_cardiovasculaire (0 ou 1)
  - infection (0 ou 1)
  - sain (0 ou 1)

Distribution approximative :
- 30% patients sains
- 20% diabétiques
- 25% hypertendus
- 15% anémiques
- 18% maladies cardiovasculaires
- 12% infections

Note : Les patients peuvent avoir plusieurs maladies simultanément (comorbidités).

## Configuration du Modèle

### Random Forest
```python
n_estimators = 200          # Nombre d'arbres
max_depth = 15              # Profondeur maximale
min_samples_split = 5       # Échantillons minimum pour split
min_samples_leaf = 2        # Échantillons minimum par feuille
class_weight = 'balanced'   # Gestion du déséquilibre des classes
```

### Split des Données
- Entraînement : 80% (1440 échantillons)
- Test : 20% (360 échantillons)

## Métriques d'Évaluation

Le modèle est évalué avec :
- **Hamming Loss** : Proportion d'erreurs par label
- **Precision** : Proportion de vrais positifs parmi les positifs prédits
- **Recall** : Proportion de vrais positifs parmi les positifs réels
- **F1-Score** : Moyenne harmonique de la précision et du recall
- **Matrices de Confusion** : Pour chaque maladie individuellement

## Visualisations Générées

### generate_data.py crée :
1. `distribution_maladies.png` - Distribution des 6 maladies
2. `distribution_features.png` - Histogrammes des 12 paramètres cliniques
3. `correlation_matrix.png` - Matrice de corrélation features/labels
4. `comorbidites.png` - Distribution des comorbidités

### train_model.py crée :
1. `confusion_matrices.png` - Matrices de confusion pour les 6 maladies
2. `metrics_comparison.png` - Comparaison Precision/Recall/F1 par maladie
3. `feature_importance.png` - Importance des features pour chaque maladie

## Exemple d'Utilisation du Modèle

```python
import joblib
import numpy as np

# Charger le modèle
model = joblib.load('random_forest_medical_model.pkl')

# Nouveau patient
patient = np.array([[
    58,      # age
    0,       # sexe (Homme)
    165,     # glycemie_jeun
    155,     # tension_systolique
    95,      # tension_diastolique
    78,      # frequence_cardiaque
    29.5,    # imc
    14.2,    # hemoglobine
    7.5,     # globules_blancs
    245,     # cholesterol_total
    1.0,     # creatinine
    36.8     # temperature
]])

# Prédiction
prediction = model.predict(patient)
# Retourne : [diabete, hypertension, anemie, cardiovasculaire, infection, sain]
```

## Fichiers Générés

Après exécution complète :

```
.
├── generate_data.py
├── train_model.py
├── README.md
├── medical_dataset_synthetic.csv
├── random_forest_medical_model.pkl
├── visualizations/
│   ├── distribution_maladies.png
│   ├── distribution_features.png
│   ├── correlation_matrix.png
│   └── comorbidites.png
└── model_visualizations/
    ├── confusion_matrices.png
    ├── metrics_comparison.png
    └── feature_importance.png
```

## Limitations et Considérations

1. **Données Synthétiques** : Les données sont générées artificiellement et ne remplacent pas des données médicales réelles.

2. **Usage Éducatif** : Ce système est conçu à des fins pédagogiques et de démonstration. Il ne doit pas être utilisé pour de vrais diagnostics médicaux.

3. **Simplifications** : Le modèle simplifie la complexité des diagnostics médicaux réels qui nécessitent l'expertise de professionnels de santé.

4. **Validation** : Le modèle n'a pas été validé sur des données cliniques réelles.

## Personnalisation

### Modifier le nombre d'échantillons

Dans `generate_data.py` :
```python
N_SAMPLES = 1800  # Modifier cette valeur
```

### Modifier la distribution des maladies

Dans `generate_data.py` :
```python
TARGET_DISTRIBUTION = {
    'sain': 0.30,
    'diabete': 0.20,
    'hypertension': 0.25,
    'anemie': 0.15,
    'maladie_cardiovasculaire': 0.18,
    'infection': 0.12
}
```

### Modifier les paramètres du modèle

Dans `train_model.py` :
```python
rf = RandomForestClassifier(
    n_estimators=200,      # Modifier ici
    max_depth=15,          # Modifier ici
    min_samples_split=5,   # Modifier ici
    # ...
)
```

## Dépannage

### Erreur : "Le fichier medical_dataset_synthetic.csv n'a pas été trouvé"
- Solution : Exécutez d'abord `python generate_data.py`

### Erreur de mémoire
- Solution : Réduisez `N_SAMPLES` dans `generate_data.py`

### Problèmes d'installation de packages
- Solution : Utilisez un environnement virtuel Python
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

<!-- 
## Auteur

Sabouni Mohammed Amine - 2025

 -->
