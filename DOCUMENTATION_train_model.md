# Documentation Complète - train_model.py

## Vue d'ensemble

Le script `train_model.py` est responsable de l'entraînement, de l'évaluation et de la sauvegarde d'un modèle de machine learning Random Forest multi-label pour le diagnostic de maladies multiples.

---

## Table des matières

1. [Imports et Configuration](#1-imports-et-configuration)
2. [Fonction load_dataset()](#2-fonction-load_dataset)
3. [Fonction train_ml_model()](#3-fonction-train_ml_model)
4. [Fonction visualize_model_results()](#4-fonction-visualize_model_results)
5. [Fonction visualize_feature_importance()](#5-fonction-visualize_feature_importance)
6. [Fonction predict_new_patient()](#6-fonction-predict_new_patient)
7. [Fonction main()](#7-fonction-main)
8. [Flux d'exécution complet](#8-flux-dexécution-complet)

---

## 1. Imports et Configuration

### Description
Cette section initialise l'environnement de travail en important les bibliothèques nécessaires et en configurant les paramètres globaux.

### Code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    hamming_loss, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import joblib
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
```

### Étapes détaillées

#### Étape 1.1 : Import des bibliothèques de base
- **numpy** : Calculs numériques et manipulation de tableaux
- **pandas** : Manipulation et analyse de données tabulaires
- **matplotlib.pyplot** : Création de graphiques
- **seaborn** : Visualisations statistiques avancées

#### Étape 1.2 : Import des outils scikit-learn
- **train_test_split** : Division des données en ensembles d'entraînement et de test
- **RandomForestClassifier** : Algorithme d'arbre de décision en ensemble
- **MultiOutputClassifier** : Wrapper pour gérer la classification multi-label
- **Métriques** : Outils d'évaluation des performances du modèle

#### Étape 1.3 : Import des outils de persistance
- **joblib** : Sauvegarde et chargement du modèle entraîné

#### Étape 1.4 : Configuration globale
- **warnings.filterwarnings('ignore')** : Supprime les avertissements pour une sortie plus propre
- **SEED = 42** : Graine aléatoire pour la reproductibilité des résultats
- **np.random.seed(SEED)** : Initialise le générateur de nombres aléatoires

---

## 2. Fonction load_dataset()

### Signature
```python
def load_dataset(filepath='medical_dataset_synthetic.csv')
```

### Description
Charge le dataset médical depuis un fichier CSV et vérifie son existence.

### Paramètres
- **filepath** (str, optionnel) : Chemin vers le fichier CSV. Par défaut : 'medical_dataset_synthetic.csv'

### Retour
- **DataFrame pandas** : Dataset chargé en mémoire
- **None** : Si le fichier n'existe pas

### Étapes détaillées

#### Étape 2.1 : Affichage du header
```python
print("=" * 70)
print(" CHARGEMENT DES DONNEES ".center(70))
print("=" * 70)
```
- Affiche un en-tête formaté pour la section de chargement

#### Étape 2.2 : Tentative de chargement
```python
try:
    df = pd.read_csv(filepath)
```
- Utilise `pd.read_csv()` pour lire le fichier CSV
- Place le code dans un bloc try-except pour gérer les erreurs

#### Étape 2.3 : Vérification et affichage des informations
```python
print(f"\nDataset charge avec succes: {filepath}")
print(f"   - Nombre d'echantillons: {len(df)}")
print(f"   - Nombre de colonnes: {len(df.columns)}")
```
- Affiche le nom du fichier chargé
- Compte et affiche le nombre de lignes (échantillons)
- Compte et affiche le nombre de colonnes

#### Étape 2.4 : Gestion des erreurs
```python
except FileNotFoundError:
    print(f"\nERREUR: Le fichier '{filepath}' n'a pas ete trouve.")
    print("Veuillez d'abord executer 'generate_data.py' pour generer le dataset.")
    return None
```
- Capture l'exception si le fichier n'existe pas
- Affiche un message d'erreur explicite
- Guide l'utilisateur vers la solution (exécuter generate_data.py)
- Retourne None pour indiquer l'échec

#### Étape 2.5 : Retour du DataFrame
```python
return df
```
- Retourne le DataFrame chargé avec succès

### Exemple d'utilisation
```python
df = load_dataset('medical_dataset_synthetic.csv')
if df is not None:
    print("Dataset pret a etre utilise")
```

---

## 3. Fonction train_ml_model()

### Signature
```python
def train_ml_model(df)
```

### Description
Fonction principale qui entraîne le modèle Random Forest multi-label, évalue ses performances et génère les métriques d'évaluation complètes.

### Paramètres
- **df** (DataFrame pandas) : Dataset contenant les features et labels

### Retour
Tuple contenant :
- **model** : Modèle MultiOutputClassifier entraîné
- **X_test** : Features de test
- **y_test** : Labels réels de test
- **y_pred** : Prédictions sur l'ensemble de test
- **feature_columns** : Liste des noms de features
- **label_columns** : Liste des noms de labels

### Étapes détaillées

#### Étape 3.1 : Affichage du header
```python
print("=" * 70)
print(" ENTRAINEMENT DU MODELE RANDOM FOREST MULTI-LABEL ".center(70))
print("=" * 70)
```
- Affiche un en-tête formaté pour la section d'entraînement

#### Étape 3.2 : Définition des features et labels
```python
feature_columns = ['age', 'sexe', 'glycemie_jeun', 'tension_systolique',
                   'tension_diastolique', 'frequence_cardiaque', 'imc',
                   'hemoglobine', 'globules_blancs', 'cholesterol_total',
                   'creatinine', 'temperature']

label_columns = ['diabete', 'hypertension', 'anemie',
                 'maladie_cardiovasculaire', 'infection', 'sain']
```
- **feature_columns** : Liste des 12 paramètres cliniques utilisés comme entrées
- **label_columns** : Liste des 6 maladies à prédire (classification multi-label)

#### Étape 3.3 : Extraction des données
```python
X = df[feature_columns].values
y = df[label_columns].values
```
- **X** : Matrice numpy de forme (n_samples, 12) contenant les features
- **y** : Matrice numpy de forme (n_samples, 6) contenant les labels binaires
- `.values` convertit le DataFrame pandas en array numpy

#### Étape 3.4 : Division train/test
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y[:, 0]
)
```
- **test_size=0.2** : 20% des données pour le test, 80% pour l'entraînement
- **random_state=SEED** : Reproductibilité de la division
- **stratify=y[:, 0]** : Stratification sur la première colonne de labels (diabète) pour maintenir la proportion des classes
- Résultat : 4 arrays numpy (X_train, X_test, y_train, y_test)

#### Étape 3.5 : Affichage des dimensions
```python
print(f"\nDonnees preparees:")
print(f"   - Features: {X_train.shape[1]}")
print(f"   - Labels: {y_train.shape[1]}")
print(f"   - Echantillons d'entrainement: {X_train.shape[0]}")
print(f"   - Echantillons de test: {X_test.shape[0]}")
```
- Affiche le nombre de features (12)
- Affiche le nombre de labels (6)
- Affiche le nombre d'échantillons d'entraînement (environ 1440)
- Affiche le nombre d'échantillons de test (environ 360)

#### Étape 3.6 : Configuration du Random Forest
```python
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=SEED,
    n_jobs=-1,
    class_weight='balanced'
)
```
- **n_estimators=200** : Nombre d'arbres de décision dans la forêt
- **max_depth=15** : Profondeur maximale de chaque arbre (évite le surapprentissage)
- **min_samples_split=5** : Nombre minimum d'échantillons pour diviser un nœud
- **min_samples_leaf=2** : Nombre minimum d'échantillons requis dans une feuille
- **random_state=SEED** : Reproductibilité des résultats
- **n_jobs=-1** : Utilise tous les cœurs CPU disponibles
- **class_weight='balanced'** : Ajuste les poids pour gérer le déséquilibre des classes

#### Étape 3.7 : Création du wrapper multi-label
```python
model = MultiOutputClassifier(rf)
```
- Encapsule le RandomForestClassifier dans un MultiOutputClassifier
- Permet de traiter chaque label indépendamment
- Crée 6 modèles Random Forest distincts (un par maladie)

#### Étape 3.8 : Entraînement du modèle
```python
model.fit(X_train, y_train)
```
- Entraîne les 6 modèles Random Forest sur les données d'entraînement
- Chaque modèle apprend à prédire une maladie spécifique
- Processus d'apprentissage basé sur les arbres de décision

#### Étape 3.9 : Prédictions sur l'ensemble de test
```python
y_pred = model.predict(X_test)
```
- Génère des prédictions pour tous les échantillons de test
- Retourne une matrice (n_test_samples, 6) avec des valeurs binaires 0 ou 1
- Chaque colonne représente la prédiction pour une maladie

#### Étape 3.10 : Calcul du Hamming Loss
```python
h_loss = hamming_loss(y_test, y_pred)
print(f"\nHamming Loss: {h_loss:.4f}")
```
- **Hamming Loss** : Fraction de labels incorrectement prédits
- Formule : (nombre total d'erreurs) / (nombre total de prédictions)
- Valeur entre 0 (parfait) et 1 (toutes les prédictions fausses)
- Plus bas = meilleur

#### Étape 3.11 : Calcul des métriques macro-averaged
```python
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
```
- **average='macro'** : Calcule la métrique pour chaque label puis fait la moyenne
- **zero_division=0** : Retourne 0 si division par zéro (évite les warnings)

**Precision** :
- Proportion de vrais positifs parmi tous les positifs prédits
- Formule : TP / (TP + FP)
- Répond à : "Parmi tous les cas diagnostiqués, combien sont vraiment malades ?"

**Recall** :
- Proportion de vrais positifs parmi tous les positifs réels
- Formule : TP / (TP + FN)
- Répond à : "Parmi tous les malades, combien ont été détectés ?"

**F1-Score** :
- Moyenne harmonique de Precision et Recall
- Formule : 2 × (Precision × Recall) / (Precision + Recall)
- Balance entre precision et recall

#### Étape 3.12 : Calcul des métriques par maladie
```python
for idx, label in enumerate(label_columns):
    prec = precision_score(y_test[:, idx], y_pred[:, idx], zero_division=0)
    rec = recall_score(y_test[:, idx], y_pred[:, idx], zero_division=0)
    f1 = f1_score(y_test[:, idx], y_pred[:, idx], zero_division=0)
    print(f"   {label.capitalize():<30} {prec:>7.4f}      {rec:>7.4f}      {f1:>7.4f}")
```
- Itère sur chaque maladie (colonne de labels)
- Calcule les 3 métriques pour chaque maladie individuellement
- Affiche un tableau formaté avec les résultats
- **y_test[:, idx]** : Extrait la colonne idx des labels réels
- **y_pred[:, idx]** : Extrait la colonne idx des prédictions

#### Étape 3.13 : Calcul et affichage des matrices de confusion
```python
for idx, label in enumerate(label_columns):
    cm = confusion_matrix(y_test[:, idx], y_pred[:, idx])
    
    print(f"\n   {label.capitalize()}:")
    print(f"      TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"      FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
```
- Génère une matrice de confusion pour chaque maladie
- **Matrice de confusion** : Tableau 2×2 montrant les prédictions vs réalité

Structure de la matrice :
```
                Prédit Négatif    Prédit Positif
Réel Négatif         TN               FP
Réel Positif         FN               TP
```

- **TN (True Negative)** : Correctement prédit comme sain
- **FP (False Positive)** : Incorrectement prédit comme malade
- **FN (False Negative)** : Malade manqué
- **TP (True Positive)** : Correctement prédit comme malade

#### Étape 3.14 : Retour des résultats
```python
return model, X_test, y_test, y_pred, feature_columns, label_columns
```
- Retourne tous les éléments nécessaires pour les analyses ultérieures
- Le modèle entraîné peut être sauvegardé et réutilisé
- Les données de test permettent d'autres analyses
- Les listes de colonnes permettent l'interprétation des résultats

### Exemple de sortie
```
==================================================================
     ENTRAINEMENT DU MODELE RANDOM FOREST MULTI-LABEL      
==================================================================

Donnees preparees:
   - Features: 12
   - Labels: 6
   - Echantillons d'entrainement: 1440
   - Echantillons de test: 360

Entrainement du modele Random Forest...
   Modele entraine avec succes

==================================================================
                 METRIQUES DE PERFORMANCE                 
==================================================================

Hamming Loss: 0.0342
(Proportion d'erreurs par label, 0 = parfait)

Metriques Macro-Averaged (moyenne sur toutes les maladies):
   - Precision: 0.9245
   - Recall:    0.9187
   - F1-Score:  0.9215

Metriques par maladie:
   ------------------------------------------------------------------
   Maladie                        Precision    Recall       F1-Score    
   ------------------------------------------------------------------
   Diabete                         0.9524      0.9524      0.9524
   Hypertension                    0.9302      0.9302      0.9302
   ...
```

---

## 4. Fonction visualize_model_results()

### Signature
```python
def visualize_model_results(y_test, y_pred, label_columns, save_path='model_visualizations')
```

### Description
Génère des visualisations graphiques des performances du modèle, incluant les matrices de confusion et les comparaisons de métriques.

### Paramètres
- **y_test** (numpy array) : Labels réels de l'ensemble de test, forme (n_samples, 6)
- **y_pred** (numpy array) : Prédictions du modèle, forme (n_samples, 6)
- **label_columns** (list) : Liste des noms des 6 maladies
- **save_path** (str, optionnel) : Dossier où sauvegarder les graphiques. Par défaut : 'model_visualizations'

### Retour
- Aucun (génère et sauvegarde des fichiers PNG)

### Étapes détaillées

#### Étape 4.1 : Création du dossier de sortie
```python
import os
os.makedirs(save_path, exist_ok=True)
```
- Importe le module os pour les opérations sur le système de fichiers
- Crée le dossier s'il n'existe pas
- **exist_ok=True** : Ne génère pas d'erreur si le dossier existe déjà

#### Étape 4.2 : Configuration du style
```python
sns.set_style("whitegrid")
```
- Configure le style des graphiques seaborn
- "whitegrid" : Fond blanc avec grille

#### Étape 4.3 : Création de la figure pour matrices de confusion
```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Matrices de Confusion - Classification Multi-Label', 
             fontsize=16, fontweight='bold')
```
- **plt.subplots(2, 3)** : Crée une grille 2×3 de sous-graphiques
- **figsize=(15, 10)** : Taille de la figure en pouces
- **suptitle** : Titre principal de la figure

#### Étape 4.4 : Génération des matrices de confusion
```python
for idx, (label, ax) in enumerate(zip(label_columns, axes.flat)):
    cm = confusion_matrix(y_test[:, idx], y_pred[:, idx])
```
- Itère sur chaque maladie et son sous-graphique correspondant
- **enumerate()** : Obtient l'index et la valeur
- **zip()** : Associe chaque label à son subplot
- **axes.flat** : Transforme la grille 2D en liste 1D
- **confusion_matrix()** : Calcule la matrice de confusion pour cette maladie

#### Étape 4.5 : Création de la heatmap
```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Non', 'Oui'],
            yticklabels=['Non', 'Oui'],
            cbar=False)
```
- **annot=True** : Affiche les valeurs dans les cellules
- **fmt='d'** : Format entier pour les annotations
- **cmap='Blues'** : Palette de couleurs bleues
- **ax=ax** : Spécifie le subplot cible
- **xticklabels/yticklabels** : Labels des axes
- **cbar=False** : Désactive la barre de couleur

#### Étape 4.6 : Configuration des axes
```python
ax.set_title(f'{label.capitalize()}', fontweight='bold')
ax.set_ylabel('Reel')
ax.set_xlabel('Predit')
```
- Définit le titre du subplot (nom de la maladie)
- Labels des axes Y (valeurs réelles) et X (prédictions)

#### Étape 4.7 : Sauvegarde du graphique
```python
plt.tight_layout()
plt.savefig(f'{save_path}/confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"   Sauvegarde: {save_path}/confusion_matrices.png")
plt.close()
```
- **tight_layout()** : Ajuste automatiquement l'espacement
- **savefig()** : Sauvegarde l'image
- **dpi=300** : Résolution haute qualité
- **bbox_inches='tight'** : Rogne les espaces blancs
- **close()** : Libère la mémoire

#### Étape 4.8 : Préparation des métriques pour comparaison
```python
metrics = {
    'Precision': [],
    'Recall': [],
    'F1-Score': []
}

for idx in range(len(label_columns)):
    metrics['Precision'].append(precision_score(y_test[:, idx], y_pred[:, idx], zero_division=0))
    metrics['Recall'].append(recall_score(y_test[:, idx], y_pred[:, idx], zero_division=0))
    metrics['F1-Score'].append(f1_score(y_test[:, idx], y_pred[:, idx], zero_division=0))
```
- Crée un dictionnaire pour stocker les métriques
- Pour chaque maladie, calcule les 3 métriques
- Stocke les valeurs dans des listes

#### Étape 4.9 : Création du graphique de comparaison
```python
fig, ax = plt.subplots(figsize=(12, 6))
```
- Crée une nouvelle figure pour le graphique en barres

#### Étape 4.10 : Création des barres groupées
```python
x = np.arange(len(label_columns))
width = 0.25

ax.bar(x - width, metrics['Precision'], width, label='Precision', color='#3498db')
ax.bar(x, metrics['Recall'], width, label='Recall', color='#2ecc71')
ax.bar(x + width, metrics['F1-Score'], width, label='F1-Score', color='#e74c3c')
```
- **x** : Positions sur l'axe X (une par maladie)
- **width** : Largeur de chaque barre
- Trois séries de barres décalées horizontalement :
  - Precision (bleue) à gauche
  - Recall (verte) au centre
  - F1-Score (rouge) à droite

#### Étape 4.11 : Configuration du graphique
```python
ax.set_xlabel('Maladies', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Comparaison des Metriques par Maladie', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([label.capitalize() for label in label_columns], rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)
```
- Labels des axes
- Titre du graphique
- Positions et labels de l'axe X
- **rotation=45** : Incline les labels pour lisibilité
- **ha='right'** : Alignement horizontal à droite
- **legend()** : Affiche la légende
- **set_ylim([0, 1.1])** : Limite Y de 0 à 1.1
- **grid()** : Ajoute une grille horizontale

#### Étape 4.12 : Sauvegarde finale
```python
plt.tight_layout()
plt.savefig(f'{save_path}/metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
```
- Sauvegarde le graphique de comparaison

### Fichiers générés
1. **confusion_matrices.png** : 6 matrices de confusion (une par maladie)
2. **metrics_comparison.png** : Graphique en barres comparant les métriques

---

## 5. Fonction visualize_feature_importance()

### Signature
```python
def visualize_feature_importance(model, feature_columns, label_columns, save_path='model_visualizations')
```

### Description
Analyse et visualise l'importance de chaque feature (paramètre clinique) dans la prédiction de chaque maladie.

### Paramètres
- **model** : Modèle MultiOutputClassifier entraîné
- **feature_columns** (list) : Liste des 12 noms de features
- **label_columns** (list) : Liste des 6 noms de maladies
- **save_path** (str, optionnel) : Dossier de sauvegarde

### Retour
- Aucun (génère un fichier PNG et affiche dans le terminal)

### Étapes détaillées

#### Étape 5.1 : Création du dossier
```python
import os
os.makedirs(save_path, exist_ok=True)
```
- Assure que le dossier de destination existe

#### Étape 5.2 : Création de la figure
```python
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Importance des Features par Maladie', fontsize=16, fontweight='bold')
```
- Grille 2×3 pour 6 graphiques (un par maladie)
- Figure plus large (18×12) pour accommoder les noms de features

#### Étape 5.3 : Boucle sur chaque maladie
```python
for idx, (label, ax) in enumerate(zip(label_columns, axes.flat)):
```
- Itère sur chaque maladie et son subplot

#### Étape 5.4 : Extraction de l'importance des features
```python
importances = model.estimators_[idx].feature_importances_
```
- **model.estimators_[idx]** : Accède au Random Forest de la maladie idx
- **feature_importances_** : Attribut sklearn contenant l'importance de chaque feature
- Valeurs entre 0 et 1, somme = 1
- Plus élevé = plus important pour la décision

**Comment l'importance est calculée :**
- Basée sur la réduction d'impureté de Gini
- Mesure combien chaque feature contribue à la pureté des nœuds
- Moyennée sur tous les arbres de la forêt

#### Étape 5.5 : Tri par importance
```python
indices = np.argsort(importances)[::-1]
```
- **np.argsort()** : Retourne les indices qui trieraient le tableau
- **[::-1]** : Inverse l'ordre (décroissant)
- Résultat : indices des features du plus important au moins important

#### Étape 5.6 : Création du graphique en barres horizontales
```python
ax.barh(range(len(feature_columns)), importances[indices], color='#3498db', alpha=0.7)
```
- **barh()** : Barres horizontales
- **range(len(feature_columns))** : Positions Y (0 à 11)
- **importances[indices]** : Valeurs triées par importance décroissante
- **alpha=0.7** : Légère transparence

#### Étape 5.7 : Configuration des axes
```python
ax.set_yticks(range(len(feature_columns)))
ax.set_yticklabels([feature_columns[i] for i in indices], fontsize=9)
ax.set_xlabel('Importance', fontweight='bold')
ax.set_title(f'{label.capitalize()}', fontweight='bold')
ax.invert_yaxis()
```
- **set_yticks()** : Positions des ticks Y
- **set_yticklabels()** : Labels des features dans l'ordre trié
- **invert_yaxis()** : Inverse l'axe Y (plus important en haut)

#### Étape 5.8 : Sauvegarde du graphique
```python
plt.tight_layout()
plt.savefig(f'{save_path}/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### Étape 5.9 : Affichage dans le terminal (Top 5)
```python
print(f"\nImportance des Features (Top 5 par maladie):")

for idx, label in enumerate(label_columns):
    importances = model.estimators_[idx].feature_importances_
    feature_importance = sorted(zip(feature_columns, importances), 
                               key=lambda x: x[1], reverse=True)[:5]
    
    print(f"\n   {label.capitalize()}:")
    for feat, imp in feature_importance:
        print(f"      - {feat:<25} {imp:.4f}")
```
- Pour chaque maladie :
  - Extrait les importances
  - **zip(feature_columns, importances)** : Associe noms et valeurs
  - **sorted(..., key=lambda x: x[1])** : Trie par importance (x[1])
  - **[:5]** : Garde seulement les 5 premiers
  - Affiche le nom et la valeur d'importance

### Exemple de sortie terminal
```
Importance des Features (Top 5 par maladie):

   Diabete:
      - glycemie_jeun            0.3542
      - imc                      0.1823
      - age                      0.1245
      - cholesterol_total        0.0987
      - tension_systolique       0.0765

   Hypertension:
      - tension_systolique       0.4123
      - tension_diastolique      0.2876
      - age                      0.1432
      - imc                      0.0654
      - frequence_cardiaque      0.0412
...
```

### Interprétation
- Features avec haute importance sont les plus déterminantes
- Par exemple, glycémie_jeun très importante pour le diabète
- Aide à comprendre les décisions du modèle
- Peut identifier des features redondantes ou inutiles

---

## 6. Fonction predict_new_patient()

### Signature
```python
def predict_new_patient(model, feature_columns, label_columns)
```

### Description
Démontre l'utilisation du modèle pour prédire les maladies d'un nouveau patient avec ses paramètres cliniques.

### Paramètres
- **model** : Modèle MultiOutputClassifier entraîné
- **feature_columns** (list) : Liste des 12 noms de features
- **label_columns** (list) : Liste des 6 noms de maladies

### Retour
- Aucun (affiche les résultats dans le terminal)

### Étapes détaillées

#### Étape 6.1 : Définition du patient exemple
```python
new_patient = {
    'age': 58,
    'sexe': 0,  # Homme
    'glycemie_jeun': 165,  # Élevée
    'tension_systolique': 155,  # Élevée
    'tension_diastolique': 95,  # Élevée
    'frequence_cardiaque': 78,
    'imc': 29.5,  # Surpoids
    'hemoglobine': 14.2,
    'globules_blancs': 7.5,
    'cholesterol_total': 245,  # Élevé
    'creatinine': 1.0,
    'temperature': 36.8
}
```
- Dictionnaire contenant tous les paramètres requis
- Valeurs choisies pour montrer un cas avec diabète et hypertension probables
- Commentaires indiquent les valeurs anormales

#### Étape 6.2 : Affichage des paramètres du patient
```python
print("\nParametres du patient:")
print(f"   - Age: {new_patient['age']} ans")
print(f"   - Sexe: {'Homme' if new_patient['sexe']==0 else 'Femme'}")
# ... (affichage de tous les paramètres)
```
- Affiche chaque paramètre de manière lisible
- Conversion sexe : 0 → "Homme", 1 → "Femme"
- Format cohérent avec les unités

#### Étape 6.3 : Préparation des features
```python
X_new = np.array([[new_patient[col] for col in feature_columns]])
```
- **List comprehension** : Extrait les valeurs dans l'ordre correct
- **np.array()** : Convertit en array numpy
- **[[...]]** : Double crochets pour forme (1, 12) - 1 échantillon, 12 features
- L'ordre doit correspondre exactement à feature_columns

#### Étape 6.4 : Prédiction binaire
```python
prediction = model.predict(X_new)[0]
```
- **model.predict()** : Retourne les prédictions binaires (0 ou 1)
- Forme de sortie : (1, 6) - 1 échantillon, 6 maladies
- **[0]** : Extrait le premier (et seul) échantillon → array de forme (6,)

#### Étape 6.5 : Calcul des probabilités
```python
probas = []
for estimator in model.estimators_:
    proba = estimator.predict_proba(X_new)[0]
    if len(proba.shape) == 1:
        probas.append([1.0 - proba[0], proba[0]])
    else:
        probas.append(proba[1])
```
- Itère sur chaque estimateur (un par maladie)
- **predict_proba()** : Retourne les probabilités de chaque classe

**Gestion de deux cas :**

**Cas 1 : Forme 1D** (une seule classe dans l'entraînement)
```python
if len(proba.shape) == 1:
    probas.append([1.0 - proba[0], proba[0]])
```
- Si le modèle n'a vu qu'une classe, proba est 1D
- Calcule manuellement [P(classe_0), P(classe_1)]

**Cas 2 : Forme 2D** (deux classes dans l'entraînement)
```python
else:
    probas.append(proba[1])
```
- proba a forme (2,) : [P(classe_0), P(classe_1)]
- Extrait P(classe_1) - probabilité de maladie positive

#### Étape 6.6 : Affichage des résultats
```python
detected_diseases = []
for idx, (label, pred, prob) in enumerate(zip(label_columns, prediction, probas)):
    status = "DETECTE" if pred == 1 else "Non detecte"
    confidence = prob if isinstance(prob, float) else prob[1] if len(prob) > 1 else prob[0]
    print(f"   - {label.capitalize():<30} {status:<15} (confiance: {confidence:.2%})")
    
    if pred == 1:
        detected_diseases.append(label.capitalize())
```
- **zip()** : Associe nom, prédiction et probabilité
- **status** : Texte selon la prédiction binaire
- **confidence** : Gère différents formats de prob
  - Si float direct : utilise tel quel
  - Si list/array : extrait la probabilité de la classe positive
- **:.2%** : Formate en pourcentage avec 2 décimales
- Collecte les maladies détectées dans une liste

#### Étape 6.7 : Résumé final
```python
if detected_diseases:
    print(f"\nMaladies detectees: {', '.join(detected_diseases)}")
    print(f"Nombre total: {len(detected_diseases)}")
else:
    print(f"\nAucune maladie detectee - Patient sain")
```
- Affiche la liste des maladies détectées
- Compte total de maladies
- Ou indique que le patient est sain

### Exemple de sortie
```
==================================================================
         EXEMPLE DE PREDICTION - NOUVEAU PATIENT         
==================================================================

Parametres du patient:
   - Age: 58 ans
   - Sexe: Homme
   - Glycemie a jeun: 165 mg/dL
   - Tension: 155/95 mmHg
   ...

DIAGNOSTIC PREDIT:
   - Diabete                      DETECTE        (confiance: 94.32%)
   - Hypertension                 DETECTE        (confiance: 89.76%)
   - Anemie                       Non detecte    (confiance: 12.34%)
   - Maladie_cardiovasculaire     Non detecte    (confiance: 23.45%)
   - Infection                    Non detecte    (confiance: 5.67%)
   - Sain                         Non detecte    (confiance: 8.90%)

Maladies detectees: Diabete, Hypertension
Nombre total: 2
```

### Utilisation pratique
Cette fonction peut être adaptée pour :
- Interface utilisateur
- API REST
- Application médicale
- Système de screening

**Code pour intégration :**
```python
def predict_patient_diseases(model, patient_data):
    """
    Prédit les maladies pour un patient donné
    
    Args:
        model: Modèle entraîné
        patient_data: dict avec toutes les features
    
    Returns:
        dict: {maladie: (prediction, probabilité)}
    """
    X = np.array([[patient_data[col] for col in feature_columns]])
    predictions = model.predict(X)[0]
    
    results = {}
    for idx, label in enumerate(label_columns):
        results[label] = (predictions[idx], probas[idx])
    
    return results
```

---

## 7. Fonction main()

### Signature
```python
def main()
```

### Description
Fonction principale qui orchestre l'exécution complète du script : chargement des données, entraînement, évaluation, visualisation et sauvegarde.

### Paramètres
- Aucun

### Retour
- Aucun

### Étapes détaillées

#### Étape 7.1 : Affichage du header principal
```python
print("\n" + "=" * 70)
print(" ENTRAINEMENT DU MODELE RANDOM FOREST MULTI-LABEL ".center(70))
print(" Systeme de Diagnostic Multi-Maladies ".center(70))
print("=" * 70 + "\n")
```
- Affiche un en-tête stylisé
- **center(70)** : Centre le texte sur 70 caractères
- Double titre pour identifier clairement le programme

#### Étape 7.2 : Chargement du dataset
```python
df = load_dataset('medical_dataset_synthetic.csv')
```
- Appelle la fonction load_dataset()
- Charge le fichier CSV généré par generate_data.py
- Retourne un DataFrame ou None en cas d'erreur

#### Étape 7.3 : Vérification du chargement
```python
if df is None:
    return
```
- Vérifie si le chargement a réussi
- Si échec (None), termine l'exécution immédiatement
- Évite les erreurs en cascade

#### Étape 7.4 : Entraînement du modèle
```python
model, X_test, y_test, y_pred, feature_columns, label_columns = train_ml_model(df)
```
- Appelle la fonction train_ml_model()
- Effectue tout le pipeline d'entraînement et d'évaluation
- Récupère 6 valeurs de retour :
  - **model** : Modèle entraîné
  - **X_test** : Features de test (360 échantillons × 12 features)
  - **y_test** : Labels réels de test (360 × 6)
  - **y_pred** : Prédictions (360 × 6)
  - **feature_columns** : Liste des 12 noms de features
  - **label_columns** : Liste des 6 noms de maladies

#### Étape 7.5 : Visualisation des résultats du modèle
```python
visualize_model_results(y_test, y_pred, label_columns)
```
- Génère les matrices de confusion
- Crée le graphique de comparaison des métriques
- Sauvegarde dans 'model_visualizations/'

#### Étape 7.6 : Visualisation de l'importance des features
```python
visualize_feature_importance(model, feature_columns, label_columns)
```
- Analyse l'importance de chaque feature
- Génère le graphique d'importance
- Affiche le top 5 dans le terminal

#### Étape 7.7 : Exemple de prédiction
```python
predict_new_patient(model, feature_columns, label_columns)
```
- Démontre l'utilisation du modèle
- Prédit pour un patient exemple
- Affiche les résultats détaillés

#### Étape 7.8 : Sauvegarde du modèle
```python
model_file = 'random_forest_medical_model.pkl'
joblib.dump(model, model_file)
print(f"Modele sauvegarde: {model_file}")
```
- Définit le nom du fichier de sauvegarde
- **joblib.dump()** : Sérialise le modèle sur disque
  - Format binaire optimisé pour numpy
  - Plus efficace que pickle pour gros objets
  - Préserve tous les attributs du modèle
- Affiche confirmation

**Le fichier sauvegardé contient :**
- Les 6 Random Forest entraînés
- Tous les hyperparamètres
- Les arbres de décision complets
- Les importances de features

**Rechargement ultérieur :**
```python
model = joblib.load('random_forest_medical_model.pkl')
```

#### Étape 7.9 : Message de fin
```python
print("\n" + "=" * 70)
print(" PROCESSUS TERMINE AVEC SUCCES ".center(70))
print("=" * 70)
print(f"\nFichiers generes:")
print(f"   - {model_file} - Modele ML entraine")
print(f"   - model_visualizations/ - Graphiques d'evaluation")
print("\n")
```
- Affiche un message de succès
- Liste tous les fichiers générés
- Aide l'utilisateur à localiser les sorties

### Flux complet d'exécution

```
main()
  │
  ├─> load_dataset()
  │     └─> Retourne DataFrame
  │
  ├─> train_ml_model(df)
  │     ├─> Préparation des données
  │     ├─> Split train/test
  │     ├─> Configuration Random Forest
  │     ├─> Entraînement
  │     ├─> Prédictions
  │     ├─> Calcul des métriques
  │     └─> Retourne (model, X_test, y_test, y_pred, features, labels)
  │
  ├─> visualize_model_results(y_test, y_pred, labels)
  │     ├─> Matrices de confusion
  │     ├─> Graphique de comparaison
  │     └─> Sauvegarde PNG
  │
  ├─> visualize_feature_importance(model, features, labels)
  │     ├─> Extraction importances
  │     ├─> Graphique barres horizontales
  │     ├─> Sauvegarde PNG
  │     └─> Affichage terminal
  │
  ├─> predict_new_patient(model, features, labels)
  │     ├─> Définition patient
  │     ├─> Prédiction
  │     ├─> Calcul probabilités
  │     └─> Affichage résultats
  │
  └─> Sauvegarde modèle (joblib)
```

---

## 8. Flux d'exécution complet

### Vue d'ensemble du pipeline

```
[Début]
   ↓
[Chargement CSV] → medical_dataset_synthetic.csv (1800 lignes, 18 colonnes)
   ↓
[Extraction X, y] → X: (1800, 12), y: (1800, 6)
   ↓
[Split train/test] → Train: 1440, Test: 360
   ↓
[Configuration RF] → 200 arbres, depth=15, balanced
   ↓
[MultiOutputClassifier] → 6 RF indépendants
   ↓
[Entraînement] → Fit sur 1440 échantillons
   ↓
[Prédiction] → Predict sur 360 échantillons
   ↓
[Métriques globales] → Hamming Loss, Precision, Recall, F1
   ↓
[Métriques par maladie] → 6 séries de métriques
   ↓
[Matrices de confusion] → 6 matrices 2×2
   ↓
[Visualisations résultats] → 2 PNG sauvegardés
   ↓
[Importance features] → Analyse + PNG
   ↓
[Prédiction exemple] → Démo sur nouveau patient
   ↓
[Sauvegarde modèle] → random_forest_medical_model.pkl
   ↓
[Fin]
```

### Détails des transformations de données

#### 1. Données d'entrée (CSV)
```
Format: 1800 lignes × 18 colonnes
Colonnes:
  - 12 features (age, sexe, glycemie_jeun, ...)
  - 6 labels (diabete, hypertension, anemie, ...)
Types: int64, float64
```

#### 2. Extraction X et y
```python
X = df[feature_columns].values  # Shape: (1800, 12)
y = df[label_columns].values     # Shape: (1800, 6)
```

#### 3. Split train/test (80/20)
```python
X_train: (1440, 12)  # 80% pour entraînement
X_test:  (360, 12)   # 20% pour test
y_train: (1440, 6)
y_test:  (360, 6)
```

#### 4. Entraînement
```
MultiOutputClassifier contient 6 Random Forest:
  - RF_diabete              (200 arbres)
  - RF_hypertension         (200 arbres)
  - RF_anemie              (200 arbres)
  - RF_cardiovasculaire    (200 arbres)
  - RF_infection           (200 arbres)
  - RF_sain                (200 arbres)

Chaque RF apprend indépendamment sur y_train[:, i]
```

#### 5. Prédiction
```python
y_pred = model.predict(X_test)  # Shape: (360, 6)

Exemple de ligne:
y_pred[0] = [1, 1, 0, 0, 0, 0]  # Diabète + Hypertension
```

#### 6. Évaluation
```
Pour chaque maladie i:
  - Precision_i = TP_i / (TP_i + FP_i)
  - Recall_i = TP_i / (TP_i + FN_i)
  - F1_i = 2 × (P_i × R_i) / (P_i + R_i)

Macro-average:
  - Precision_macro = mean(Precision_i)
  - Recall_macro = mean(Recall_i)
  - F1_macro = mean(F1_i)

Hamming Loss:
  - HL = (Nombre total d'erreurs) / (360 × 6)
```

### Chronologie typique d'exécution

```
00:00 - Démarrage du script
00:01 - Dataset chargé (1800 échantillons)
00:02 - Données préparées et splittées
00:03 - Début entraînement Random Forest
01:30 - Entraînement terminé (6 modèles)
01:31 - Prédictions sur test set
01:32 - Calcul des métriques
01:33 - Affichage des résultats
01:34 - Génération des matrices de confusion
01:38 - Sauvegarde des graphiques
01:40 - Analyse importance des features
01:45 - Prédiction patient exemple
01:46 - Sauvegarde du modèle
01:47 - Fin du processus
```

### Fichiers et dossiers générés

```
Arborescence après exécution:

projet/
├── train_model.py                          (Script principal)
├── medical_dataset_synthetic.csv           (Input - généré par generate_data.py)
├── random_forest_medical_model.pkl         (Modèle sauvegardé - 50-100 MB)
└── model_visualizations/                   (Dossier de sortie)
    ├── confusion_matrices.png              (6 matrices 2×2)
    ├── metrics_comparison.png              (Barres groupées)
    └── feature_importance.png              (6 barres horizontales)
```

### Points clés de performance

**Temps d'exécution :**
- Chargement : ~1 seconde
- Entraînement : ~1-2 minutes (dépend du CPU)
- Évaluation : ~2-3 secondes
- Visualisations : ~5-10 secondes
- Total : ~2 minutes

**Utilisation mémoire :**
- Dataset : ~2 MB
- Modèle entraîné : ~50-100 MB (en mémoire)
- Graphiques : ~5 MB (sur disque)

**Précision attendue :**
- Hamming Loss : 0.02-0.05 (très bon)
- F1-Score macro : 0.90-0.95 (excellent)
- Variation selon les maladies

### Gestion des erreurs

Le script gère plusieurs scénarios d'erreur :

1. **Fichier CSV manquant**
```python
if df is None:
    return  # Arrêt propre du script
```

2. **Avertissements sklearn**
```python
warnings.filterwarnings('ignore')  # Supprime les warnings
```

3. **Division par zéro dans les métriques**
```python
zero_division=0  # Retourne 0 au lieu d'erreur
```

4. **Formats de probabilité variables**
```python
if len(proba.shape) == 1:
    # Gestion cas particulier
else:
    # Gestion cas normal
```

### Extensibilité

Le script peut être étendu pour :

1. **Hyperparameter tuning**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'estimator__n_estimators': [100, 200, 300],
    'estimator__max_depth': [10, 15, 20]
}
grid_search = GridSearchCV(model, param_grid, cv=5)
```

2. **Validation croisée**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
```

3. **Autres algorithmes**
```python
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

mlp = MLPClassifier(hidden_layer_sizes=(100, 50))
svm = SVC(kernel='rbf')
```

4. **Sauvegarde des métriques**
```python
import json

metrics_dict = {
    'hamming_loss': h_loss,
    'precision_macro': precision_macro,
    # ...
}
with open('metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)
```

---

## Conclusion

Le script `train_model.py` implémente un pipeline complet de machine learning pour la classification multi-label de maladies médicales. Il est structuré de manière modulaire avec des fonctions spécialisées, facilite la compréhension et l'extension, et produit des résultats reproductibles grâce à la graine aléatoire fixe.

Les principales forces du script :
- Pipeline clair et bien organisé
- Évaluation complète avec multiples métriques
- Visualisations informatives
- Gestion d'erreurs robuste
- Code réutilisable et extensible

Le script peut être utilisé tel quel pour l'éducation ou adapté pour des applications réelles avec des données médicales authentiques et une validation clinique appropriée.
