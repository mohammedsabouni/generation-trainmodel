"""
Entraînement et Évaluation du Modèle Random Forest Multi-Label
Projet: Système de Diagnostic Multi-Maladies
Auteur: Projet de Fin d'Année
Date: 2025

Ce script charge le dataset généré et entraîne un modèle Random Forest
multi-label pour la prédiction de maladies multiples.
"""

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

# Configuration pour reproductibilité
SEED = 42
np.random.seed(SEED)


# ==================== CHARGEMENT DES DONNÉES ====================

def load_dataset(filepath='medical_dataset_synthetic.csv'):
    """Charge le dataset depuis le fichier CSV"""
    print("=" * 70)
    print(" CHARGEMENT DES DONNEES ".center(70))
    print("=" * 70)
    
    try:
        df = pd.read_csv(filepath)
        print(f"\nDataset charge avec succes: {filepath}")
        print(f"   - Nombre d'echantillons: {len(df)}")
        print(f"   - Nombre de colonnes: {len(df.columns)}")
        print("\n" + "=" * 70 + "\n")
        return df
    except FileNotFoundError:
        print(f"\nERREUR: Le fichier '{filepath}' n'a pas ete trouve.")
        print("Veuillez d'abord executer 'generate_data.py' pour generer le dataset.")
        return None


# ==================== ENTRAÎNEMENT DU MODÈLE ====================

def train_ml_model(df):
    """Entraîne un Random Forest multi-label et évalue les performances"""
    print("=" * 70)
    print(" ENTRAINEMENT DU MODELE RANDOM FOREST MULTI-LABEL ".center(70))
    print("=" * 70)
    
    # Préparation des données
    feature_columns = ['age', 'sexe', 'glycemie_jeun', 'tension_systolique',
                       'tension_diastolique', 'frequence_cardiaque', 'imc',
                       'hemoglobine', 'globules_blancs', 'cholesterol_total',
                       'creatinine', 'temperature']
    
    label_columns = ['diabete', 'hypertension', 'anemie',
                     'maladie_cardiovasculaire', 'infection', 'sain']
    
    X = df[feature_columns].values
    y = df[label_columns].values
    
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y[:, 0]
    )
    
    print(f"\nDonnees preparees:")
    print(f"   - Features: {X_train.shape[1]}")
    print(f"   - Labels: {y_train.shape[1]}")
    print(f"   - Echantillons d'entrainement: {X_train.shape[0]}")
    print(f"   - Echantillons de test: {X_test.shape[0]}")
    
    # Entraînement
    print(f"\nEntrainement du modele Random Forest...")
    
    # Configuration: Random Forest avec MultiOutputClassifier pour multi-label
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
        class_weight='balanced'  # Gérer le déséquilibre des classes
    )
    
    model = MultiOutputClassifier(rf)
    model.fit(X_train, y_train)
    
    print(f"   Modele entraine avec succes")
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Évaluation
    print(f"\n" + "=" * 70)
    print(" METRIQUES DE PERFORMANCE ".center(70))
    print("=" * 70)
    
    # Hamming Loss (plus bas = meilleur, 0 = parfait)
    h_loss = hamming_loss(y_test, y_pred)
    print(f"\nHamming Loss: {h_loss:.4f}")
    print(f"(Proportion d'erreurs par label, 0 = parfait)")
    
    # Métriques globales
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"\nMetriques Macro-Averaged (moyenne sur toutes les maladies):")
    print(f"   - Precision: {precision_macro:.4f}")
    print(f"   - Recall:    {recall_macro:.4f}")
    print(f"   - F1-Score:  {f1_macro:.4f}")
    
    # Métriques par maladie
    print(f"\nMetriques par maladie:")
    print(f"   {'-' * 66}")
    print(f"   {'Maladie':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"   {'-' * 66}")
    
    for idx, label in enumerate(label_columns):
        prec = precision_score(y_test[:, idx], y_pred[:, idx], zero_division=0)
        rec = recall_score(y_test[:, idx], y_pred[:, idx], zero_division=0)
        f1 = f1_score(y_test[:, idx], y_pred[:, idx], zero_division=0)
        print(f"   {label.capitalize():<30} {prec:>7.4f}      {rec:>7.4f}      {f1:>7.4f}")
    
    print(f"   {'-' * 66}")
    
    # Matrices de confusion par maladie
    print(f"\nMatrices de Confusion par Maladie:")
    
    for idx, label in enumerate(label_columns):
        cm = confusion_matrix(y_test[:, idx], y_pred[:, idx])
        
        # Afficher dans le terminal
        print(f"\n   {label.capitalize()}:")
        print(f"      TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
        print(f"      FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    print("\n" + "=" * 70 + "\n")
    
    return model, X_test, y_test, y_pred, feature_columns, label_columns


# ==================== VISUALISATION DES RÉSULTATS ====================

def visualize_model_results(y_test, y_pred, label_columns, save_path='model_visualizations'):
    """Crée des visualisations des résultats du modèle"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("Generation des visualisations du modele...")
    
    # Configuration du style
    sns.set_style("whitegrid")
    
    # 1. Matrices de confusion
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Matrices de Confusion - Classification Multi-Label', 
                 fontsize=16, fontweight='bold')
    
    for idx, (label, ax) in enumerate(zip(label_columns, axes.flat)):
        cm = confusion_matrix(y_test[:, idx], y_pred[:, idx])
        
        # Créer la heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Non', 'Oui'],
                    yticklabels=['Non', 'Oui'],
                    cbar=False)
        
        ax.set_title(f'{label.capitalize()}', fontweight='bold')
        ax.set_ylabel('Reel')
        ax.set_xlabel('Predit')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"   Sauvegarde: {save_path}/confusion_matrices.png")
    plt.close()
    
    # 2. Comparaison des métriques par maladie
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = {
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    for idx in range(len(label_columns)):
        metrics['Precision'].append(precision_score(y_test[:, idx], y_pred[:, idx], zero_division=0))
        metrics['Recall'].append(recall_score(y_test[:, idx], y_pred[:, idx], zero_division=0))
        metrics['F1-Score'].append(f1_score(y_test[:, idx], y_pred[:, idx], zero_division=0))
    
    x = np.arange(len(label_columns))
    width = 0.25
    
    ax.bar(x - width, metrics['Precision'], width, label='Precision', color='#3498db')
    ax.bar(x, metrics['Recall'], width, label='Recall', color='#2ecc71')
    ax.bar(x + width, metrics['F1-Score'], width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Maladies', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Comparaison des Metriques par Maladie', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([label.capitalize() for label in label_columns], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   Sauvegarde: {save_path}/metrics_comparison.png")
    plt.close()
    
    print(f"\nVisualisations du modele sauvegardees dans '{save_path}/'")


def visualize_feature_importance(model, feature_columns, label_columns, save_path='model_visualizations'):
    """Visualise l'importance des features pour chaque maladie"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\nGeneration des graphiques d'importance des features...")
    
    # Créer une figure avec subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Importance des Features par Maladie', fontsize=16, fontweight='bold')
    
    for idx, (label, ax) in enumerate(zip(label_columns, axes.flat)):
        importances = model.estimators_[idx].feature_importances_
        
        # Trier par importance
        indices = np.argsort(importances)[::-1]
        
        # Plot
        ax.barh(range(len(feature_columns)), importances[indices], color='#3498db', alpha=0.7)
        ax.set_yticks(range(len(feature_columns)))
        ax.set_yticklabels([feature_columns[i] for i in indices], fontsize=9)
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title(f'{label.capitalize()}', fontweight='bold')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"   Sauvegarde: {save_path}/feature_importance.png")
    plt.close()
    
    # Afficher aussi dans le terminal
    print(f"\nImportance des Features (Top 5 par maladie):")
    
    for idx, label in enumerate(label_columns):
        importances = model.estimators_[idx].feature_importances_
        feature_importance = sorted(zip(feature_columns, importances), 
                                   key=lambda x: x[1], reverse=True)[:5]
        
        print(f"\n   {label.capitalize()}:")
        for feat, imp in feature_importance:
            print(f"      - {feat:<25} {imp:.4f}")


# ==================== PRÉDICTION SUR NOUVEAU PATIENT ====================

def predict_new_patient(model, feature_columns, label_columns):
    """Exemple de prédiction sur un nouveau patient"""
    print("\n" + "=" * 70)
    print(" EXEMPLE DE PREDICTION - NOUVEAU PATIENT ".center(70))
    print("=" * 70)
    
    # Patient exemple: Homme de 58 ans avec suspicion de diabète et hypertension
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
    
    print("\nParametres du patient:")
    print(f"   - Age: {new_patient['age']} ans")
    print(f"   - Sexe: {'Homme' if new_patient['sexe']==0 else 'Femme'}")
    print(f"   - Glycemie a jeun: {new_patient['glycemie_jeun']} mg/dL")
    print(f"   - Tension: {new_patient['tension_systolique']}/{new_patient['tension_diastolique']} mmHg")
    print(f"   - Frequence cardiaque: {new_patient['frequence_cardiaque']} bpm")
    print(f"   - IMC: {new_patient['imc']} kg/m2")
    print(f"   - Hemoglobine: {new_patient['hemoglobine']} g/dL")
    print(f"   - Globules blancs: {new_patient['globules_blancs']} x10^3/uL")
    print(f"   - Cholesterol total: {new_patient['cholesterol_total']} mg/dL")
    print(f"   - Creatinine: {new_patient['creatinine']} mg/dL")
    print(f"   - Temperature: {new_patient['temperature']} C")
    
    # Préparer les features
    X_new = np.array([[new_patient[col] for col in feature_columns]])
    
    # Prédiction
    prediction = model.predict(X_new)[0]
    
    # Obtenir les probabilités (moyenne des probabilités de tous les arbres)
    probas = []
    for estimator in model.estimators_:
        proba = estimator.predict_proba(X_new)[0]
        # Gérer le cas où une classe n'a qu'une seule valeur
        if len(proba.shape) == 1:
            probas.append([1.0 - proba[0], proba[0]])
        else:
            probas.append(proba[1])  # Probabilité de la classe positive
    
    print(f"\nDIAGNOSTIC PREDIT:")
    
    detected_diseases = []
    for idx, (label, pred, prob) in enumerate(zip(label_columns, prediction, probas)):
        status = "DETECTE" if pred == 1 else "Non detecte"
        confidence = prob if isinstance(prob, float) else prob[1] if len(prob) > 1 else prob[0]
        print(f"   - {label.capitalize():<30} {status:<15} (confiance: {confidence:.2%})")
        
        if pred == 1:
            detected_diseases.append(label.capitalize())
    
    if detected_diseases:
        print(f"\nMaladies detectees: {', '.join(detected_diseases)}")
        print(f"Nombre total: {len(detected_diseases)}")
    else:
        print(f"\nAucune maladie detectee - Patient sain")
    
    print("\n" + "=" * 70 + "\n")


# ==================== FONCTION PRINCIPALE ====================

def main():
    """Fonction principale du programme"""
    print("\n" + "=" * 70)
    print(" ENTRAINEMENT DU MODELE RANDOM FOREST MULTI-LABEL ".center(70))
    print(" Systeme de Diagnostic Multi-Maladies ".center(70))
    print("=" * 70 + "\n")
    
    # 1. Charger le dataset
    df = load_dataset('medical_dataset_synthetic.csv')
    
    if df is None:
        return
    
    # 2. Entraîner le modèle
    model, X_test, y_test, y_pred, feature_columns, label_columns = train_ml_model(df)
    
    # 3. Visualiser les résultats
    visualize_model_results(y_test, y_pred, label_columns)
    
    # 4. Visualiser l'importance des features
    visualize_feature_importance(model, feature_columns, label_columns)
    
    # 5. Exemple de prédiction
    predict_new_patient(model, feature_columns, label_columns)
    
    # 6. Sauvegarder le modèle
    model_file = 'random_forest_medical_model.pkl'
    joblib.dump(model, model_file)
    print(f"Modele sauvegarde: {model_file}")
    
    print("\n" + "=" * 70)
    print(" PROCESSUS TERMINE AVEC SUCCES ".center(70))
    print("=" * 70)
    print(f"\nFichiers generes:")
    print(f"   - {model_file} - Modele ML entraine")
    print(f"   - model_visualizations/ - Graphiques d'evaluation")
    print("\n")


if __name__ == "__main__":
    main()
