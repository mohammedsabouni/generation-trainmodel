"""
Générateur de Données Médicales Synthétiques Multi-Label
Projet: Système de Diagnostic Multi-Maladies avec Random Forest
Auteur: Projet de Fin d'Année
Date: 2025

Ce script génère un dataset synthétique réaliste de patients avec:
- Features cliniques basées sur des ranges médicaux documentés
- Labels multi-label (plusieurs maladies simultanées possibles)
- Corrélations pathologiques réalistes
- Comorbidités fréquentes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration pour reproductibilité
SEED = 42
np.random.seed(SEED)

# ==================== CONFIGURATION ====================
N_SAMPLES = 1800  # Nombre d'échantillons à générer

# Ranges médicaux normaux (source: standards cliniques internationaux)
NORMAL_RANGES = {
    'glycemie_jeun': (70, 100),      # mg/dL
    'tension_systolique': (90, 120),  # mmHg
    'tension_diastolique': (60, 80),  # mmHg
    'frequence_cardiaque': (60, 100), # bpm
    'imc': (18.5, 24.9),              # kg/m²
    'hemoglobine_h': (13, 17),        # g/dL (hommes)
    'hemoglobine_f': (12, 15),        # g/dL (femmes)
    'globules_blancs': (4, 11),       # ×10³/µL
    'cholesterol_total': (150, 200),  # mg/dL
    'creatinine': (0.6, 1.2),         # mg/dL
    'temperature': (36.5, 37.5)       # °C
}

# Distribution cible des maladies
TARGET_DISTRIBUTION = {
    'sain': 0.30,
    'diabete': 0.20,
    'hypertension': 0.25,
    'anemie': 0.15,
    'maladie_cardiovasculaire': 0.18,
    'infection': 0.12
}


# ==================== FONCTIONS DE GÉNÉRATION ====================

def generate_healthy_patient(age):
    """Génère les paramètres d'un patient sain"""
    sexe = np.random.choice([0, 1])  # 0=M, 1=F
    
    # Variation normale autour des valeurs saines
    glycemie = np.random.uniform(75, 100)
    tension_sys = np.random.uniform(100, 125)
    tension_dia = np.random.uniform(65, 82)
    freq_cardiaque = np.random.uniform(60, 95)
    imc = np.random.uniform(19, 24.5)
    
    # Hémoglobine selon le sexe
    if sexe == 0:  # Homme
        hemoglobine = np.random.uniform(13.5, 16.5)
    else:  # Femme
        hemoglobine = np.random.uniform(12.5, 14.8)
    
    globules_blancs = np.random.uniform(4.5, 10)
    cholesterol = np.random.uniform(160, 195)
    creatinine = np.random.uniform(0.7, 1.1)
    temperature = np.random.uniform(36.6, 37.3)
    
    return {
        'age': age,
        'sexe': sexe,
        'glycemie_jeun': glycemie,
        'tension_systolique': tension_sys,
        'tension_diastolique': tension_dia,
        'frequence_cardiaque': freq_cardiaque,
        'imc': imc,
        'hemoglobine': hemoglobine,
        'globules_blancs': globules_blancs,
        'cholesterol_total': cholesterol,
        'creatinine': creatinine,
        'temperature': temperature,
        'diabete': 0,
        'hypertension': 0,
        'anemie': 0,
        'maladie_cardiovasculaire': 0,
        'infection': 0,
        'sain': 1
    }


def generate_diabetic_patient(age, with_comorbidity=False):
    """Génère un patient diabétique (Type 2)"""
    sexe = np.random.choice([0, 1])
    
    # Diabète: glycémie élevée, souvent IMC élevé
    glycemie = np.random.uniform(130, 280)  # Hyperglycémie
    imc = np.random.uniform(25, 38)  # Souvent en surpoids/obèse
    
    # Autres paramètres avec légère variation
    tension_sys = np.random.uniform(110, 145)
    tension_dia = np.random.uniform(70, 92)
    freq_cardiaque = np.random.uniform(65, 100)
    
    if sexe == 0:
        hemoglobine = np.random.uniform(13, 16)
    else:
        hemoglobine = np.random.uniform(12, 14.5)
    
    globules_blancs = np.random.uniform(5, 10.5)
    cholesterol = np.random.uniform(180, 250)  # Souvent élevé
    creatinine = np.random.uniform(0.8, 1.4)  # Peut être affecté
    temperature = np.random.uniform(36.5, 37.4)
    
    labels = {
        'diabete': 1,
        'hypertension': 0,
        'anemie': 0,
        'maladie_cardiovasculaire': 0,
        'infection': 0,
        'sain': 0
    }
    
    # Comorbidité: Diabète + Hypertension (30-40% des cas)
    if with_comorbidity and np.random.random() < 0.35:
        labels['hypertension'] = 1
        tension_sys = np.random.uniform(145, 180)
        tension_dia = np.random.uniform(92, 110)
    
    # Comorbidité: Diabète + Cardiovasculaire (si âge >55)
    if with_comorbidity and age > 55 and np.random.random() < 0.25:
        labels['maladie_cardiovasculaire'] = 1
        cholesterol = np.random.uniform(240, 300)
    
    return {
        'age': age,
        'sexe': sexe,
        'glycemie_jeun': glycemie,
        'tension_systolique': tension_sys,
        'tension_diastolique': tension_dia,
        'frequence_cardiaque': freq_cardiaque,
        'imc': imc,
        'hemoglobine': hemoglobine,
        'globules_blancs': globules_blancs,
        'cholesterol_total': cholesterol,
        'creatinine': creatinine,
        'temperature': temperature,
        **labels
    }


def generate_hypertensive_patient(age, with_comorbidity=False):
    """Génère un patient hypertendu"""
    sexe = np.random.choice([0, 1])
    
    # Hypertension: tension élevée
    tension_sys = np.random.uniform(145, 190)
    tension_dia = np.random.uniform(92, 115)
    
    # Corrélation avec âge et IMC
    if age > 60:
        imc = np.random.uniform(24, 33)
    else:
        imc = np.random.uniform(22, 30)
    
    glycemie = np.random.uniform(80, 115)
    freq_cardiaque = np.random.uniform(65, 105)
    
    if sexe == 0:
        hemoglobine = np.random.uniform(13, 16.5)
    else:
        hemoglobine = np.random.uniform(12, 15)
    
    globules_blancs = np.random.uniform(4.5, 10)
    cholesterol = np.random.uniform(170, 240)
    creatinine = np.random.uniform(0.8, 1.3)
    temperature = np.random.uniform(36.5, 37.4)
    
    labels = {
        'diabete': 0,
        'hypertension': 1,
        'anemie': 0,
        'maladie_cardiovasculaire': 0,
        'infection': 0,
        'sain': 0
    }
    
    # Comorbidité: Hypertension + Cardiovasculaire (forte corrélation)
    if with_comorbidity and np.random.random() < 0.40:
        labels['maladie_cardiovasculaire'] = 1
        cholesterol = np.random.uniform(240, 310)
        if age < 50:
            age = np.random.uniform(50, 75)  # Ajuster l'âge
    
    return {
        'age': age,
        'sexe': sexe,
        'glycemie_jeun': glycemie,
        'tension_systolique': tension_sys,
        'tension_diastolique': tension_dia,
        'frequence_cardiaque': freq_cardiaque,
        'imc': imc,
        'hemoglobine': hemoglobine,
        'globules_blancs': globules_blancs,
        'cholesterol_total': cholesterol,
        'creatinine': creatinine,
        'temperature': temperature,
        **labels
    }


def generate_anemic_patient(age):
    """Génère un patient anémique"""
    sexe = np.random.choice([0, 1])
    
    # Anémie: hémoglobine basse
    if sexe == 0:  # Homme
        hemoglobine = np.random.uniform(8, 12.5)
    else:  # Femme
        hemoglobine = np.random.uniform(7.5, 11.5)
    
    glycemie = np.random.uniform(75, 110)
    tension_sys = np.random.uniform(95, 130)
    tension_dia = np.random.uniform(60, 85)
    freq_cardiaque = np.random.uniform(70, 110)  # Peut être élevée (compensation)
    imc = np.random.uniform(18, 27)
    globules_blancs = np.random.uniform(4, 9)
    cholesterol = np.random.uniform(150, 210)
    creatinine = np.random.uniform(0.6, 1.2)
    temperature = np.random.uniform(36.4, 37.3)
    
    return {
        'age': age,
        'sexe': sexe,
        'glycemie_jeun': glycemie,
        'tension_systolique': tension_sys,
        'tension_diastolique': tension_dia,
        'frequence_cardiaque': freq_cardiaque,
        'imc': imc,
        'hemoglobine': hemoglobine,
        'globules_blancs': globules_blancs,
        'cholesterol_total': cholesterol,
        'creatinine': creatinine,
        'temperature': temperature,
        'diabete': 0,
        'hypertension': 0,
        'anemie': 1,
        'maladie_cardiovasculaire': 0,
        'infection': 0,
        'sain': 0
    }


def generate_cardiovascular_patient(age):
    """Génère un patient avec maladie cardiovasculaire"""
    sexe = np.random.choice([0, 1])
    
    # Cardiovasculaire: cholestérol élevé, tension souvent élevée, âge >50
    if age < 50:
        age = np.random.uniform(50, 80)
    
    cholesterol = np.random.uniform(245, 330)
    tension_sys = np.random.uniform(130, 170)
    tension_dia = np.random.uniform(85, 105)
    
    glycemie = np.random.uniform(85, 135)
    freq_cardiaque = np.random.uniform(60, 100)
    imc = np.random.uniform(24, 34)
    
    if sexe == 0:
        hemoglobine = np.random.uniform(12.5, 16)
    else:
        hemoglobine = np.random.uniform(11.5, 14.5)
    
    globules_blancs = np.random.uniform(5, 10.5)
    creatinine = np.random.uniform(0.8, 1.5)
    temperature = np.random.uniform(36.5, 37.4)
    
    return {
        'age': age,
        'sexe': sexe,
        'glycemie_jeun': glycemie,
        'tension_systolique': tension_sys,
        'tension_diastolique': tension_dia,
        'frequence_cardiaque': freq_cardiaque,
        'imc': imc,
        'hemoglobine': hemoglobine,
        'globules_blancs': globules_blancs,
        'cholesterol_total': cholesterol,
        'creatinine': creatinine,
        'temperature': temperature,
        'diabete': 0,
        'hypertension': 0,
        'anemie': 0,
        'maladie_cardiovasculaire': 1,
        'infection': 0,
        'sain': 0
    }


def generate_infected_patient(age):
    """Génère un patient avec infection/inflammation"""
    sexe = np.random.choice([0, 1])
    
    # Infection: globules blancs élevés, température élevée
    globules_blancs = np.random.uniform(11.5, 22)
    temperature = np.random.uniform(37.8, 39.5)
    
    glycemie = np.random.uniform(80, 130)  # Peut être légèrement élevée (stress)
    tension_sys = np.random.uniform(100, 140)
    tension_dia = np.random.uniform(65, 90)
    freq_cardiaque = np.random.uniform(75, 120)  # Tachycardie possible
    imc = np.random.uniform(19, 30)
    
    if sexe == 0:
        hemoglobine = np.random.uniform(12, 16)
    else:
        hemoglobine = np.random.uniform(11, 14.5)
    
    cholesterol = np.random.uniform(150, 220)
    creatinine = np.random.uniform(0.7, 1.3)
    
    return {
        'age': age,
        'sexe': sexe,
        'glycemie_jeun': glycemie,
        'tension_systolique': tension_sys,
        'tension_diastolique': tension_dia,
        'frequence_cardiaque': freq_cardiaque,
        'imc': imc,
        'hemoglobine': hemoglobine,
        'globules_blancs': globules_blancs,
        'cholesterol_total': cholesterol,
        'creatinine': creatinine,
        'temperature': temperature,
        'diabete': 0,
        'hypertension': 0,
        'anemie': 0,
        'maladie_cardiovasculaire': 0,
        'infection': 1,
        'sain': 0
    }


def generate_dataset(n_samples=1800):
    """
    Génère le dataset complet avec distribution réaliste des maladies
    et comorbidités
    """
    print(f"Generation de {n_samples} echantillons de patients...")
    
    data = []
    
    # Calculer le nombre d'échantillons par catégorie
    n_sain = int(n_samples * TARGET_DISTRIBUTION['sain'])
    n_diabete = int(n_samples * TARGET_DISTRIBUTION['diabete'])
    n_hypertension = int(n_samples * TARGET_DISTRIBUTION['hypertension'])
    n_anemie = int(n_samples * TARGET_DISTRIBUTION['anemie'])
    n_cardiovasculaire = int(n_samples * TARGET_DISTRIBUTION['maladie_cardiovasculaire'])
    n_infection = int(n_samples * TARGET_DISTRIBUTION['infection'])
    
    # Générer patients sains
    print(f"   Generation de {n_sain} patients sains...")
    for _ in range(n_sain):
        age = np.random.uniform(20, 75)
        data.append(generate_healthy_patient(age))
    
    # Générer patients diabétiques (avec possibilité de comorbidités)
    print(f"   Generation de {n_diabete} patients diabetiques...")
    for _ in range(n_diabete):
        age = np.random.uniform(35, 80)  # Diabète Type 2 plus fréquent >35 ans
        with_comorbidity = np.random.random() < 0.30  # 30% avec comorbidités
        data.append(generate_diabetic_patient(age, with_comorbidity))
    
    # Générer patients hypertendus (avec possibilité de comorbidités)
    print(f"   Generation de {n_hypertension} patients hypertendus...")
    for _ in range(n_hypertension):
        age = np.random.uniform(40, 80)  # Plus fréquent avec l'âge
        with_comorbidity = np.random.random() < 0.35  # 35% avec comorbidités
        data.append(generate_hypertensive_patient(age, with_comorbidity))
    
    # Générer patients anémiques
    print(f"   Generation de {n_anemie} patients anemiques...")
    for _ in range(n_anemie):
        age = np.random.uniform(20, 75)
        data.append(generate_anemic_patient(age))
    
    # Générer patients cardiovasculaires
    print(f"   Generation de {n_cardiovasculaire} patients cardiovasculaires...")
    for _ in range(n_cardiovasculaire):
        age = np.random.uniform(50, 80)
        data.append(generate_cardiovascular_patient(age))
    
    # Générer patients avec infection
    print(f"   Generation de {n_infection} patients avec infection...")
    for _ in range(n_infection):
        age = np.random.uniform(20, 75)
        data.append(generate_infected_patient(age))
    
    # Convertir en DataFrame
    df = pd.DataFrame(data)
    
    # Mélanger les données
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    # Arrondir les valeurs pour plus de réalisme
    df['age'] = df['age'].round(0).astype(int)
    df['glycemie_jeun'] = df['glycemie_jeun'].round(1)
    df['tension_systolique'] = df['tension_systolique'].round(0).astype(int)
    df['tension_diastolique'] = df['tension_diastolique'].round(0).astype(int)
    df['frequence_cardiaque'] = df['frequence_cardiaque'].round(0).astype(int)
    df['imc'] = df['imc'].round(1)
    df['hemoglobine'] = df['hemoglobine'].round(1)
    df['globules_blancs'] = df['globules_blancs'].round(1)
    df['cholesterol_total'] = df['cholesterol_total'].round(0).astype(int)
    df['creatinine'] = df['creatinine'].round(2)
    df['temperature'] = df['temperature'].round(1)
    
    print(f"\nDataset genere avec succes: {len(df)} echantillons\n")
    
    return df


# ==================== ANALYSE ET VISUALISATION ====================

def analyze_dataset(df):
    """Analyse statistique complète du dataset"""
    print("=" * 70)
    print(" ANALYSE STATISTIQUE DU DATASET".center(70))
    print("=" * 70)
    
    # Informations générales
    print(f"\nInformations generales:")
    print(f"   - Nombre total d'echantillons: {len(df)}")
    print(f"   - Nombre de features: {len(df.columns) - 6}")  # -6 labels
    print(f"   - Nombre de labels: 6")
    print(f"   - Valeurs manquantes: {df.isnull().sum().sum()}")
    
    # Distribution des maladies
    print(f"\nDistribution des maladies:")
    label_columns = ['diabete', 'hypertension', 'anemie', 
                     'maladie_cardiovasculaire', 'infection', 'sain']
    
    for label in label_columns:
        count = df[label].sum()
        percentage = (count / len(df)) * 100
        print(f"   - {label.capitalize():30s}: {count:4d} cas ({percentage:5.1f}%)")
    
    # Patients avec comorbidités
    df['n_maladies'] = df[label_columns[:-1]].sum(axis=1)  # Exclure 'sain'
    n_comorbidites = (df['n_maladies'] >= 2).sum()
    pct_comorbidites = (n_comorbidites / len(df)) * 100
    
    print(f"\nComorbidites:")
    print(f"   - Patients avec 2+ maladies: {n_comorbidites} ({pct_comorbidites:.1f}%)")
    print(f"   - Distribution:")
    for n in range(5):
        count = (df['n_maladies'] == n).sum()
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"      * {n} maladie(s): {count:4d} patients ({pct:5.1f}%)")
    
    # Statistiques descriptives des features
    print(f"\nStatistiques descriptives des features cliniques:")
    feature_columns = ['age', 'glycemie_jeun', 'tension_systolique', 
                       'tension_diastolique', 'frequence_cardiaque', 'imc',
                       'hemoglobine', 'globules_blancs', 'cholesterol_total',
                       'creatinine', 'temperature']
    
    stats_df = df[feature_columns].describe().T[['mean', 'std', 'min', 'max']]
    print(stats_df.to_string())
    
    # Distribution par sexe
    print(f"\nDistribution par sexe:")
    sexe_counts = df['sexe'].value_counts()
    print(f"   - Hommes (0): {sexe_counts.get(0, 0)} ({sexe_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"   - Femmes (1): {sexe_counts.get(1, 0)} ({sexe_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    print("\n" + "=" * 70 + "\n")


def visualize_dataset(df, save_path='visualizations'):
    """Crée des visualisations du dataset"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("Generation des visualisations...")
    
    # Configuration du style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Distribution des maladies
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distribution des Maladies dans le Dataset', fontsize=16, fontweight='bold')
    
    label_columns = ['diabete', 'hypertension', 'anemie', 
                     'maladie_cardiovasculaire', 'infection', 'sain']
    label_names = ['Diabete', 'Hypertension', 'Anemie', 
                   'Maladie\nCardiovasculaire', 'Infection', 'Sain']
    
    for idx, (label, name) in enumerate(zip(label_columns, label_names)):
        ax = axes[idx // 3, idx % 3]
        counts = df[label].value_counts()
        colors = ['#e74c3c', '#2ecc71']
        ax.bar(['Non', 'Oui'], [counts.get(0, 0), counts.get(1, 0)], color=colors)
        ax.set_title(name, fontweight='bold')
        ax.set_ylabel('Nombre de patients')
        
        # Ajouter les pourcentages
        total = len(df)
        for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
            pct = (v / total) * 100
            ax.text(i, v + 20, f'{v}\n({pct:.1f}%)', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/distribution_maladies.png', dpi=300, bbox_inches='tight')
    print(f"   Sauvegarde: {save_path}/distribution_maladies.png")
    plt.close()
    
    # 2. Distribution des features cliniques
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Distribution des Parametres Cliniques', fontsize=16, fontweight='bold')
    
    feature_columns = ['age', 'glycemie_jeun', 'tension_systolique', 
                       'tension_diastolique', 'frequence_cardiaque', 'imc',
                       'hemoglobine', 'globules_blancs', 'cholesterol_total',
                       'creatinine', 'temperature', 'sexe']
    
    feature_names = ['Age (annees)', 'Glycemie (mg/dL)', 'Tension Systolique (mmHg)',
                     'Tension Diastolique (mmHg)', 'Frequence Cardiaque (bpm)', 'IMC (kg/m2)',
                     'Hemoglobine (g/dL)', 'Globules Blancs (x10^3/uL)', 'Cholesterol (mg/dL)',
                     'Creatinine (mg/dL)', 'Temperature (C)', 'Sexe']
    
    for idx, (col, name) in enumerate(zip(feature_columns, feature_names)):
        ax = axes[idx // 4, idx % 4]
        if col == 'sexe':
            counts = df[col].value_counts()
            ax.bar(['Homme', 'Femme'], [counts.get(0, 0), counts.get(1, 0)], 
                   color=['#3498db', '#e91e63'])
        else:
            ax.hist(df[col], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        ax.set_title(name, fontweight='bold', fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('Frequence')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/distribution_features.png', dpi=300, bbox_inches='tight')
    print(f"   Sauvegarde: {save_path}/distribution_features.png")
    plt.close()
    
    # 3. Matrice de corrélation
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Sélectionner toutes les colonnes numériques
    corr_data = df.drop(columns=['n_maladies'], errors='ignore')
    correlation_matrix = corr_data.corr()
    
    # Créer la heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax)
    ax.set_title('Matrice de Correlation - Features et Labels', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"   Sauvegarde: {save_path}/correlation_matrix.png")
    plt.close()
    
    # 4. Comorbidités
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_temp = df.copy()
    label_columns = ['diabete', 'hypertension', 'anemie', 
                     'maladie_cardiovasculaire', 'infection', 'sain']
    df_temp['n_maladies'] = df_temp[label_columns[:-1]].sum(axis=1)
    comorbidity_counts = df_temp['n_maladies'].value_counts().sort_index()
    
    colors_comorb = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    bars = ax.bar(comorbidity_counts.index, comorbidity_counts.values, 
                  color=colors_comorb[:len(comorbidity_counts)])
    
    ax.set_xlabel('Nombre de maladies simultanees', fontweight='bold')
    ax.set_ylabel('Nombre de patients', fontweight='bold')
    ax.set_title('Distribution des Comorbidites', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/comorbidites.png', dpi=300, bbox_inches='tight')
    print(f"   Sauvegarde: {save_path}/comorbidites.png")
    plt.close()
    
    print(f"\nToutes les visualisations ont ete sauvegardees dans '{save_path}/'")


# ==================== FONCTION PRINCIPALE ====================

def main():
    """Fonction principale du programme"""
    print("\n" + "=" * 70)
    print(" GENERATEUR DE DONNEES MEDICALES SYNTHETIQUES ".center(70))
    print(" Systeme de Diagnostic Multi-Label ".center(70))
    print("=" * 70 + "\n")
    
    # 1. Générer le dataset
    df = generate_dataset(N_SAMPLES)
    
    # 2. Sauvegarder en CSV
    output_file = 'medical_dataset_synthetic.csv'
    df.to_csv(output_file, index=False)
    print(f"Dataset sauvegarde: {output_file}")
    
    # 3. Analyse statistique
    analyze_dataset(df)
    
    # 4. Visualisations
    visualize_dataset(df)
    
    print("\n" + "=" * 70)
    print(" PROCESSUS TERMINE AVEC SUCCES".center(70))
    print("=" * 70)
    print(f"\nFichiers generes:")
    print(f"   - {output_file} - Dataset CSV")
    print(f"   - visualizations/ - Graphiques et analyses")
    print("\n")


if __name__ == "__main__":
    main()
