# PARTIE A — Chargement & choix des variables

import pandas as pd, numpy as np 
import warnings; warnings.filterwarnings('ignore')

df = pd.read_excel('C:/Users/SONITA/Documents/learning_py/ODC/DSCHANG/tp_pandas/scoring_resiliation/data/dataset_assurance_ML.xlsx') 
df.to_csv('C:/Users/SONITA/Documents/learning_py/ODC/DSCHANG/tp_pandas/scoring_resiliation/data/dataset_assurance_ML.csv', index=False, encoding='utf-8-sig') 

df = pd.read_csv('C:/Users/SONITA/Documents/learning_py/ODC/DSCHANG/tp_pandas/scoring_resiliation/data/dataset_assurance_ML.csv', encoding='utf-8-sig') 
print(df.shape)

print('\nNombre de valeurs manquantes : \n',df.isnull().sum())
print('\nDoublons :',df.duplicated().sum())

#Etape 2 : la variable cible
TARGET = 'Résiliation' 
print(df[TARGET].value_counts()) 
print(df[TARGET].value_counts(normalize=True).round(2)) 

#Etape 3
print(pd.crosstab(df['Statut Contrat'], df[TARGET]))

#Etape 4
num_cols = ['Âge', 'Salaire Annuel (€)', 'Prime Annuelle (€)', 'Ancienneté (mois)', 
            'Coeff. Bonus-Malus', 'Nb Sinistres (3 ans)', 
            'Montant Sinistres (€)', 'Score Risque (0-100)'] 
cat_cols = ['Type Contrat', 'Catégorie Prof.', 'Usage Véhicule', 'Dernier Sinistre'] 
X = df[num_cols + cat_cols] 
y = df[TARGET] 
print(X.shape, y.shape)

#Etape 5
print(X[num_cols].corrwith(y).round(3).sort_values(ascending=False)) 
print((df.groupby('Dernier Sinistre')[TARGET].mean()*100).round(2).sort_values()) 

#PARTIE B — Pipeline, entraînement & évaluation

#Etape 6
from sklearn.model_selection import train_test_split 
  
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42, stratify=y) 
print(X_train.shape, X_test.shape) 
print(y_train.mean().round(2), y_test.mean().round(2))

#Etape 7
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
  
preprocessor = ColumnTransformer([ 
    ('num', StandardScaler(), num_cols), 
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols), 
])

#Etape 8
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
  
candidats = { 
    'Régression Logistique': LogisticRegression( 
        max_iter=1000, class_weight='balanced', random_state=42), 
    'Random Forest': RandomForestClassifier( 
        n_estimators=300, max_depth=4, min_samples_leaf=10, 
        class_weight='balanced', random_state=42), 
} 
pipelines = {nom: Pipeline([('prep', preprocessor), ('model', algo)]) 
             for nom, algo in candidats.items()}

#Etape 9
from sklearn.model_selection import cross_val_score 

for nom, pipe in pipelines.items(): 
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc') 
    print(f'{nom:22s} AUC = {scores.mean():.3f} ± {scores.std():.3f}') 

#Etape 10
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
confusion_matrix, classification_report) 

pipeline = pipelines['Random Forest'] 
pipeline.fit(X_train, y_train) 
y_pred  = pipeline.predict(X_test) 
y_proba = pipeline.predict_proba(X_test)[:, 1] 
print('Accuracy :', round(accuracy_score(y_test, y_pred),3))
print('F1       :', round(f1_score(y_test, y_pred), 3)) 
print('ROC-AUC  :', round(roc_auc_score(y_test, y_proba), 3)) 
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred, target_names=['Reste', 'Résilie'])) 

#PARTIE C — Sauvegarde & interrogation du modèle

#Etape 12
import joblib, os 

os.makedirs('../models', exist_ok=True)
joblib.dump(pipeline, '../models/pipeline_resiliation.pkl') 
print(os.path.getsize('../models/pipeline_resiliation.pkl') / 1024, 'Ko')

#Etape 13
import json 

meta = { 
    'modele': 'Random Forest', 
    'auc_test': round(float(roc_auc_score(y_test, y_proba)), 3), 
    'num_cols': num_cols, 
    'cat_cols': cat_cols, 
    'num_ranges': {c: {'min': float(X[c].min()), 'max': float(X[c].max()),'median': float(X[c].median())} for c in num_cols}, 
    'cat_values': {c: sorted(X[c].unique().tolist()) for c in cat_cols}, 
} 
with open('../models/metadata.json', 'w', encoding='utf-8') as f: 
    json.dump(meta, f, ensure_ascii=False, indent=2) 

#Etape 14

modele = joblib.load('../models/pipeline_resiliation.pkl') 
  
client = pd.DataFrame([{ 
    'Âge': 34, 'Salaire Annuel (€)': 28000, 'Prime Annuelle (€)': 950, 
    'Ancienneté (mois)': 6, 'Coeff. Bonus-Malus': 1.25, 'Nb Sinistres (3 ans)': 3, 
    'Montant Sinistres (€)': 4200, 'Score Risque (0-100)': 72, 
    'Type Contrat': 'Bronze', 'Catégorie Prof.': 'Entrepreneur', 
    'Usage Véhicule': 'Professionnel', 'Dernier Sinistre': 'Vol', 
}]) 
print('Classe :', modele.predict(client)) 
print('Proba  :', modele.predict_proba(client)[0, 1].round(3))

#Etape 15
try: 
    modele.predict(client.drop(columns=['Score Risque (0-100)'])) 
except Exception as e: 
    print('ERREUR :', e)
