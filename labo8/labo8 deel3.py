# Labo 8
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
current_file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file_path)
os.chdir(script_directory)

# 3. Deel 3 - Modelvalidatie met Cross-Validation (Nieuwe Dataset)
## 1.1 Stap 1 - Verkenning (kort en doelgericht)
### 1. Toon de eerste rijen van de dataset
df = sns.load_dataset('tips')
print(df.head(5))
print()
### 2. Controleer datatypes en ontbrekende waarden
print("Datatypes van de kolommen:")
print(df.dtypes)
print("Aantal ontbrekende waarden per kolom:")
print(df.isnull().sum())
print()
### 3. Kies minstens 3 features + 1 target
features = ['total_bill', 'size', 'sex']
target = 'tip'

### 4. Motiveer kort je keuze in commentaar
"""
Ik heb gekozen voor de features 'total_bill', 'size' en 'sex' en de target 'tip omdat de totale rekening (total_bill) waarschijnlijk een sterke invloed heeft op het bedrag van de fooi (tip). Het aantal personen aan tafel (size) kan ook van invloed zijn, omdat grotere groepen mogelijk meer fooi geven. Ten slotte kan het geslacht van de klant (sex) ook een invloed hebben op het bedrag van de fooi.
"""

## 1.2 Stap 2 - Probleemformulering
"""
Dit is een regressieprobleem omdat we een continue variabele ('tip') willen voorspellen op basis van de geselecteerde features ('total_bill', 'size', 'sex').
Het doel van het model is om te voorspellen hoeveel fooi een klant zal geven op basis van de totale rekening, het aantal personen aan tafel en het geslacht van de klant. We willen een model ontwikkelen dat in staat is om nauwkeurige voorspellingen te doen over het bedrag van de fooi, wat kan helpen bij het begrijpen van klantgedrag en het optimaliseren van service in een restaurantomgeving.
"""

## 1.3 Stap 3 - Voorbereiding data
### 1. Splits in X (features) en y (target)
X = df[features]
y = df[target]
### 2. Voer minimale preprocessing uit:
# Encoding indien nodig
X = pd.get_dummies(X, columns=['sex'], drop_first=True)
# Schalen indien relevant
scaler = StandardScaler()
X[['total_bill', 'size']] = scaler.fit_transform(X[['total_bill', 'size']])
print(X.head(5))
print()

## 1.4 Stap 4 - Model + Cross-Validation
### 1. Kies een passend model:
# Regressie → bv. Linear Regression
model = LinearRegression()
### 2. Gebruik k-fold cross-validation (k=5)
### 3. Gebruik cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
### 4. Toon: scores per fold, gemidddelde score, standaardafwijking
print("MSE scores per fold:", -scores)
print("Gemiddelde MSE:", -scores.mean())
print("Standaardafwijking MSE:", scores.std())
print()

## 1.5 Stap 5 - Metrieken
# R² score
r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("R² scores per fold:", r2_scores)
print("Gemiddelde R²:", r2_scores.mean())
print("Standaardafwijking R²:", r2_scores.std())
print()

# Mean Squared Error
mse_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print("MSE scores per fold:", -mse_scores)
print("Gemiddelde MSE:", -mse_scores.mean())
print("Standaardafwijking MSE:", mse_scores.std())
print()

## 1.6 Stap 6 - Interpretatie
"""
De gemiddelde R² score geeft aan hoeveel van de variatie in de targetvariabele (tip) wordt verklaard door het model. Een hogere R² score betekent dat het model beter past bij de data. De spreiding van de scores (standaardafwijking) geeft aan hoe consistent het model presteert over de verschillende folds. Een kleine spreiding suggereert dat het model stabiel is, terwijl een grote spreiding kan wijzen op overfitting of onderfitting. In dit geval vind ik de R² score het meest relevant omdat het direct aangeeft hoe goed het model de variatie in de targetvariabele verklaart. Om het model te verbeteren, zou ik kunnen experimenteren met meer geavanceerde regressiemodellen, zoals Random Forest of Gradient Boosting, en ook meer features toevoegen die mogelijk relevant zijn voor het voorspellen van de tip.
"""