import pandas as pd
import os
import matplotlib.pyplot as plt



# Stap 1: Inladen dataset
# Zorg ervoor dat 'Trust_in_the_Police_in_Belgium.csv' in dezelfde map staat als dit Python-bestand
dataset = pd.read_csv('ESSBE5.csv')

# Stap 2: Verken de dataset
# Samenvatting van de dataset
print("Samenvatting van de dataset:")
print(dataset.head())
print(dataset.tail())

# Aantal rijen en kolommen
print("Aantal rijen en kolommen:")
print(dataset.shape)

# DIPKEVER analyse
print("DIPKEVER analyse:")
print(dataset.info())

# Stap 3: Data cleaning en voorbereiding
# Ontbrekende waarden controleren en behandelen
print("Ontbrekende waarden behandelen:")
print(dataset.isnull().sum())

# Gegevens verwijderen
# dataset = dataset.dropna()  # Uncomment deze regel als je ontbrekende waarden wilt verwijderen

# Stap 4: Visualisatie van de dataset
# Visualisatie van het vertrouwen in de politie (variabele trstplc)
plt.figure(figsize=(10, 6))
plt.hist(dataset['trstplc'], bins=11, color='skyblue', edgecolor='black')
plt.title('Vertrouwen in de Belgische politie')
plt.xlabel('Vertrouwen (schaal van 0 tot 10)')
plt.ylabel('Aantal respondenten')
plt.grid(True)
plt.show()

# Visualisatie van de leeftijd (variabele agea)
plt.figure(figsize=(10, 6))
plt.hist(dataset['agea'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Leeftijd van de respondenten')
plt.xlabel('Leeftijd')
plt.ylabel('Aantal respondenten')
plt.grid(True)
plt.show()

# Breng beide variabelen samen in een scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(dataset['agea'], dataset['trstplc'], color='orange')
plt.title('Vertrouwen in de politie per leeftijd')
plt.xlabel('Leeftijd')
plt.ylabel('Vertrouwen (schaal van 0 tot 10)')
plt.grid(True)
plt.show()

# Stap 5: Analyse
# Gemiddelde leeftijd
gemiddelde_leeftijd = int(dataset['agea'].mean())
print("Gemiddelde leeftijd van de respondenten:", gemiddelde_leeftijd)

# Gemiddeld vertrouwen in de politie
gemiddeld_vertrouwen = int(dataset['trstplc'].mean())
print("Gemiddeld vertrouwen in de Belgische politie:", gemiddeld_vertrouwen)

# Correlaties tussen variabelen
correlaties = dataset.corr(numeric_only=True)
print("Correlaties tussen variabelen:")
print(correlaties)

# Bepaal of er meer mannelijke of vrouwelijke deelnemers zijn
gender_counts = dataset['female'].value_counts()
print("Aantal mannelijke en vrouwelijke deelnemers:")
print(gender_counts)