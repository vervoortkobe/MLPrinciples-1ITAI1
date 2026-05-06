# Labo 8
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
current_file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file_path)
os.chdir(script_directory)

# 1. Deel 1 - Data-exploratie en eerste inzichten
## 1.1 Voorbereiding
### 3. Lees de dataset in
df = pd.read_csv ("cars.csv")
### 4. Toon de eerste vijf rijen van de dataset
print(df.head(5))
print()

## 1.2 Analyse en Visualisaties
### 5. Controleer de datatypes van de variabelen
print(f'Het datatype van speed is: {df["speed"].dtype}')
print(f'Het datatype van dist is: {df["dist"].dtype}')
print()
### 6. Controleer op ontbrekende data
print("Aantal ontbrekende waarden per kolom:")
print(df.isnull().sum())
print()
### 7. Bepaal de minimum- en maximumwaarden
print(f'Minimum snelheid: {df["speed"].min()}')
print(f'Maximum snelheid: {df["speed"].max()}')
print(f'Minimum afstand: {df["dist"].min()}')
print(f'Maximum afstand: {df["dist"].max()}')
print()
### 8. Genereer een histogram voor de variabele 'dist'
plt.figure(figsize=(8, 5))
sns.histplot(df['dist'], bins=20, kde=True)
plt.title('Histogram van afstand (dist)')
plt.xlabel('Afstand (dist)')
plt.ylabel('Frequentie')
plt.show()
"""
Uit deze visualisatie kan ik afleiden dat de meeste remafstanden zich bevinden tussen 0 en 100, met een piek rond de 20-30. Er zijn ook enkele uitschieters die verder gaan dan 100 (zelfs tot 120), wat aangeeft dat er enkele gevallen zijn waarbij de remafstand aanzienlijk groter is. De verdeling lijkt enigszins scheef te zijn, met een langere staart aan de rechterkant, wat suggereert dat er meer gevallen zijn met kleinere dan met grotere remafstanden.
"""
### 9. Maak een correlatiematrix
correlation_matrix = df[['speed', 'dist']].corr()
plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlatiematrix tussen speed en dist')
plt.show()
"""
De correlatie tussen 'speed' en 'dist' is zeer sterk positief, met een correlatiecoëfficiënt van 0.81. Dit betekent dat er een sterke relatie is tussen de snelheid van een auto en de afstand die nodig is om tot stilstand te komen. Naarmate de snelheid toeneemt, neemt ook de remafstand toe, wat logisch is. De sterke positieve correlatie suggereert dat snelheid een belangrijke factor is bij het bepalen van de remafstand, en dat hogere snelheden leiden tot langere remafstanden. Dit inzicht kan nuttig zijn voor verkeersveiligheid en het begrijpen van de risico's van hoge snelheden op de weg.
"""
### 10. Maak een lijndiagram
plt.figure(figsize=(8, 5))
plt.plot(df.index, df['speed'], marker='o')
plt.title('Lijndiagram van snelheid over tijd')
plt.xlabel('Tijd')
plt.ylabel('Snelheid (speed)')
plt.show()
"""
Interpreteer wat de trend lijkt te zijn.
We zien dat de snelheid (speed) in de loop van de tijd alleen maar een stijgende trend vertoont. Dit betekent dat de snelheid van de auto's in de dataset over het algemeen toeneemt naarmate de tijd vordert. Er zijn geen dalingen of fluctuaties in de snelheid (enkel korte momenten van stabilisatie), wat suggereert dat er een vrij consistente toename is in de snelheid van de auto's gedurende de tijdsperiode die in de dataset is opgenomen.
"""
### Reflectie
"""
Uit deze dataset heb ik geleerd dat er een sterke positieve correlatie bestaat tussen de snelheid van een auto en de afstand die nodig is om tot stilstand te komen. Dit betekent dat hogere snelheden leiden tot langere remafstanden, wat een belangrijk inzicht is voor verkeersveiligheid. In een realistische context kan dit betekenen dat het belangrijk is om snelheidslimieten te handhaven en bewustzijn te creëren over de risico's van hoge snelheden, aangezien het langer duurt om tot stilstand te komen bij hogere snelheden, wat kan leiden tot ernstigere ongevallen. Daarnaast heb ik geleerd dat de meeste remafstanden zich bevinden tussen 0 en 100, met enkele uitschieters tot 120 (bij 24 mph), wat suggereert dat er variabiliteit is in de remafstanden afhankelijk van verschillende factoren zoals wegcondities, voertuigtype, en rijgedrag.
"""