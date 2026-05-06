# Labo 8
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
current_file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file_path)
os.chdir(script_directory)

# 2. Deel 2 - Uitgebreide analyse en keuzevrijheid in visualisatie
## 1.1 Voorbereiding
### 2. Lees de dataset in
df = pd.read_csv ("Boston.csv")
print(df.head(5))
print()
### 3. Bekijk de documentatie van de dataset (klik op de 'DOC' link naast de dataset) en selecteer minstens vier variabelen die je relevant vindt om verder te onderzoeken.
"""
Uit de documentatie van de Boston dataset heb ik de volgende vier variabelen geselecteerd om verder te onderzoeken:
1. crim: Per capita crime rate by town
2. zn: Proportion of residential land zoned for lots over 25,000 sq.ft.
3. indus: Proportion of non-retail business acres per town
4. chas: Charles River dummy variable
Deze variabelen zijn relevant omdat ze verschillende aspecten van de stedelijke omgeving en de sociale context van de wijken in Boston vertegenwoordigen. CRIM geeft inzicht in de criminaliteitsniveaus, ZN en INDUS bieden informatie over de landgebruikspatronen en economische activiteiten, terwijl CHAS aangeeft of een wijk grenst aan de Charles River, wat mogelijk invloed kan hebben op de leefbaarheid en vastgoedwaarden in die gebieden.
"""

## 1.2 Data-analyse en Visualisatie
### 4. Controleer de datatypes en ontbrekende waarden
print(f'Het datatype van crim is: {df["crim"].dtype}')
print(f'Het datatype van zn is: {df["zn"].dtype}')
print(f'Het datatype van indus is: {df["indus"].dtype}')
print(f'Het datatype van chas is: {df["chas"].dtype}')
print("Aantal ontbrekende waarden per kolom:")
print(df.isnull().sum())
print()
### 5. Bereken de gemiddelde, minimale en maximale waarde
print(f'Gemiddelde CRIM: {df["crim"].mean()}')
print(f'Minimum CRIM: {df["crim"].min()}')
print(f'Maximum CRIM: {df["crim"].max()}')
print(f'Gemiddelde ZN: {df["zn"].mean()}')
print(f'Minimum ZN: {df["zn"].min()}')
print(f'Maximum ZN: {df["zn"].max()}')
print(f'Gemiddelde INDUS: {df["indus"].mean()}')
print(f'Minimum INDUS: {df["indus"].min()}')
print(f'Maximum INDUS: {df["indus"].max()}')
print(f'Gemiddelde CHAS: {df["chas"].mean()}')
print(f'Minimum CHAS: {df["chas"].min()}')
print(f'Maximum CHAS: {df["chas"].max()}')
print()
### 6. Maak een correlatiematrix
correlation_matrix = df[['crim', 'zn', 'indus', 'chas']].corr()
plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlatiematrix tussen crim, zn, indus en chas')
plt.show()
"""
Welke relaties tussen variabelen vallen je op?
Uit de correlatiematrix valt op dat er een gemiddeld negatieve correlatie is tussen 'crim' en 'zn' (-0.2), wat suggereert dat hogere criminaliteitsniveaus geassocieerd zijn met een lager percentage van residentieel land dat is bestemd voor grote percelen. 
Er is ook een positieve correlatie tussen 'crim' en 'indus' (0.41), wat aangeeft dat hogere criminaliteitsniveaus geassocieerd zijn met een hoger percentage van niet-retail zakelijke acres per stad. 
De variabele 'chas' heeft een zwakke negatieve correlatie met 'crim' (-0.056), wat suggereert dat wijken die grenzen aan de Charles River mogelijk iets lagere criminaliteitsniveaus hebben, hoewel deze relatie niet sterk is. 
Over het algemeen lijken er enkele interessante relaties te zijn tussen deze variabelen, waarbij criminaliteit een centrale rol speelt in verband met zowel landgebruik als nabijheid van de Charles River.
"""
### 7. Tijdreeksanalyse
plt.figure(figsize=(8, 5))
plt.plot(df.index, df['medv'], marker='o')
plt.title('Tijdreeksplot van huisprijzen over de jaren')
plt.xlabel('Jaar')
plt.ylabel('Huisprijzen (medv)')
plt.show()
"""
Aangezien er geen jaartallen in de dataset staan, is deze tijdreeksplot niet echt representatief voor een tijdsverloop. De index van de dataset vertegenwoordigt niet noodzakelijkerwijs opeenvolgende jaren, dus het is moeilijk om hier conclusies uit te trekken over trends in huisprijzen over de tijd.
"""
### 8. Distributie van jaarlijkse veranderingen
df['medv_change'] = df['medv'].diff()
plt.figure(figsize=(8, 5))
sns.histplot(df['medv_change'], bins=20, kde=True)
plt.title('Histogram van jaarlijkse veranderingen in huisprijzen (medv_change)')
plt.xlabel('Jaarlijkse verandering in huisprijzen (medv_change)')
plt.ylabel('Frequentie')
plt.show()
"""
Dit histogram toont de distributie van jaarlijkse veranderingen in huisprijzen (medv_change). De meeste jaarlijkse veranderingen lijken zich te concentreren rond nul, wat suggereert dat de meeste jaren geen grote stijgingen of dalingen in huisprijzen hebben. Er zijn echter enkele uitschieters aan zowel de positieve als negatieve kant, wat aangeeft dat er jaren zijn geweest met aanzienlijke stijgingen of dalingen in huisprijzen. De verdeling lijkt enigszins scheef te zijn, met een kleine staart aan de negatieve kant, wat suggereert dat er iets meer jaren zijn geweest met dalingen in huisprijzen dan met stijgingen. Over het algemeen lijkt de markt voor huisprijzen in deze dataset relatief stabiel te zijn, met enkele uitzonderlijke jaren die aanzienlijke veranderingen hebben gekend.
"""
### 9. Zelfstandige visualisatiekeuze
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['chas'], y=df['medv'])
plt.title('Boxplot van huisprijzen en grenzen aan de Charles River')
plt.xlabel('Grenst aan Charles River (chas)')
plt.ylabel('Huisprijzen (medv)')
plt.show()
"""
Deze boxplot toont de relatie tussen het al dan niet grenzen aan de Charles River (chas) en de huisprijzen (medv). Ik heb deze visualisatie gekozen omdat ik geïnteresseerd ben in het onderzoeken of er een verschil is in huisprijzen tussen wijken die wel of niet grenzen aan de Charles River. Uit de boxplot blijkt dat huizen die grenzen aan de Charles River over het algemeen hogere mediane prijzen hebben en minder variatie in prijzen vertonen in vergelijking met huizen die niet grenzen aan de rivier. Dit suggereert dat nabijheid van de Charles River mogelijk een positieve invloed heeft op de waarde van huizen, wat een interessant inzicht is voor potentiële huizenkopers en vastgoedontwikkelaars.
"""
### Reflectie
"""
Uit de analyse van de Boston dataset heb ik geleerd dat verschillende variabelen een rol spelen in de huizenmarkt, maar dat nabijheid van de Charles River (chas) een opvallende invloed lijkt te hebben op huisprijzen (medv). Huizen die grenzen aan de Charles River hebben over het algemeen hogere mediane prijzen en minder variatie in prijzen, wat suggereert dat deze locatie een aantrekkelijke factor is voor kopers. Daarnaast blijkt dat criminaliteit (crim) een negatieve correlatie heeft met de hoeveelheid residentieel land dat is bestemd voor grote percelen (zn) en een positieve correlatie met niet-retail zakelijke acres (indus), wat kan wijzen op een verband tussen stedelijke ontwikkeling, criminaliteit en huisprijzen. Deze trends kunnen betekenen dat de huizenmarkt in Boston sterk wordt beïnvloed door zowel geografische factoren (zoals nabijheid van de rivier) als sociale en economische factoren (zoals criminaliteit en landgebruik), wat belangrijk is voor zowel kopers als beleidsmakers om te overwegen bij het nemen van beslissingen over vastgoed en stadsplanning.
"""