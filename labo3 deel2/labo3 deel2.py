# Labo 3 - Deel 2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
current_file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file_path)
os.chdir(script_directory)

# Deel 1: datatypes en structuren in AI
# 1. Lees de Titanic dataset in met Pandas en toon de eerste 5 rijen.
titanic = sns.load_dataset('titanic')
df = pd.DataFrame(titanic)
print(df.head())

# 2. Controleer de datatypes van de kolommen
print(df.dtypes)
print()
print(df.select_dtypes(np.number).columns)
print()

numerical_vars = [col for col in df.columns
                  if pd.api.types.is_numeric_dtype(df[col])
                  and not pd.api.types.is_bool_dtype(df[col])]
categorical_vars = [col for col in df.columns
                    if not pd.api.types.is_numeric_dtype(df[col])
                    and not pd.api.types.is_bool_dtype(df[col])]

print("Numerieke variabelen:", numerical_vars)
print("Categorische variabelen:", categorical_vars)
print()

# 3. Converteer een categorische variabele naar numerieke representatie
le = LabelEncoder()
df['class_encoded'] = le.fit_transform(df['class'])
print("Class variabele voor en na encoding:")
print(df[['class', 'class_encoded']].head())
print("Class mapping:", dict(zip(le.classes_, range(len(le.classes_)))))
print()

# 4. Beantwoord de volgende vragen in de reflectie van je portfolio:
# Welke variabele(n) kunnen ordinaal worden beschouwd?
# Welke variabelen zijn nominaal en moeten anders worden behandeld?
"""
pclass, survived en sex kunnen ordinaal worden beschouwd. pclass heeft een natuurlijke hiërarchie (1=hoogste klasse > 2 > 3=laagste klasse). survived is binair ordinaal (1=overleefd > 0=overleden). sex kan technisch ordinaal worden behandeld voor veel modellen, hoewel het biologisch nominaal is.

who, embark_town, embarked, deck en class zijn nominaal omdat ze geen natuurlijke volgorde hebben. Deze vereisen One-Hot Encoding om kunstmatige hiërarchieën te vermijden die modellen kunnen misleiden. Label Encoding zou hier foutieve numerieke relaties impliceren (bijv. deck A=0 < deck B=1).
"""

# Reflectie portfolio: waarom is het belangrijk om het juiste datatype te kiezen in AI-toepassingen?
"""
Het juiste datatype voorkomt dat modellen foute aannames maken over datarelaties, wat leidt tot slechtere voorspellingen en overfitting. Nominale variabelen met Label Encoding doen alsof categorieën een volgorde hebben, wat lineaire modellen verwart. One-Hot Encoding behoudt de ware aard van nominale data maar verhoogt dimensionaliteit. Tree-based modellen verdragen Label Encoding beter omdat ze geen lineaire relaties aannemen. Uiteindelijk bepaalt datatype-keuze de robuustheid en generaliseerbaarheid van AI-modellen.
"""

# Deel 2: omgaan met Categorical Data in AI
# 5. Voer een feature-representatie uit op de kolommen ‘sex’ en ‘embark_town’.
# Gebruik One-Hot Encoding voor ‘embark_town’.
print("ORIGINELE KOLOMMEN")
print("Sex:")
print(df['sex'].head(10).to_string())
print()
print("Embark_town:")
print(df['embark_town'].head(10).to_string())
print()

le_sex = LabelEncoder()
df['sex_le'] = le_sex.fit_transform(df['sex'])
print("LABEL ENCODING 'sex'")
print("Sex LE mapping:", dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_))))
print("Sex LE:", df['sex_le'].head(10).to_string())
print()

# Gebruik Label Encoding voor ‘sex’.
df_encoded = pd.get_dummies(df, columns=['embark_town'], prefix='embark_town', drop_first=True)
print("ONE-HOT ENCODING 'embark_town'")
print("Nieuwe kolommen:", [col for col in df_encoded.columns if 'embark_town' in col])
print(df_encoded[['embark_town_Queenstown', 'embark_town_Southampton']].head(10).to_string())
print()

print("VERGELIJKING VOOR/NA")
print("Aantal kolommen voor:", len(df.columns))
print("Aantal kolommen na:", len(df_encoded.columns))
print("Toegevoegde kolommen:", len(df_encoded.columns) - len(df.columns))
print()

# 6. Vergelijk de impact van beide methodes:
# Wat is het verschil tussen de twee?
# Welke methode is het meest geschikt en waarom?
"""
Label Encoding converteert 'sex' naar één numerieke kolom (female=0, male=1), terwijl One-Hot Encoding 'embark_town' omzet in meerdere binaire kolommen (één per haven). Label Encoding is compacter maar impliceert een volgorde tussen categorieën, wat bij 'sex' geen probleem vormt door de binaire natuur. One-Hot Encoding voorkomt deze foute ordinaliteit bij 'embark_town' met vier havens, elke als onafhankelijke feature. Label Encoding is meest geschikt voor 'sex' vanwege efficiëntie; One-Hot Encoding past beter bij 'embark_town' om multicollineariteit en hiërarchische interpretatiefouten te vermijden.
"""

# Deel 3: data cleaning en voorbereiding
# 7. Controleer op ontbrekende waarden en beslis wat de beste aanpak is.
# Imputeren met de gemiddelde waarde
# Verwijderen van records
# Vullen met de meest voorkomende waarde
print("Ontbrekende waarden per kolom:")
print(df.isnull().sum())
print()

# 8. Voer de gekozen strategie uit en beargumenteer in je portfolio waarom je deze keuze hebt gemaakt.
df['age'] = df['age'].fillna(df['age'].median())
most_frequent_town = df['embark_town'].mode()[0]
df['embark_town'] = df['embark_town'].fillna(most_frequent_town)
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df.drop(columns=['deck'], inplace=True)
print("Controle na cleaning (aantal null-waarden):")
print(df.isnull().sum())
print(f"Huidige vorm van de dataset: {df.shape}")
print()

"""
Age (Imputeren met mediaan):
    Keuze: Ik heb de ontbrekende waarden ingevuld met de mediaan van de bestaande leeftijden.
    Argumentatie: Ongeveer 20% van de data in de kolom 'age' ontbrak. Het verwijderen van al deze rijen zou leiden tot een aanzienlijk verlies van trainingsdata voor het AI-model. Ik heb gekozen voor de mediaan in plaats van het gemiddelde (mean), omdat de leeftijd in de Titanic-dataset uitschieters bevat (zoals zeer jonge kinderen en oudere passagiers). De mediaan is robuuster en wordt minder beïnvloed door deze uitschieters.

Embark_town (Vullen met de meest voorkomende waarde):
    Keuze: Ik heb de twee ontbrekende waarden ingevuld met de 'modus' (Southampton).
    Argumentatie: Aangezien er slechts twee waarden ontbraken op een totaal van 891, is de impact van deze gok minimaal. Het vullen met de meest voorkomende categorie (de modus) is voor categorische data de standaardprocedure wanneer het aantal ontbrekende waarden verwaarloosbaar is.

Deck (Verwijderen van de kolom):
    Keuze: Ik heb de volledige kolom 'deck' verwijderd.
    Argumentatie: Meer dan 77% van de data in deze kolom ontbrak. Als ik deze waarden zou proberen te "raden" (imputeren), zou ik een enorme hoeveelheid ruis en onbetrouwbare informatie in het model introduceren. De kolom bevat simpelweg niet genoeg informatie om bruikbaar te zijn voor een AI-toepassing.

Conclusie:
Door deze strategieën toe te passen, heb ik een "schone" dataset gecreëerd zonder gaten, terwijl de integriteit en de statistische verhoudingen van de belangrijkste variabelen behouden zijn gebleven. Dit is essentieel voor het trainen van een accuraat model.
"""

# 9. Controleer de dataset opnieuw: zijn er nog ontbrekende waarden?
missing_values_after = df.isnull().sum()
print(missing_values_after)
print()
# fillna() vult ontbrekende waarden op basis van wat jij kiest:
# - een vaste waarde
# - de mediaan
# - het gemiddelde
# - de modus
# - forward/backward fill

# df['age'] = df['age'].fillna(df['age'].median())

# Extra: correlatiematrix van numerieke variabelen
# 10. Maak een correlatiematrix van de numerieke variabelen en visualiseer deze met een heatmap.
numeric_df = df.select_dtypes(include=[np.number])

corr_matrix = numeric_df.corr()
print("Correlatiematrix:")
print(corr_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlatiematrix van numerieke variabelen")
plt.show()

# 11. Welke variabelen vertonen de sterkste correlaties? Wat zou dit kunnen betekenen in de context van de Titanic?
"""
De sterkste relaties die we uit de correlatiematrix kunnen afleiden zijn:
- 'fare' en 'pclass': hoe hoger de klasse, hoe duurder het ticket (-0.55).
- 'survived' en 'sex_le': vrouwen overleefden veel vaker dan mannen (-0.54).
- 'survived' en 'pclass': passagiers in hogere klassen hadden een grotere overlevingskans (-0.34).
- 'age' en 'pclass': oudere passagiers zaten vaker in hogere klassen (-0.37).

De sterkste correlaties in de Titanic‑dataset tonen duidelijk dat geslacht en sociale klasse de belangrijkste factoren waren die de overlevingskans bepaalden. Vrouwen en passagiers uit de eerste klasse hadden een aanzienlijk grotere kans om te overleven. (Vrouwen en kinderen worden altijd eerst geëvacueerd.) De correlaties tussen pclass en fare bevestigen bovendien de sociale hiërarchie aan boord: rijkere passagiers betaalden meer en zaten in betere klassen.
"""

# Reflectie en evaluatie (Portfolio)
# Wat heb je geleerd over hoe data correct moet worden voorgeformatteerd voordat het wordt gebruikt in AI-modellen?
# Welke technieken waren het meest zinvol voor categorische variabelen?
# Waarom is een juiste datarepresentatie cruciaal voor machine learning?
"""
Ik heb geleerd dat ruwe data meestal niet direct bruikbaar is voor machine learning. Het preprocessen van data is essentieel om ontbrekende waarden op te lossen, zodat modellen geen fouten maken of patronen missen. Ook is het erg van belang dat de datatypes correct geïnterpreteerd worden (numeriek, categorisch, ordinaal) door het model. Ruis en irrelevante variabelen moeten best verwijderd worden om overfitting te voorkomen. Aldus is het cruciaal om een consistente, betrouwbare input dataset te creëren voor elk algoritme. Zonder deze stappen kan een model verkeerde conclusies trekken of sterk in performance dalen.

Voor categorische data bleken vooral deze technieken belangrijk:
- Label Encoding voor binaire of ordinale variabelen (zoals sex of pclass)
- One‑Hot Encoding voor nominale variabelen zonder natuurlijke volgorde (zoals embark_town)
- Imputatie van ontbrekende waarden met geschikte methoden (mediaan voor numeriek, modus voor categorisch)
Deze technieken zorgen ervoor dat het model de categorieën correct begrijpt zonder kunstmatige hiërarchie.

Een correcte representatie bepaalt hoe een model de wereld “ziet”.
Als variabelen verkeerd worden gecodeerd, kunnen modellen foute aannames maken en ontstaan er valse relaties. Ook kunnen modellen misleid (bij bv. One-Hot Encoding) worden door numerieke schalen die niet bestaan. Dit leidt tot slechte generalisatie en overfitting, omdat het model leert van ruis in plaats van echte patronen. Goede representatie zorgt ervoor dat het model echte patronen leert in plaats van het trekken van foute conclusies uit data met irrelevante variabelen.
"""