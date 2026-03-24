# Deel 2 - Uitgebreide analyse en keuzevrijheid in visualisatie
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
current_file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file_path)
os.chdir(script_directory)

# Lees de dataset in
df = pd.read_csv("Boston.csv")
print(df.head(5))

# Controleer de datatypes en ontbrekende waarden
print(f"Datatype van 'crim': {df["crim"].dtypes}")
print(f"Datatype van 'age': {df["age"].dtypes}")
print(f"Datatype van 'black': {df["black"].dtypes}")
print(f"Datatype van 'ptratio': {df["ptratio"].dtypes}")
print(f"Datatype van 'lstat': {df["lstat"].dtypes}")
print(f"Aantal ontbrekende waarden in 'crim': {df["crim"].isnull().sum()}")
print(f"Aantal ontbrekende waarden in 'age': {df["age"].isnull().sum()}")
print(f"Aantal ontbrekende waarden in 'black': {df["black"].isnull().sum()}")
print(f"Aantal ontbrekende waarden in 'ptratio': {df["ptratio"].isnull().sum()}")
print(f"Aantal ontbrekende waarden in 'lstat': {df["lstat"].isnull().sum()}")

# Bereken de gemiddelde, minimale en maximale waarde
stats = {}
vars = ["crim", "age", "black", "ptratio", "lstat"]
for var in vars:
    stats[var] = {
        "Gemiddelde": df[var].mean(),
        "Minimum": df[var].min(),
        "Maximum": df[var].max()
    }
tabel = pd.DataFrame(stats)
print(tabel.T)

# Maak een correlatiematrix
sb.heatmap(df[vars].corr(), annot=True, cmap="Reds", center=0, linewidths=0.5)
plt.title("Correlatiematrix van de geselecteerde variabelen")
plt.tight_layout()
plt.show()
"""
Ik zie enkele opvallende correlaties:
- "crim" heeft een sterke negatieve correlatie met "black" (-0.39), wat suggereert dat gebieden met hogere criminaliteitsgraden vaak minder zwarte inwoners hebben.
- "age" heeft een zwak positieve correlatie met "crim" (0.35), wat kan betekenen dat oudere huizen vaker in gebieden met hogere criminaliteitsgraden liggen.
- "ptratio" heeft een zwak positieve correlatie met "crim" (0.29), wat kan duiden op een verband tussen hogere student-leraarverhoudingen en hogere criminaliteitsgraden.
- "lstat" heeft een matig positieve correlatie met "crim" (0.46), wat suggereert dat een lagere sociale status geassocieerd kan zijn met hogere criminaliteitsgraden.
Ik merk ook op dat "black" negatief gecorreleerd is met alle andere variabelen, wat betekent dat een hogere waarde van "black" (meer zwarte inwoners), geassocieerd is met lagere criminaliteitsgraden, minder oude huizen, minder studenten per leeraar en een hogere sociale status. Deze correlaties zouden kunnen wijzen op mogelijke segregatie of sociaaleconomische verschillen in de dataset (wijken rond Boston).
"""

# Distributieanalyse van de huizenprijzen
plt.figure()
plt.hist(df["medv"], bins=20, edgecolor="black")
plt.xlabel("medv (mediane huizenwaarde in duizenden $)")
plt.ylabel("Frequentie")
plt.title("Verdeling van de huizenprijzen (medv)")
plt.tight_layout()
plt.show()
"""
De verdeling van het histogram is rechtsscheef, wat betekent dat de meeste huizen een relatief lage tot gemiddelde prijs hebben, terwijl er een kleinere groep huizen is die aanzienlijk duurder zijn. Er zijn enkele uitschieters aan de rechterkant van het histogram, wat suggereert dat er een aantal huizen zijn met zeer hoge prijzen. De meeste huizen lijken echter in het bereik van $10.000 tot $30.000 te liggen, met een piek rond de $20.000. Deze scheve verdeling kan wijzen op ongelijkheid van welvaart tussen wijken in de dataset.
"""

# Onderzoek de invloed van de afstand tot het zakencentrum (dis) en de criminaliteitsgraad (crim) op de huizenprijzen (medv)
plt.figure()
plt.scatter(df["dis"], df["medv"], alpha=0.6)
plt.xlabel("dis (afstand tot werkcentra)")
plt.ylabel("medv")
plt.title("Afstand tot werkcentra vs huizenprijzen")
plt.tight_layout()
plt.show()
"""
Ik zie een positief verband tussen de afstand tot werkcentra (dis) en de huizenprijzen (medv). Huizen die verder weg liggen van de werkcentra zullen over het algemeen hogere prijzen hebben. Dit kan er op wijzen dat de duurdere huizen zich bevinden in buitenwijken of landelijke, rustige gebieden die verder van het stadscentrum liggen, en dat de werkcentra mogelijk gelegen zijn in oudere, dichter bebouwde en goedkopere wijken.
"""

plt.figure()
plt.scatter(df["crim"], df["medv"], alpha=0.6)
plt.xlabel("crim (criminaliteitsgraad)")
plt.ylabel("medv")
plt.title("Criminaliteitsgraad vs huizenprijzen")
plt.tight_layout()
plt.show()
"""
Ik merk een negatief verband op tussen de criminaliteitsgraad (crim) en de huizenprijzen (medv). Huizen in gebieden met een hogere criminaliteitsgraad hebben over het algemeen lagere prijzen. Dit kan erop wijzen dat kopers minder bereid zijn om te betalen voor huizen in gebieden met meer criminaliteit, wat kan leiden tot lagere huizenprijzen in die gebieden. De meeste wijken hebben een lage criminaliteitsgraad en een brede spreiding in huizenprijzen, maar er zijn enkele wijken met zeer hoge criminaliteit en extreem lage huizenprijzen. Wijken met veel criminaliteit zijn dus aanzienlijk goedkoper.
"""

# Zelfstandige visualisatiekeuze
median_crim = df["crim"].median()
df["crime_group"] = np.where(df["crim"] > median_crim, 
                             "Hoge criminaliteit", 
                             "Lage criminaliteit")

sb.boxplot(x="crime_group", y="medv", data=df)
plt.xlabel("Criminaliteitsniveau")
plt.ylabel("medv")
plt.title("Huizenprijzen bij lage vs hoge criminaliteit")
plt.tight_layout()
plt.show()
"""
Ik koos voor een boxplot omdat deze visualisatie niet alleen het verschil in gemiddelde of mediaan toont tussen groepen, maar ook inzicht geeft in de spreiding, scheefheid en uitschieters binnen elke groep. In tegenstelling tot het scatterplot, dat vooral de algemene trend weergeeft, maakt de boxplot een directe vergelijking mogelijk tussen wijken met lage en hoge criminaliteit. 

Wat meteen opvalt, is dat de mediaan van de huizenprijzen bij lage criminaliteit ongeveer $5000 hoger ligt dan bij hoge criminaliteit. Daarnaast is de spreiding bij lage criminaliteit veel groter: er zijn zowel goedkope als zeer dure wijken, terwijl bij hoge criminaliteit de prijzen lager en meer geconcentreerd zijn. Tot slot zijn er bij lage criminaliteit meerdere uitschieters met prijzen boven de $45.000, wat wijst op een kleine groep zeer duurdere, veiligere wijken.
"""

"""
Reflectie:
Wat mij het meest opviel tijdens deze analyse, is hoe sterk verschillende buurtkenmerken met elkaar verweven zijn en samen de huizenprijzen beïnvloeden. Criminaliteit blijkt een van de belangrijkste voorspellers: wijken met veel criminaliteit zijn consequent goedkoper, en dat verschil is niet klein. Ook de sociale status van een wijk (lstat) speelt een grote rol. Hoe hoger het percentage lagere status, hoe lager de prijzen. Wat me verraste, is dat afstand tot werkcentra (dis) positief samenhangt met prijzen: verder weg wonen is in Boston blijkbaar juist aantrekkelijker, misschien vanwege de rust of de natuurlijke omgeving. De negatieve correlatie van black met alle andere variabelen viel me ook op; dat wijst mogelijk op sociaaleconomische scheefheid of segregatie in de stad.

Voor de huizenmarkt betekenen deze trends dat prijzen sterk worden gedreven door veiligheid en sociale samenstelling, niet alleen door locatie. Wijken met een lagere status of hogere criminaliteit zijn moeilijker te verkopen tegen hoge prijzen, wat kan leiden tot verdere achteruitgang als er niet wordt geïnvesteerd. Andersom kunnen wijken die als veilig en gewild worden gezien, juist steeds duurder worden, wat gentrificatie in de hand werkt.

Door het analyseren van deze dataset, heb ik een beter begrip gekregen van hoe verschillende factoren en complexe sociale dynamieken samenhangen en invloed hebben op de huizenprijzen en de leefbaarheid van wijken.
"""