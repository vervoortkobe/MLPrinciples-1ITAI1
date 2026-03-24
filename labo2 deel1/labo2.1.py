# Deel 1 - Eerste inzichten uit een dataset
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
current_file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file_path)
os.chdir(script_directory)

# Lees de dataset in
df = pd.read_csv("cars.csv", index_col=0)

# Toon de eerste vijf rijen van de dataset
print(df.head(5))

# Controleer de datatypes van de variabelen
print(f"Datatype van 'speed': {df["speed"].dtypes}")
print(f"Datatype van 'dist': {df["dist"].dtypes}")

# Controleer op ontbrekende data
print(f"Aantal ontbrekende waarden in 'speed': {df["speed"].isnull().sum()}")
print(f"Aantal ontbrekende waarden in 'dist': {df["dist"].isnull().sum()}")

# Bepaal de minimum- en maximumwaarden
print(f"Minimumwaarde van 'speed': {df["speed"].min()}")
print(f"Maximumwaarde van 'speed': {df["speed"].max()}")
print(f"Minimumwaarde van 'dist': {df["dist"].min()}")
print(f"Maximumwaarde van 'dist': {df["dist"].max()}")

# Genereer een histogram voor de variabele 'dist'
df["dist"].hist(bins=20, color='orange', edgecolor='black', linewidth=1.2)
plt.title("Histogram van Afstand")
plt.xlabel("Afstand")
plt.ylabel("Frequentie")
plt.grid(False)
plt.tight_layout()
plt.show()

# Maak een correlatiematrix
sb.heatmap(df.corr(), annot=True, cmap="Reds", center=0, linewidths=0.5)
plt.title("Correlatiematrix van snelheid en remafstand")
plt.tight_layout()
plt.show()
"""
We zien dat er een vrij sterke positieve correlatie (van 0.81) is tussen snelheid en remafstand. Dit betekent dat naarmate de snelheid toeneemt, de remafstand ook toeneemt. Dit is logisch, omdat hogere snelheden meer tijd en afstand vereisen om tot stilstand te komen.
Dus auto's die sneller rijden, hebben een langere remafstand nodig om veilig tot stilstand te komen.
"""

# Maak een lijndiagram
# sb.lineplot(data=df, x="dist", y="speed", errorbar=None)
plt.plot(df.index, df['speed'], 'o-', markerfacecolor='lightblue', markersize=5, linewidth=1.5)
plt.title("Evolutie van Snelheid")
plt.xlabel("Index")
plt.ylabel("Snelheid")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

"""
Reflectie:
Het histogram van de remafstand heeft een rechtsscheve verdeling, wat betekent dat de meeste auto's een relatief korte remafstand hebben, terwijl er enkele auto's zijn met een veel langere remafstand. De meeste remafstanden liggen tussen de 20 en 60 voet, maar er zijn enkele uitschieters met langere remafstanden. Dit kan veroorzaakt worden door verschillende factoren, zoals bv. de snelheid van de auto, de staat van de weg, of het type remsysteem.

De correlatiematrix toont een sterke positieve correlatie tussen snelheid en remafstand, wat logisch is omdat hogere snelheden meer tijd en afstand vereisen om tot stilstand te komen. Dit benadrukt het belang van het aanpassen van de snelheid aan de omstandigheden op de weg, aangezien hogere snelheden kunnen leiden tot langere remafstanden en dus een groter risico op ongevallen.

Door deze dataset te analyseren, kan ik beter begrijpen hoe snelheid en remafstand met elkaar samenhangen, wat belangrijk is voor verkeersveiligheid. Hierdoor wordt er duidelijke gemaakt dat het belangrijk is om de snelheid aan te passen aan de omstandigheden, zoals het weer, de staat van de weg, en het verkeer, om zo de remafstand te minimaliseren en de veiligheid te verhogen.
"""