# Labo 3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
current_file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file_path)
os.chdir(script_directory)

# Opdracht
# 1. Download de dataset

# 2. Lees de dataset in door middel van Pandas. Dit zorgt ervoor dat je dataset nu een tabel (DataFrame) in Python is.
df = pd.DataFrame(pd.read_csv("austpop.csv"))

# 3. Geef nu een tabel weer van je dataset. Zorg voor een titel en de juiste benamingen voor de assen
print("Tabel van de dataset:")
print(df.head(5))
print()

# 4. Genereer 2 plots over de bevolkingsdata in SA en ACT. Analyseer de gegevens, wat zijn de verschillen? Wat kan je concluderen?
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sb.lineplot(data=df, x="year", y="SA")
plt.title("Bevolkingsdata in SA over de jaren")
plt.xlabel("year")
plt.ylabel("SA")

plt.subplot(1, 2, 2)
sb.lineplot(data=df, x="year", y="ACT")
plt.title("Bevolkingsdata in ACT over de jaren")
plt.xlabel("year")
plt.ylabel("ACT")
plt.tight_layout()
plt.show()

"""
De bevolkingsgroei in SA verloopt geleidelijk en redelijk constant over de jaren heen. In ACT begint de bevolking veel kleiner, maar groeit ze vanaf 1950 zeer snel. Terwijl SA een lineair groeipatroon toont, vertoont ACT een duidelijke exponentiële stijging. Dit suggereert dat SA al eerder ontwikkeld en bevolkt was, terwijl ACT pas later een snelle uitbreiding kende. De sterke groei in ACT hangt waarschijnlijk samen met de ontwikkeling van Canberra (in plaats van Melbourne) als nationaal bestuurscentrum van het Australian Capital Territory.
"""

# 5. Wat is het datatype van de data in uw plot? Gebruik hiervoor de attribuut {dtype}
print("Datatype van de data in de plot:")
data_type_sa = df['SA'].dtype
data_type_act = df['ACT'].dtype
print(f"SA: {data_type_sa}")
print(f"ACT: {data_type_act}")
print()

# 6. Genereer een plot met de verdeling van de bevolkingsdata in ACT. Kijk naar de verdeling in uw plot
# rond gegevens van de staat ACT. Volgt dit volgens jou de normale verdeling? Normaal verdeelde data
# heeft de volgende eigenschappen: observaties rond het gemiddelde zijn het waarschijnlijkst. Hoe verder
# waardes van het gemiddelde af liggen, hoe onwaarschijnlijker het is deze waarden te observeren.
plt.figure(figsize=(8, 6))
sb.histplot(df['ACT'], kde=True, bins=20)
plt.title("Verdeling van bevolkingsdata in ACT")
plt.xlabel("ACT")
plt.ylabel("Frequentie")
plt.show()

"""
Nee, de verdeling van de bevolkingsdata in ACT volgt geen normale verdeling.  
Het plot toont een sterke rechtsscheve verdeling met een piek bij lage waarden rond 0 en een lange staart naar hoge waarden richting de 350. Bij een normale verdeling zou de piek rond het gemiddelde moeten staan met symmetrische afnames aan beide zijden, wat hier ontbreekt. 
Wat vooral opvalt is dat de lage waarden domineren, terwijl hogere waarden zeldzamer zijn; dit past bij exponentiële groei.
"""

# 7. Zijn er limieten waartussen de waarden zich bevinden in uw plot? Zo ja, welke?
print("Limieten van de waarden in het plot:")
print(f"Min: {df['ACT'].min()}")
print(f"Mean: {df['ACT'].mean()}")
print(f"Max: {df['ACT'].max()}")
print()

# 8. Wat zijn de modi in beide figuren? Een modus is de waarde die het meeste voorkomt.
mode_sa = df['SA'].mode().values
mode_act = df['ACT'].mode().values
print("Modi in de figuren:")
print(f"SA: {mode_sa}")
print(f"ACT: {mode_act}")
print()

# 9. Bereken het gemiddelde van de bevolkingsdata voor elke staat over alle jaren. Plot vervolgens de gemiddelde bevolking per staat.
states = [col for col in df.columns if col.lower() != 'year' and col.lower() != 'aust' and col.lower() != 'rownames']
mean_values = [df[col].mean() for col in states]
print("Gemiddelde bevolking per staat:")
for state, mean in zip(states, mean_values):
    print(f"{state}: {mean:.2f}")
print()

plt.figure(figsize=(8, 6))
sb.barplot(x=states, y=mean_values, color='lightskyblue', gap=0.3)
plt.title("Gemiddelde bevolking per staat")
plt.xlabel("Staat")
plt.ylabel("Gemiddelde bevolking")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 10. Bereken de jaarlijkse groeipercentages van de bevolking voor elke staat. Plot vervolgens de jaarlijkse groeipercentages in lijngrafieken.
plt.figure(figsize=(12, 7))
df_growth = df.copy()
df_growth[states] = df[states].pct_change() * 100
for state in states:
    sb.lineplot(x=df['year'], y=df_growth[state], label=state)

plt.title("Jaarlijkse groeipercentage van de bevolking per staat")
plt.xlabel("Jaar")
plt.ylabel("Groeipercentage")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()