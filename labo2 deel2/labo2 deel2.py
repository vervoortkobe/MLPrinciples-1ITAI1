# Labo/Project 2 - Uitgebreide Data-analyse en Interpretatie
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

# Identificeer de 3 variabelen met de hoogste absolute correlatie met "medv"
# Bereken de top 3 sterkst gecorreleerde variabelen (positief of negatief).
correlation_matrix = df.corr()
medv_corr = correlation_matrix['medv'].abs().sort_values(ascending=False)
top_3_features = medv_corr.index[1:4]
print(top_3_features)

plt.figure(figsize=(10, 6))
sb.heatmap(correlation_matrix[top_3_features].iloc[1:], annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlatiematrix van de top 3 variabelen")
plt.tight_layout()
plt.show()

# Geef een korte interpretatie: wat betekent dit in de context van huizenprijzen (portfolio)?
"""
We kunnen aflezen uit de correlatiematrix van de top 3 variabelen dat deze variabelen een sterke relatie hebben met de huizenprijzen (medv). 
Deze huizenprijzen (medv) en de variabelen "lstat", "rm" en "ptratio" zijn dus sterk gecorreleerd, wat betekent dat veranderingen in deze variabelen waarschijnlijk een aanzienlijke invloed hebben op de huizenprijzen.
- "lstat" (percentage van lagere status bevolking) heeft een negatieve correlatie (-0.74), wat suggereert dat een hogere sociale status geassocieerd is met lagere huizenprijzen.
- "rm" (gemiddeld aantal kamers per woning) heeft een positieve correlatie (0.70), wat betekent dat meer kamers per woning geassocieerd zijn met hogere huizenprijzen.
- "ptratio" (student-leraar ratio) heeft ook een negatieve correlatie (-0.51), wat suggereert dat hogere student-leraar ratio's geassocieerd zijn met lagere huizenprijzen.
"""

# Maak scatterplots van medv tegen de 2 meest significante variabelen.
# Gebruik Seaborn’s regplot om een regressielijn toe te voegen.
# Voeg titels en labels toe.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sb.regplot(x='lstat', y='medv', data=df, scatter_kws={'alpha':0.5}, line_kws={'color': 'red', 'linewidth': 2},)
plt.title("Regressieplot: lstat vs medv")
plt.xlabel("lstat")
plt.ylabel("medv")

plt.subplot(1, 2, 2)
sb.regplot(x='rm', y='medv', data=df, scatter_kws={'alpha':0.5}, line_kws={'color': 'red', 'linewidth': 2}, )
plt.title("Regressieplot: rm vs medv")
plt.xlabel("rm")
plt.ylabel("medv")
plt.tight_layout()
plt.show()

# Schrijf kort op wat je observeert in de trends (portfolio).
"""
In de lstat vs medv plot toont de regressielijn een sterke negatieve lineaire relatie: hogere percentages lager status bevolking correleren met lagere huizenprijzen, met punten redelijk rond de lijn.
Voor rm vs medv is de relatie positief lineair: meer kamers leiden tot hogere prijzen, maar met spreiding en niet-lineariteit bij hoge waarden.
Beide plots bevestigen lineaire trends zonder duidelijke kromming, dus lineaire regressie lijkt geschikt.
"""

# Onderzoek een mogelijke niet-lineaire relatie
# Kies de variabele met de hoogste niet-lineaire trend in de scatterplots.
# Pas een log-transformatie of kwadratische transformatie toe op deze variabele.
# Maak opnieuw een scatterplot met de getransformeerde variabele en ‘medv’.
df['log_lstat'] = np.log(df['lstat'])
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sb.regplot(x='lstat', y='medv', data=df, scatter_kws={'alpha':0.5}, line_kws={'color': 'red', 'linewidth': 2},)
plt.title("Oorspronkelijke data: lstat vs medv")
plt.xlabel("lstat")
plt.ylabel("medv")

plt.subplot(1, 2, 2)
sb.regplot(x='log_lstat', y='medv', data=df, scatter_kws={'alpha':0.5}, line_kws={'color': 'red', 'linewidth': 2},)
plt.title("Na log-transformatie: log(lstat) vs medv")
plt.xlabel("log(lstat)")
plt.ylabel("medv")
plt.tight_layout()
plt.show()

# Beantwoord de vraag: is de relatie na transformatie duidelijker?
"""
Ja, na de log-transformatie van lstat is de relatie met medv duidelijker en lineairder. De punten liggen dichter bij de regressielijn, wat suggereert dat de log-transformatie de niet-lineariteit heeft verminderd en een sterkere lineaire relatie heeft onthuld tussen lstat en medv.
Dit komt doordat de log-transformatie de invloed van extreme waarden heeft verminderd, waardoor de algemene trend beter zichtbaar is.
Log-transformatie vermindert de invloed van extreme waarden (uitschieters) en maakt de relatie meer lineair, zodat een regressiemodel beter kan passen.
"""

# Kwadratische transformatie
df['quad_lstat'] = np.power(df['lstat'], 2)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sb.regplot(x='lstat', y='medv', data=df, scatter_kws={'alpha':0.5}, line_kws={'color': 'red', 'linewidth': 2},)
plt.title("Oorspronkelijke data: lstat vs medv")
plt.xlabel("lstat")
plt.ylabel("medv")

plt.subplot(1, 2, 2)
sb.regplot(x='quad_lstat', y='medv', data=df, scatter_kws={'alpha':0.5}, line_kws={'color': 'red', 'linewidth': 2},)
plt.title("Na kwadratische transformatie: lstat^2 vs medv")
plt.xlabel("lstat^2")
plt.ylabel("medv")
plt.tight_layout()
plt.show()

# Beantwoord de vraag: is de relatie na transformatie duidelijker?
"""
Nee, we zien bij de kwadratische transformatie dat de relatie tussen lstat^2 en medv nog steeds niet lineair is, en er is meer spreiding in de punten rond de regressielijn. Dit komt doordat de kwadratische transformatie de niet-lineariteit heeft versterkt in plaats van verminderd, waardoor de relatie tussen lstat^2 en medv minder duidelijk is.
Kwadratische transformatie versterkt de verschillen tussen hoge en lage waarden en kan helpen om een betere voorspelling te maken als een variabele een gebogen effect heeft op ‘medv’.
"""