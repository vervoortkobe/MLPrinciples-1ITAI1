# Deel 2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
current_file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file_path)
os.chdir(script_directory)
 
# Statistieken per categorie:
OECDGas = pd.read_csv('OECDGas.csv')
OECDGas = pd.DataFrame(OECDGas) 
# Identificeer de variabele prijs in de OECDGas dataset.
stats_table = OECDGas.groupby('year')['price'].agg(['mean', 'min', 'max'])
print("Statistieken per jaar (price):")
print(stats_table)
# Bereken en toon de gemiddelde, minimale en maximale waarden per categorie in een overzichtelijke tabel.
# Je kan hiervoor gebruikmaken van de volgende functies in pandas: groupby(), agg() en pivot_table() in pandas.
pivot_stats = OECDGas.pivot_table(values='price', index='year', aggfunc=['mean', 'min', 'max'])
print("\nPivot Table met statistieken:")
print(pivot_stats)

# Boxplot van prijzen per categorie
# Maak een boxplot van de gasprijzen voor elke categorie.
# Voeg labels en een titel toe aan het plot.
sb.boxplot(x='year', y='price', data=OECDGas)
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Boxplot van gasprijzen per jaar')
plt.show()
# Analyseer de boxplot en identificeer eventuele uitschieters.
"""
De medianen van de gasprijzen per jaar liggen allemaal ongeveer rond 0. De spreiding varieert per jaar en we zien veel uitschieters. Hieruit kunnen we dus besluiten dat de gasprijzen per jaar erg verschillen en dat er veel variatie is in de prijzen. Dit duidt op een onstabiele markt of op verschillende factoren die de prijzen beïnvloeden.
"""

# Tijdreeksanalyse
# Converteer de 'year'-kolom naar een datetimeformaat (gebruik de functie pd.to_datetime()).
OECDGas['year'] = pd.to_datetime(OECDGas['year'], format='%Y')
# Maak een tijdreeksplot van de gasprijzen over de jaren.
mean_prices = OECDGas.groupby('year')['price'].mean().reset_index()
mean_prices["year_dt"] = pd.to_datetime(mean_prices["year"], format='%Y')

plt.plot(mean_prices["year_dt"], mean_prices['price'], marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Mean Price')
plt.title('Tijdreeksplot van gemiddelde gasprijzen per jaar')
plt.grid(True)
plt.show()

# Distributie van jaarlijkse veranderingen:
# Bereken de jaarlijkse veranderingen in gasprijzen (gebruik de functie diff()).
mean_prices['price_diff'] = mean_prices['price'].diff()
# Maak een histogram van deze jaarlijkse veranderingen.
plt.hist(mean_prices['price_diff'].dropna(), bins=20, color='blue', edgecolor='black')
plt.xlabel('Jaarlijkse verandering in gasprijzen')
plt.ylabel('Frequentie')
plt.title('Histogram van jaarlijkse veranderingen in gasprijzen')
plt.show()

# Statistieken van specifieke jaarbereiken:
# Selecteer een specifiek jaar- of periodebereik (bijvoorbeeld 2010-2015).
period_data = OECDGas[(OECDGas['year'].dt.year >= 2010) & (OECDGas['year'].dt.year <= 2015)].dropna(subset=['price'])
print("Statistieken voor de periode 2010-2015:")
# Bereken statistieken (gemiddelde, standaarddeviatie, etc.) voor de gasprijzen binnen dat bereik.
period_stats = period_data['price'].agg(['mean', 'std', 'min', 'max', 'count'])
# Presenteer de resultaten in een tabel.
print(period_stats)
"""
Deze statistieken geven allemaal NaN (Not a Number) terug, omdat er geen data beschikbaar is voor deze periode. De beschikbare data loopt van 1960 tot en met 1978.
"""

# Matplotlib en Seaborn:
# Herhaal een van de bovenstaande plots (bijvoorbeeld boxplot) met behulp van Matplotlib en Seaborn in plaats van de standaard plotfuncties.
# Vergelijk de esthetiek en mogelijkheden van Matplotlib en Seaborn met die van de standaard plotfuncties.
plt.figure(figsize=(10, 6))
years = sorted(OECDGas['year'].dt.year.unique())
data = [OECDGas[OECDGas['year'].dt.year == year]['price'].dropna() for year in years]
plt.boxplot(data, tick_labels=years)
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Boxplot van gasprijzen per jaar (Matplotlib)')
plt.xticks(rotation=45)
plt.show()


OECDGas['year'] = OECDGas['year'].dt.year
OECDGas.boxplot(column='price', by='year', grid=False, showfliers=True)
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Boxplot van gasprijzen per jaar (pandas)')
plt.suptitle('')
plt.show()
