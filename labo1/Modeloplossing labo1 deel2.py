import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Bepaal het absolute pad naar het huidige Python-bestand
current_file_path = os.path.abspath(__file__)
# Haal de directory op waarin het Python-bestand zich bevindt
script_directory = os.path.dirname(current_file_path)
# Stel de working directory in op de directory van het Python-bestand
os.chdir(script_directory)

# -----------------------------------------------------------
# Statistieken per categorie
# -----------------------------------------------------------

# Laad de OECDGas dataset (zorg dat OECDGas.csv beschikbaar is in je folder)
OECDGas = pd.read_csv('OECDGas.csv')

# Converteer naar een pandas DataFrame (optioneel als het al een DataFrame is)
OECDGas = pd.DataFrame(OECDGas)

# Identificeer de variabele 'price'
# (We gaan er voor dit voorbeeld van uit dat 'year' de categorisatievariabele is)
# Bereken en toon de gemiddelde, minimale en maximale waarden per jaar
stats_table = OECDGas.groupby('year')['price'].agg(['mean', 'min', 'max'])
print("Statistieken per jaar (price):")
print(stats_table)

# Alternatief: gebruik een pivot_table om dezelfde statistieken in één tabel weer te geven
pivot_stats = OECDGas.pivot_table(values='price', index='year', aggfunc=['mean', 'min', 'max'])
print("\nPivot Table met statistieken:")
print(pivot_stats)

# -----------------------------------------------------------
# Boxplot van prijzen per categorie
# -----------------------------------------------------------
# Maak een boxplot van de gasprijzen voor elke categorie (in dit geval per jaar)
plt.figure(figsize=(10,6))
sns.boxplot(x='year', y='price', data=OECDGas)
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Boxplot van gasprijzen per jaar')
plt.show()
# Analyseer de boxplot visueel op eventuele uitschieters.

# -----------------------------------------------------------
# Tijdreeksanalyse
# -----------------------------------------------------------
# Converteer de 'year'-kolom naar een datetimeformaat
OECDGas['year_dt'] = pd.to_datetime(OECDGas['year'], format='%Y')

# Bereken de gemiddelde prijs per jaar
mean_prices = OECDGas.groupby('year')['price'].mean().reset_index()
# Converteer de 'year' kolom in mean_prices naar datetime
mean_prices['year_dt'] = pd.to_datetime(mean_prices['year'], format='%Y')

# Maak een tijdreeksplot van de gemiddelde gasprijzen per jaar
plt.figure(figsize=(10,6))
plt.plot(mean_prices['year_dt'], mean_prices['price'], marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Mean Price')
plt.title('Tijdreeksplot van gemiddelde gasprijzen per jaar')
plt.grid(True)
plt.show()

# -----------------------------------------------------------
# Distributie van jaarlijkse veranderingen
# -----------------------------------------------------------
# Bereken de jaarlijkse veranderingen in de gemiddelde gasprijs met diff()
mean_prices['price_change'] = mean_prices['price'].diff()

# Maak een histogram van de jaarlijkse veranderingen (verwijder de eerste NA-waarde)
plt.figure(figsize=(10,6))
plt.hist(mean_prices['price_change'].dropna(), bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Jaarlijkse verandering in prijs')
plt.ylabel('Frequentie')
plt.title('Histogram van jaarlijkse veranderingen in gasprijs')
plt.show()

# -----------------------------------------------------------
# Statistieken van specifieke jaarbereiken
# -----------------------------------------------------------
# Selecteer een specifiek jaar- of periodebereik, bijvoorbeeld 2010-2015
# (Controleer of deze jaren in de dataset voorkomen; pas zonodig de periode aan)
subset_period = OECDGas[(OECDGas['year'] >= 2010) & (OECDGas['year'] <= 2015)]
period_stats = subset_period['price'].agg(['mean', 'std', 'min', 'max'])
print("Statistieken voor de periode 2010-2015:")
print(period_stats)

# -----------------------------------------------------------
# Matplotlib en Seaborn: Boxplot herhalen
# -----------------------------------------------------------
# Boxplot met Matplotlib
plt.figure(figsize=(10,6))
years = sorted(OECDGas['year'].unique())
data_list = [OECDGas[OECDGas['year'] == year]['price'] for year in years]
plt.boxplot(data_list, labels=years)
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Boxplot van gasprijzen per jaar (Matplotlib)')
plt.show()

# Boxplot met Seaborn
plt.figure(figsize=(10,6))
sns.boxplot(x='year', y='price', data=OECDGas)
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Boxplot van gasprijzen per jaar (Seaborn)')
plt.show()
