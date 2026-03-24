import pandas as pd
import os
import matplotlib.pyplot as plt

current_file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file_path)
os.chdir(script_directory)


# Gegevens inladen (ga ervan uit dat 'OECDGas.csv' het CSV-bestand is)
OECDGas = pd.read_csv('OECDGas.csv')

# Converteer naar een pandas DataFrame (optioneel als het al een DataFrame is)
OECDGas = pd.DataFrame(OECDGas)

# Toon een overzicht van de dataset
summary = OECDGas.describe()

# Toon de samenvatting
print(summary)

# Sorteer de dataset op jaar
OECDGas = OECDGas.sort_values(by='year')

# Bereken het gemiddelde van de prijs gegroepeerd op jaar
mean_prices = OECDGas.groupby('year')['price'].mean()

# Maak een lijndiagram van de gemiddelde prijzen per jaar
plt.plot(mean_prices.index, mean_prices.values, marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Mean Price')
plt.title('Mean Gas Prices Over the Years')
plt.grid(True)
plt.show()