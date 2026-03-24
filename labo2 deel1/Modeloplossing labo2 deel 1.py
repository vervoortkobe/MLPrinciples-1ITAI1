import pandas as pd
import os
import matplotlib.pyplot as plt

print("Current working directory:", os.getcwd())
os.chdir(r'PAD NAAR JE WORKING DIRECTORY')
print(os.getcwd())

# Inlezen van de dataset als pandas DataFrame
df = pd.read_csv('cars.csv')  # Zorg ervoor dat het pad correct is

# Nu is 'df' een tabel (DataFrame) in Python

# Geef nu een tabel weer van je dataset
print(df)

# Toon nu de evolutie van de snelheid door middel van een histogram
plt.figure(figsize=(8, 6))
plt.hist(df['speed'], bins=20, color='skyblue', edgecolor='black')  # Aantal bins kan worden aangepast
plt.xlabel('Snelheid')
plt.ylabel('Frequentie')
plt.title('Evolutie van de snelheid')
plt.show()

# Omdat we met snelheid werken (ruimtelijke dimensie), toon je nu een plot
plt.figure(figsize=(10, 6))
plt.plot(df['speed'], marker='o', linestyle='-', color='skyblue')
plt.xlabel('Index')
plt.ylabel('Snelheid')
plt.title('Evolutie van Snelheid')
plt.grid(True)
plt.show()

# Controleer het datatype van de variabele 'speed'
print("Datatype van 'speed':", df['speed'].dtype)

# Controleer het datatype van de variabele 'distance'
print("Datatype van 'distance':", df['dist'].dtype)

# Controleer op ontbrekende data in de kolom 'speed'
missing_speed = df['speed'].isnull().sum()

# Toon het aantal ontbrekende waarden
print("Aantal ontbrekende waarden in 'speed':", missing_speed)

# Controleer op ontbrekende data in de kolom 'dist'
missing_distance = df['dist'].isnull().sum()

# Toon het aantal ontbrekende waarden
print("Aantal ontbrekende waarden in 'dist':", missing_distance)

# Controleer de minimum- en maximumwaarden van 'speed'
min_speed = df['speed'].min()
max_speed = df['speed'].max()

# Controleer de minimum- en maximumwaarden van 'dist'
min_distance = df['dist'].min()
max_distance = df['dist'].max()

# Toon de resultaten
print("Minimumwaarde van 'speed':", min_speed)
print("Maximumwaarde van 'speed':", max_speed)
print("Minimumwaarde van 'dist':", min_distance)
print("Maximumwaarde van 'dist':", max_distance)

# Maak een histogram van de variabele 'distance'
plt.figure(figsize=(8, 6))
plt.hist(df['dist'], bins=20, color='orange', edgecolor='black')  # Aantal bins kan worden aangepast
plt.xlabel('Afstand')
plt.ylabel('Frequentie')
plt.title('Histogram van de afstand')
plt.show()

