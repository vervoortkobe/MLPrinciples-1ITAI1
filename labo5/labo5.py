# Labo 5
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
current_file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file_path)
os.chdir(script_directory)

# Stap 1: Dataset inladen
import pandas as pd
from sklearn.datasets import fetch_openml
# Laad de dataset in vanuit de UCI Machine Learning Repository
# boston = fetch_openml(data_id=531)
df = pd.read_csv("Boston.csv")
# Converteer de dataset naar een DataFrame
# df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
df = pd.DataFrame(df)

# 2. Verken de dataset
print(df.head(5))
print(df.shape)
print()

print("Ontbrekende waarden in de dataset:")
print(df.isnull().sum())
print()

# 3a. Selecteer invoer- en uitvoervariabelen
X = df[['rm']]
y = df['medv']

# 3b. Split de dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model bouwen
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# 5. Model trainen
model.fit(X_train, y_train)

# 6. Voorspelling maken
y_pred = model.predict(X_test)

# 7. Evalueren van het model (MSE)
from sklearn.metrics import mean_squared_error
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print()
print("Coëfficiënten van het model:")
coeff_df = pd.DataFrame(data=model.coef_, index=X.columns, columns=['Coefficient'])
print(coeff_df)

# 8. Analyseer de resultaten
"""
De verkregen MSE-waarde van 46,144 duidt op een aanzienlijke gemiddelde afwijking tussen de voorspelde en de werkelijke huisprijzen. Dit betekent dus dat de voorspelling van de prijs in de testset afwijkt van de realiteit, wat suggereert dat het model nog niet helemaal nauwkeurig/accuraat is. De positieve coëfficiënt van 9,35 bevestigt echter wel de sterke samenhang tussen het aantal kamers en de prijs; elke extra kamer zorgt voor een aanzienlijke waardestijging. 

Om het model te verbeteren, zouden we extra variabelen kunnen toevoegen, zoals de misdaadscijfers ('crim') of de nabijheid van scholen ('dis'), om een completer beeld van de prijsvorming te krijgen. Daarnaast zou het kunnen helpen om uitschieters in de dataset die de regressielijn nu mogelijk uit balans trekken, te verwijderen. Ten slotte zou het onderzoeken van niet-lineaire verbanden, zoals autocorrelatie, kunnen leiden tot minder fouten en betere voorspellingen.
"""