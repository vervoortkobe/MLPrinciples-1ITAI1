import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Lees de dataset in
data = pd.read_csv('ESSBE5.csv')

# Verken de structuur van de dataset
print("Informatie over de dataset:")
print(data.info())

# Behandel ontbrekende waarden en dubbele rijen
# We controleren op ontbrekende waarden en duplicaten en behandelen deze indien nodig

# Selecteer relevante variabelen
selected_features = ['agea', 'female', 'hincfel', 'plcpvcr', 'trstplc']
data = data[selected_features]

# Verwijder rijen met NaN-waarden in de target variabele (y)
data.dropna(subset=['trstplc'], inplace=True)

# Split de dataset in features (X) en target variable (y)
X = data.drop(columns=['trstplc']) # features
y = data['trstplc'] # target variable

# Split de dataset in een trainingsset en een testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definieer de pipeline met een imputer en het lineaire regressiemodel
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), # vervang ontbrekende waarden door het gemiddelde van de kolom
    ('regressor', LinearRegression()) # lineaire regressie model
])

# Modeltraining
pipeline.fit(X_train, y_train)

# Model evaluatie
# Evalueer het getrainde model op de testset
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Interpretatie
# Analyseer de resultaten en interpretaties van het model
# Bekijk de coëfficiënten van het model om de impact van elke variabele te begrijpen
coefficients = pd.DataFrame(pipeline.named_steps['regressor'].coef_, X.columns, columns=['Coefficient'])
print("Coëfficiënten van het model:")
print(coefficients)


# Analyse:
# Het resultaat van de Mean Squared Error (MSE) geeft aan hoe ver de voorspelde waarden van het model afwijken 
# van de werkelijke waarden. In dit geval is de MSE ongeveer 3.122. Een lage MSE geeft aan dat het model goede voorspellingen maakt, 
# terwijl een hoge MSE aangeeft dat het model slechte voorspellingen maakt.

# De coëfficiënten van het model geven de invloed weer van elke variabele op de voorspelde waarde. 
# Een positieve coëfficiënt betekent dat een toename in die variabele gepaard gaat met een toename in de voorspelde waarde, 
# terwijl een negatieve coëfficiënt betekent dat een toename in die variabele gepaard gaat met een afname in de voorspelde waarde.

# In dit specifieke geval zien we dat de variabele 'agea' een negatieve coëfficiënt heeft, 
# wat betekent dat een hogere leeftijd geassocieerd is met een lagere voorspelde waarde van het vertrouwen in de politie. 
# De variabele 'female' heeft ook een negatieve coëfficiënt, wat suggereert dat vrouwelijke respondenten gemiddeld een lager 
# vertrouwen hebben dan mannelijke respondenten. De variabele 'hincfel' heeft de grootste negatieve invloed op het model, 
# wat betekent dat respondenten die aangeven dat ze het moeilijk hebben met hun huidige inkomen gemiddeld een veel lager 
# vertrouwen hebben in de politie. Aan de andere kant heeft de variabele 'plcpvcr' een positieve invloed, wat aangeeft dat een 
# hogere waardering voor het vermogen van de politie om misdaden te voorkomen, gepaard gaat met een hoger vertrouwen in de politie.

