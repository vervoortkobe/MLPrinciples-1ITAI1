from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


data_file = Path(__file__).resolve().parent / "ESSBE5.csv"

# 1. Data voorbereiden
data = pd.read_csv(data_file)
data = data.dropna(subset=["trstplc"])

pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),  # vervang ontbrekende waarden door het gemiddelde
        ("regressor", LinearRegression()),  # lineair regressiemodel
    ]
)

# 2. Selecteer relevante variabelen
# De doelvariabele hoort niet in de input-features, anders lekt het juiste antwoord in het model.
selected_features = ["agea", "female", "hincfel", "plcpvcr"]
X = data[selected_features]
y = data["trstplc"]

# 3. Splits de dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train het model en evalueer het
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# 5. Analyse
importance = pipeline.named_steps["regressor"].coef_
print("\nModel Coëfficiënten:")
for feature, coef in zip(selected_features, importance):
    print(f"{feature}: {coef:.4f}")
