# Deel 1
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
current_file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file_path)
os.chdir(script_directory)

#  Download de dataset en importeer deze
df = pd.DataFrame(pd.read_csv("penguins_bayes.csv", index_col=0))

# Schrijf code om een samenvatting van je dataset te tonen
print(df.describe())

# Sorteer de dataset op snavellengte
df = df.sort_values(by="bill_length_mm")

# Bereken de gemiddelde snavellengte gegroepeerd op geslacht
mean_bills = df.groupby("sex")["bill_length_mm"].mean()

# Maak een histogram van de snavellengte per geslacht
df.hist(column="bill_length_mm", by="sex", bins=20)
plt.suptitle("Histogram van snavellengte per geslacht")
plt.show()