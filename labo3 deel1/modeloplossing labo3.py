import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Inlezen van de dataset als pandas DataFrame
df = pd.read_csv('austpop.csv')  # Zorg ervoor dat het pad correct wijst naar uw huidige folder waarin austpop.csv staat

# Stap 2: Geef een tabel weer van je dataset
print("Tabel van de dataset:")
print(df.head())

# Stap 3: Genereer 2 plots over de bevolkingsdata in SA en ACT
plt.figure(figsize=(12, 6))

# Plot voor SA
plt.subplot(1, 2, 1)
sns.lineplot(x='year', y='SA', data=df)
plt.title('Bevolkingsdata in SA over de jaren')

# Plot voor ACT
plt.subplot(1, 2, 2)
sns.lineplot(x='year', y='ACT', data=df)
plt.title('Bevolkingsdata in ACT over de jaren')

plt.tight_layout()
plt.show()

# Stap 4: Wat is het datatype van de data in uw plot?
data_type_sa = df['SA'].dtype
data_type_act = df['ACT'].dtype
print("\nDatatype van bevolkingsdata in SA:", data_type_sa)
print("Datatype van bevolkingsdata in ACT:", data_type_act)

# Stap 5: Kijk naar de verdeling in uw plot rond gegevens van de staat ACT
plt.figure(figsize=(10, 6))
sns.histplot(df['ACT'], bins=20, kde=True)
plt.title('Verdeling van bevolkingsdata in ACT')
plt.xlabel('Bevolking')
plt.ylabel('Frequentie')
plt.show()

# Stap 7: Zijn er limieten waartussen de waarden zich bevinden in uw plot? Zo ja, welke?
lower_limit_act = df['ACT'].min()
upper_limit_act = df['ACT'].max()
print("\nOnderste limiet van bevolkingsdata in ACT:", lower_limit_act)
print("Bovenste limiet van bevolkingsdata in ACT:", upper_limit_act)

# Stap 8: Wat zijn de modi in beide figuren?
mode_sa = df['SA'].mode().values
mode_act = df['ACT'].mode().values
print("\nModus van bevolkingsdata in SA:", mode_sa)
print("Modus van bevolkingsdata in ACT:", mode_act)

# Stap 9: Bereken het gemiddelde van de bevolkingsdata voor elke staat over alle jaren
state_means = df[['NSW', 'Vic', 'Qld', 'SA', 'WA', 'Tas', 'NT', 'ACT']].mean()

# Plot de gemiddelde bevolking per staat
plt.figure(figsize=(10, 6))
state_means.plot(kind='bar', color='skyblue')
plt.title('Gemiddelde bevolking per staat')
plt.xlabel('Staat')
plt.ylabel('Gemiddelde bevolking')
plt.xticks(rotation=45)
plt.show()

# Stap 10: Bereken de jaarlijkse groeipercentages van de bevolking voor elke staat
df_growth = df[['NSW', 'Vic', 'Qld', 'SA', 'WA', 'Tas', 'NT', 'ACT']].pct_change() * 100

# Plot de jaarlijkse groeipercentages voor elke staat in afzonderlijke lijngrafieken
plt.figure(figsize=(12, 8))
for state in ['NSW', 'Vic', 'Qld', 'SA', 'WA', 'Tas', 'NT', 'ACT']:
    plt.plot(df['year'], df_growth[state], label=state)

plt.title('Jaarlijkse groeipercentage van de bevolking per staat')
plt.xlabel('Jaar')
plt.ylabel('Groeipercentage')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()




