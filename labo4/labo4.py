# Labo 4
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
current_file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file_path)
os.chdir(script_directory)

# 1. Observeer op de website hoeveel rijen deze dataset heeft, alsook hoeveel kolommen (variabelen). Voor een beter begrip raad ik aan om de documentatie van de dataset te bekijken.
"""
1704 rows, 10 columns
essround, edition, idno, cntry, trstplc, agea, female, eduyrs, hincfel, plcpvcr
"""

# 2. Inladen dataset: download de dataset en lees deze in in uw Python bestand door de gebruikelijke manier. Als je Pandas gebruikt bij het inlezen van de data, wordt je data meteen naar een DataFrame omgezet, wat handig is om straks bewerkingen uit te voeren op je data.
df = pd.read_csv("ESSBE5.csv")

# 3. Verken de dataset:
# - Geef een samenvatting weer van de dataset met de eerste en laatste set rijen.
print("Samenvatting van de dataset:")
print(df.head(5))
print(df.tail(5))
# - Geef weer hoeveel rijen en kolommen (variabelen) de dataset bevat.
print("Aantal rijen en kolommen:")
print(df.shape)

# Voer nu de DIPKEVER analyse uit op elke variabele van de dataset. Kijk in de cursus naar de criteria van deze analyse.
# Controleer op ontbrekende waarden
print("\nOverzicht van datatypes en ontbrekende waarden:")
print(df.info())

print("\nOntbrekende waarden per kolom:")
missing_values = df.isnull().sum()
print(missing_values)

"""
Variabele: rownames
Duidelijkheid: De variabele 'rownames' is een index variabele die de rijnummers aangeeft.
Informatief: De variabele 'rownames' heeft een relatief hoge informatie-inhoud met 1704 unieke waarden op 1704 observaties.
Privacy: De variabele 'rownames' lijkt op het eerste gezicht geen directe privacygevoelige informatie te bevatten die individuen identificeerbaar maakt, mits geaggregeerde analyse.
Kwaliteit: De variabele 'rownames' is van type 'int64'. Dit is een numeriek type, wat goed is voor kwantitatieve analyses. Er zijn geen directe aanwijzingen voor technische problemen.
Ethiek: De variabele 'rownames' lijkt op het eerste gezicht geen specifieke ethische problemen op te werpen, mits analyses op een neutrale en respectvolle manier worden uitgevoerd.
Voorradig: De variabele 'rownames' is volledig voorradig; er zijn geen ontbrekende waarden.
Echtheid: Gezien de bron (ESS) wordt aangenomen dat de data voor 'rownames' betrouwbaar en een redelijke afspiegeling van de realiteit zijn, maar dit moet in de context van de survey-methodologie worden bekeken.
Relevantie: De variabele 'rownames' is waarschijnlijk niet direct relevant voor de kern van de probleemstelling (vertrouwen in Belgische politie), maar kan nuttig zijn voor metadata of data management.
Conclusie: Dit is een indexvariabele en geen inhoudelijke variabele. Ze heeft veel unieke waarden, maar precies daarom is ze niet informatief voor inhoudelijke analyse. Omdat het een technische rij-identificatie is, hoort ze niet in een ML-model. Voor privacy is het risico beperkt, maar analytisch en functioneel is ze overbodig.

Variabele: essround
Duidelijkheid: De variabele 'essround' geeft de survey ronde aan, in dit geval ESS ronde 5.
Informatief: De variabele 'essround' heeft een beperkte informatie-inhoud met slechts 1 unieke waarde. Controleer of dit voldoende is voor analyse.
Privacy: De variabele 'essround' lijkt op het eerste gezicht geen directe privacygevoelige informatie te bevatten die individuen identificeerbaar maakt, mits geaggregeerde analyse.
Kwaliteit: De variabele 'essround' is van type 'int64'. Dit is een numeriek type, wat goed is voor kwantitatieve analyses. Er zijn geen directe aanwijzingen voor technische problemen.
Ethiek: De variabele 'essround' lijkt op het eerste gezicht geen specifieke ethische problemen op te werpen, mits analyses op een neutrale en respectvolle manier worden uitgevoerd.
Voorradig: De variabele 'essround' is volledig voorradig; er zijn geen ontbrekende waarden.
Echtheid: Gezien de bron (ESS) wordt aangenomen dat de data voor 'essround' betrouwbaar en een redelijke afspiegeling van de realiteit zijn, maar dit moet in de context van de survey-methodologie worden bekeken.
Relevantie: De variabele 'essround' is waarschijnlijk niet direct relevant voor de kern van de probleemstelling (vertrouwen in Belgische politie), maar kan nuttig zijn voor metadata of data management.
Conclusie: Deze variabele is constant in deze dataset, dus ze voegt niets toe aan voorspelling of interpretatie. Ze is wel volledig aanwezig en technisch correct. Omdat er slechts één unieke waarde is, heeft ze geen echte verklarende kracht. Je laat deze best weg.

Variabele: edition
Duidelijkheid: De variabele 'edition' is over het algemeen duidelijk te interpreteren. Dit kan een editienummer van de dataset zijn.
Informatief: De variabele 'edition' heeft een beperkte informatie-inhoud met slechts 1 unieke waarde. Controleer of dit voldoende is voor analyse.
Privacy: De variabele 'edition' lijkt op het eerste gezicht geen directe privacygevoelige informatie te bevatten die individuen identificeerbaar maakt, mits geaggregeerde analyse.
Kwaliteit: De variabele 'edition' is van type 'float64'. Dit is een numeriek type, wat goed is voor kwantitatieve analyses. Er zijn geen directe aanwijzingen voor technische problemen.
Ethiek: De variabele 'edition' lijkt op het eerste gezicht geen specifieke ethische problemen op te werpen, mits analyses op een neutrale en respectvolle manier worden uitgevoerd.
Voorradig: De variabele 'edition' is volledig voorradig; er zijn geen ontbrekende waarden.
Echtheid: Gezien de bron (ESS) wordt aangenomen dat de data voor 'edition' betrouwbaar en een redelijke afspiegeling van de realiteit zijn, maar dit moet in de context van de survey-methodologie worden bekeken.
Relevantie: De variabele 'edition' is waarschijnlijk niet direct relevant voor de kern van de probleemstelling (vertrouwen in Belgische politie), maar kan nuttig zijn voor metadata of data management.
Conclusie: Ook dit is vermoedelijk metadata over de datasetversie. Omdat de waarde overal gelijk is, heeft ze geen analytische meerwaarde. Ze is technisch in orde, maar niet relevant voor de onderzoeksvraag. Voor ML is dit dus geen bruikbare feature.

Variabele: idno
Duidelijkheid: De variabele 'idno' is over het algemeen duidelijk te interpreteren. Unieke identificatiecode voor elke respondent.
Informatief: De variabele 'idno' heeft een relatief hoge informatie-inhoud met 1704 unieke waarden op 1704 observaties.
Privacy: De variabele 'idno' bevat unieke identifiers. Deze moet zorgvuldig behandeld worden om privacy van individuen te waarborgen en mag niet direct gedeeld worden in analyses die individuen identificeerbaar maken.
Kwaliteit: De variabele 'idno' is van type 'int64'. Dit is een numeriek type, wat goed is voor kwantitatieve analyses. Er zijn geen directe aanwijzingen voor technische problemen.
Ethiek: De variabele 'idno' lijkt op het eerste gezicht geen specifieke ethische problemen op te werpen, mits analyses op een neutrale en respectvolle manier worden uitgevoerd.
Voorradig: De variabele 'idno' is volledig voorradig; er zijn geen ontbrekende waarden.
Echtheid: Gezien de bron (ESS) wordt aangenomen dat de data voor 'idno' betrouwbaar en een redelijke afspiegeling van de realiteit zijn, maar dit moet in de context van de survey-methodologie worden bekeken.
Relevantie: De variabele 'idno' is waarschijnlijk niet direct relevant voor de kern van de probleemstelling (vertrouwen in Belgische politie), maar kan nuttig zijn voor metadata of data management.
Conclusie: Dit is een unieke identificator per respondent. Zo’n variabele mag je niet als feature gebruiken, omdat ze niets inhoudelijks verklaart en omdat ze direct naar personen kan verwijzen binnen de dataset. Vanuit privacy en ethiek is dit het duidelijkste geval van uitsluiten. Bewaren kan als sleutelveld in een aparte veilige tabel, maar niet in het model.

Variabele: cntry
Duidelijkheid: De variabele 'cntry' is over het algemeen duidelijk te interpreteren. Landcode, in dit geval 'BE' voor België.
Informatief: De variabele 'cntry' heeft een beperkte informatie-inhoud met slechts 1 unieke waarde. Controleer of dit voldoende is voor analyse.
Privacy: De variabele 'cntry' lijkt op het eerste gezicht geen directe privacygevoelige informatie te bevatten die individuen identificeerbaar maakt, mits geaggregeerde analyse.
Kwaliteit: De variabele 'cntry' is van type 'object'. Dit is een categorisch type. De kwaliteit is voldoende voor verdere analyse.
Ethiek: Variabelen zoals 'cntry' kunnen gevoelig zijn in ethische overwegingen, met name bij het trekken van conclusies die kunnen leiden tot stereotypering of discriminatie. Het is cruciaal om resultaten met deze variabelen zorgvuldig te interpreteren en eventuele biases te minimaliseren.
Voorradig: De variabele 'cntry' is volledig voorradig; er zijn geen ontbrekende waarden.
Echtheid: Gezien de bron (ESS) wordt aangenomen dat de data voor 'cntry' betrouwbaar en een redelijke afspiegeling van de realiteit zijn, maar dit moet in de context van de survey-methodologie worden bekeken.
Relevantie: De variabele 'cntry' is zeer waarschijnlijk relevant voor het onderzoeken van het vertrouwen in de Belgische politie, hetzij direct, hetzij als contextuele variabele.
Conclusie: In deze dataset is dit gewoon België, dus inhoudelijk draagt de variabele niets bij aan differentiatie tussen respondenten. Voor een landenspecifieke analyse is ze als context nuttig, maar hier is ze constant. Daardoor is ze niet relevant als predictor. Voor ML zou ze alleen zin hebben in een multi-country dataset.

Variabele: trstplc
Duidelijkheid: De variabele 'trstplc' is over het algemeen duidelijk te interpreteren. Vertrouwen in de politie, geschaald van 0 (helemaal geen vertrouwen) tot 10 (volledig vertrouwen).
Informatief: De variabele 'trstplc' lijkt voldoende informatief te zijn met 11 unieke waarden.
Privacy: De variabele 'trstplc' lijkt op het eerste gezicht geen directe privacygevoelige informatie te bevatten die individuen identificeerbaar maakt, mits geaggregeerde analyse.
Kwaliteit: De variabele 'trstplc' is van type 'float64'. Dit is een numeriek type, wat goed is voor kwantitatieve analyses. Er zijn geen directe aanwijzingen voor technische problemen.
Ethiek: De variabele 'trstplc' lijkt op het eerste gezicht geen specifieke ethische problemen op te werpen, mits analyses op een neutrale en respectvolle manier worden uitgevoerd.
Voorradig: De variabele 'trstplc' heeft 3 ontbrekende waarden (0.18%). Dit is een beheersbaar aantal dat geïmputeerd of verwijderd kan worden.
Echtheid: Gezien de bron (ESS) wordt aangenomen dat de data voor 'trstplc' betrouwbaar en een redelijke afspiegeling van de realiteit zijn, maar dit moet in de context van de survey-methodologie worden bekeken.
Relevantie: De variabele 'trstplc' is zeer waarschijnlijk relevant voor het onderzoeken van het vertrouwen in de Belgische politie, hetzij direct, hetzij als contextuele variabele.
Conclusie: Dit is de belangrijkste variabele in de analyse: vertrouwen in de politie. De schaal is duidelijk, de informatie-inhoud is voldoende groot en de missing rate is laag. Ethisch is dit normaal verantwoord, zolang je geen stigmatiserende conclusies trekt over groepen. Dit is een logische uitkomstvariabele.

Variabele: agea
Duidelijkheid: De variabele 'agea' is over het algemeen duidelijk te interpreteren. Leeftijd van de respondent in jaren.
Informatief: De variabele 'agea' heeft een relatief hoge informatie-inhoud met 73 unieke waarden op 1704 observaties.
Privacy: De variabele 'agea' lijkt op het eerste gezicht geen directe privacygevoelige informatie te bevatten die individuen identificeerbaar maakt, mits geaggregeerde analyse.
Kwaliteit: De variabele 'agea' is van type 'int64'. Dit is een numeriek type, wat goed is voor kwantitatieve analyses. Er zijn geen directe aanwijzingen voor technische problemen.
Ethiek: Variabelen zoals 'agea' kunnen gevoelig zijn in ethische overwegingen, met name bij het trekken van conclusies die kunnen leiden tot stereotypering of discriminatie. Het is cruciaal om resultaten met deze variabelen zorgvuldig te interpreteren en eventuele biases te minimaliseren.
Voorradig: De variabele 'agea' is volledig voorradig; er zijn geen ontbrekende waarden.
Echtheid: Gezien de bron (ESS) wordt aangenomen dat de data voor 'agea' betrouwbaar en een redelijke afspiegeling van de realiteit zijn, maar dit moet in de context van de survey-methodologie worden bekeken.
Relevantie: De variabele 'agea' is zeer waarschijnlijk relevant voor het onderzoeken van het vertrouwen in de Belgische politie, hetzij direct, hetzij als contextuele variabele.
Conclusie: Leeftijd is duidelijk, volledig voorradig en inhoudelijk relevant. Technisch is de variabele goed bruikbaar. Ethisch moet je wel opletten dat je geen model maakt dat op ongepaste wijze leeftijdsgroepen benadeelt of vereenvoudigt. Als verklarende variabele is leeftijd meestal verantwoord.

Variabele: female
Duidelijkheid: De variabele 'female' is over het algemeen duidelijk te interpreteren. Geslacht van de respondent (0 = man, 1 = vrouw).
Informatief: De variabele 'female' heeft een beperkte informatie-inhoud met slechts 2 unieke waarden. Controleer of dit voldoende is voor analyse.
Privacy: De variabele 'female' lijkt op het eerste gezicht geen directe privacygevoelige informatie te bevatten die individuen identificeerbaar maakt, mits geaggregeerde analyse.
Kwaliteit: De variabele 'female' is van type 'int64'. Dit is een numeriek type, wat goed is voor kwantitatieve analyses. Er zijn geen directe aanwijzingen voor technische problemen.
Ethiek: Variabelen zoals 'female' kunnen gevoelig zijn in ethische overwegingen, met name bij het trekken van conclusies die kunnen leiden tot stereotypering of discriminatie. Het is cruciaal om resultaten met deze variabelen zorgvuldig te interpreteren en eventuele biases te minimaliseren.
Voorradig: De variabele 'female' is volledig voorradig; er zijn geen ontbrekende waarden.
Echtheid: Gezien de bron (ESS) wordt aangenomen dat de data voor 'female' betrouwbaar en een redelijke afspiegeling van de realiteit zijn, maar dit moet in de context van de survey-methodologie worden bekeken.
Relevantie: De variabele 'female' is zeer waarschijnlijk relevant voor het onderzoeken van het vertrouwen in de Belgische politie, hetzij direct, hetzij als contextuele variabele.
Conclusie: Geslacht is een klassieke achtergrondvariabele, maar ook een gevoelig persoonskenmerk. In deze dataset is de codering binair, wat technisch eenvoudig is, maar inhoudelijk beperkt. Voor sommige analyses is ze verdedigbaar als controlevariabele, maar je moet nagaan of ze echt noodzakelijk is. Gebruik ze dus alleen als ze functioneel bijdraagt.

Variabele: eduyrs
Duidelijkheid: De variabele 'eduyrs' is over het algemeen duidelijk te interpreteren. Aantal jaren onderwijs voltooid.
Informatief: De variabele 'eduyrs' lijkt voldoende informatief te zijn met 24 unieke waarden.
Privacy: De variabele 'eduyrs' lijkt op het eerste gezicht geen directe privacygevoelige informatie te bevatten die individuen identificeerbaar maakt, mits geaggregeerde analyse.
Kwaliteit: De variabele 'eduyrs' is van type 'float64'. Dit is een numeriek type, wat goed is voor kwantitatieve analyses. Er zijn geen directe aanwijzingen voor technische problemen.
Ethiek: Variabelen zoals 'eduyrs' kunnen gevoelig zijn in ethische overwegingen, met name bij het trekken van conclusies die kunnen leiden tot stereotypering of discriminatie. Het is cruciaal om resultaten met deze variabelen zorgvuldig te interpreteren en eventuele biases te minimaliseren.
Voorradig: De variabele 'eduyrs' heeft 36 ontbrekende waarden (2.11%). Dit is een beheersbaar aantal dat geïmputeerd of verwijderd kan worden.
Echtheid: Gezien de bron (ESS) wordt aangenomen dat de data voor 'eduyrs' betrouwbaar en een redelijke afspiegeling van de realiteit zijn, maar dit moet in de context van de survey-methodologie worden bekeken.
Relevantie: De variabele 'eduyrs' is zeer waarschijnlijk relevant voor het onderzoeken van het vertrouwen in de Belgische politie, hetzij direct, hetzij als contextuele variabele.
Conclusie: Aantal jaren onderwijs is inhoudelijk relevant en behoorlijk informatief. De missings zijn beperkt en technisch beheersbaar. Ethisch is het normaal toelaatbaar als verklarende factor, zolang je het niet gebruikt om ongelijkheid te reproduceren zonder kritische interpretatie. Deze variabele is bruikbaar.

Variabele: hincfel
Duidelijkheid: De variabele 'hincfel' is over het algemeen duidelijk te interpreteren. Gevoelde hoogte van het huishoudinkomen, vaak geschaald van 1 tot 10 (laag naar hoog).
Informatief: De variabele 'hincfel' lijkt voldoende informatief te zijn met 10 unieke waarden.
Privacy: De variabele 'hincfel' lijkt op het eerste gezicht geen directe privacygevoelige informatie te bevatten die individuen identificeerbaar maakt, mits geaggregeerde analyse.
Kwaliteit: De variabele 'hincfel' is van type 'float64'. Dit is een numeriek type, wat goed is voor kwantitatieve analyses. Er zijn geen directe aanwijzingen voor technische problemen.
Ethiek: Variabelen zoals 'hincfel' kunnen gevoelig zijn in ethische overwegingen, met name bij het trekken van conclusies die kunnen leiden tot stereotypering of discriminatie. Het is cruciaal om resultaten met deze variabelen zorgvuldig te interpreteren en eventuele biases te minimaliseren.
Voorradig: De variabele 'hincfel' heeft 7 ontbrekende waarden (0.41%). Dit is een beheersbaar aantal dat geïmputeerd of verwijderd kan worden.
Echtheid: Gezien de bron (ESS) wordt aangenomen dat de data voor 'hincfel' betrouwbaar en een redelijke afspiegeling van de realiteit zijn, maar dit moet in de context van de survey-methodologie worden bekeken.
Relevantie: De variabele 'hincfel' is zeer waarschijnlijk relevant voor het onderzoeken van het vertrouwen in de Belgische politie, hetzij direct, hetzij als contextuele variabele.
Conclusie: Deze variabele meet subjectief ervaren inkomen en is nuttig als sociaaleconomische contextvariabele. De variabele is voldoende informatief en bijna volledig aanwezig. Ze is niet sterk privacygevoelig, maar wel gevoelig in interpretatie omdat inkomenservaring samenhangt met sociale ongelijkheid. In een verantwoord model kan ze opgenomen worden.

Variabele: plcpvcr
Duidelijkheid: De variabele 'plcpvcr' is over het algemeen duidelijk te interpreteren. Perceptie van criminaliteit in de omgeving, geschaald van 0 tot 10 (geen criminaliteit tot veel criminaliteit).
Informatief: De variabele 'plcpvcr' lijkt voldoende informatief te zijn met 11 unieke waarden.
Privacy: De variabele 'plcpvcr' lijkt op het eerste gezicht geen directe privacygevoelige informatie te bevatten die individuen identificeerbaar maakt, mits geaggregeerde analyse.
Kwaliteit: De variabele 'plcpvcr' is van type 'float64'. Dit is een numeriek type, wat goed is voor kwantitatieve analyses. Er zijn geen directe aanwijzingen voor technische problemen.
Ethiek: De variabele 'plcpvcr' lijkt op het eerste gezicht geen specifieke ethische problemen op te werpen, mits analyses op een neutrale en respectvolle manier worden uitgevoerd.
Voorradig: De variabele 'plcpvcr' heeft 14 ontbrekende waarden (0.82%). Dit is een beheersbaar aantal dat geïmputeerd of verwijderd kan worden.
Echtheid: Gezien de bron (ESS) wordt aangenomen dat de data voor 'plcpvcr' betrouwbaar en een redelijke afspiegeling van de realiteit zijn, maar dit moet in de context van de survey-methodologie worden bekeken.
Relevantie: De variabele 'plcpvcr' is zeer waarschijnlijk relevant voor het onderzoeken van het vertrouwen in de Belgische politie, hetzij direct, hetzij als contextuele variabele.
Conclusie: Deze variabele is sterk relevant voor vertrouwen in de politie, omdat waargenomen criminaliteit vaak samenhangt met veiligheid en vertrouwen. De kwaliteit is goed en de missing rate laag. Ethisch moet je opletten dat perceptie niet wordt verward met objectieve criminaliteit. Toch is ze als verklarende variabele goed verdedigbaar.
"""
"""
Voor een verantwoord ML-proces zou ik deze indeling maken:
  - Uitsluiten: rownames, essround, edition, idno, cntry.
  - Gebruiken als target: trstplc.
  - Gebruiken als features: agea, eduyrs, hincfel, plcpvcr.
  - Voorzichtig gebruiken: female, enkel indien je expliciet wilt controleren op geslachtsverschillen en je dit ethisch kan verantwoorden.

Conclusie
De dataset is in het algemeen bruikbaar voor ML en van degelijke kwaliteit. De meeste problemen zitten niet in technische fouten, maar in relevantie, privacy en ethische geschiktheid van identificerende of constante variabelen. Voor jouw onderzoek naar vertrouwen in de Belgische politie zijn vooral trstplc, agea, eduyrs, hincfel en plcpvcr verantwoord; de overige variabelen laat je beter weg.
"""

# 4. Data cleaning en voorbereiding
# Behandelen van ontbrekende waarden: imputeren met het gemiddelde
print("\nOntbrekende waarden voor imputatie:")
print(df.isnull().sum())

for column in ['trstplc', 'eduyrs', 'hincfel', 'plcpvcr']:
    if df[column].isnull().any():
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)
        print(f"Ontbrekende waarden in '{column}' zijn geïmputeerd met het gemiddelde ({mean_value:.2f}).")

print("\nOntbrekende waarden na imputatie:")
print(df.isnull().sum())

# Dubbele rijen (duplicaten) of kolommen verwijderen
# Eerst controleren we op dubbele rijen
num_duplicate_rows = df.duplicated().sum()
if num_duplicate_rows > 0:
    df.drop_duplicates(inplace=True)
    print(f"\n{num_duplicate_rows} dubbele rijen verwijderd.")
else:
    print("\nGeen dubbele rijen gevonden in de dataset.")

# Verwijderen van de 'rownames' kolom omdat deze redundant is
if 'rownames' in df.columns:
    df.drop(columns=['rownames'], inplace=True)
    print("Kolom 'rownames' verwijderd omdat deze redundant is.")

print(f"\nDe dataset heeft nu {df.shape[0]} rijen en {df.shape[1]} kolommen na cleaning.")

# 5. Visualisatie van de dataset voor de analyse
# Visualiseer nu de numerieke waarden van de variabele trstplc om het vertrouwen in de Belgische politie te visualiseren.
plt.figure(figsize=(10, 6))
sns.histplot(df['trstplc'], bins=11, kde=False, color='lightskyblue', edgecolor='black')
plt.title('Vertrouwen in de Belgische politie')
plt.xlabel('Vertrouwen (schaal van 0 tot 10)')
plt.ylabel('Aantal respondenten')
plt.xticks(range(0, 11, 2))
plt.grid(axis='both', alpha=0.75)
plt.tight_layout()
plt.show()

# Doe hetzelfde voor de variabele agea
plt.figure(figsize=(10, 6))
sns.histplot(df['agea'], bins=20, kde=False, color='lightgreen', edgecolor='black')
plt.title('Leeftijd van de respondenten')
plt.xlabel('Leeftijd')
plt.ylabel('Aantal respondenten')
plt.xticks(range(20, 100, 10))
plt.xlim(10, 100)
plt.grid(axis='both', alpha=0.75)
plt.tight_layout()
plt.show()

# Breng nu beide variabelen samen in één histogram of plot, zodat we per leeftijd kunnen lezen wat het vertrouwen is in de Belgische politie. Maak gebruik van een lijndiagram of scatterplot.
plt.figure(figsize=(10, 6))
sns.scatterplot(x='agea', y='trstplc', data=df, color='orange', s=50, edgecolor=None, linewidth=0)
plt.title('Vertrouwen in de politie per leeftijd')
plt.xlabel('Leeftijd')
plt.ylabel('Vertrouwen (schaal van 0 tot 10)')
plt.yticks(range(0, 11, 2))
plt.grid(True, linestyle='-', alpha=0.6)
plt.tight_layout()
plt.show()

# 6. Analyse
# Bereken het gemiddelde van de volgende variabelen: agea, trstplc, plcpvcr
gemiddelde_leeftijd = df['agea'].mean()
gemiddeld_vertrouwen_politie = df['trstplc'].mean()
gemiddelde_perceptie_criminaliteit = df['plcpvcr'].mean()

print(f"Gemiddelde leeftijd van de respondenten: {gemiddelde_leeftijd:.0f}")
print(f"Gemiddeld vertrouwen in de Belgische politie (schaal 0-10): {gemiddeld_vertrouwen_politie:.1f}")
print(f"Gemiddelde perceptie van criminaliteit in de omgeving (schaal 0-10): {gemiddelde_perceptie_criminaliteit:.1f}")

# Zijn er meer mannelijke of vrouwelijke deelnemers?
# Aangenomen dat 'female': 0 = man, 1 = vrouw
geslachtsverdeling = df['female'].value_counts()
aantal_mannen = geslachtsverdeling.get(0, 0) # Gebruik .get() om fouten te voorkomen als een waarde ontbreekt
aantal_vrouwen = geslachtsverdeling.get(1, 0)

print(f"\nAantal mannelijke deelnemers: {aantal_mannen}")
print(f"Aantal vrouwelijke deelnemers: {aantal_vrouwen}")

if aantal_mannen > aantal_vrouwen:
    print("Er zijn meer mannelijke deelnemers.")
elif aantal_vrouwen > aantal_mannen:
    print("Er zijn meer vrouwelijke deelnemers.")
else:
    print("Het aantal mannelijke en vrouwelijke deelnemers is gelijk.")

# Zoek uit of er correlaties bestaan tussen variabelen. Gebruik hiervoor de corr()-functie.
# Gebruik numeric_only=True om enkel numerieke kolommen mee te nemen.
correlatie_tabel = df.corr(numeric_only=True)

print("\nCorrelaties tussen variabelen:")
print(correlatie_tabel)

# Heatmap van de correlaties
plt.figure(figsize=(10, 8))
sns.heatmap(correlatie_tabel, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlaties tussen variabelen")
plt.tight_layout()
plt.show()

"""
Bij het analyseren van de correlatietabel valt meteen op dat er een matig positief verband bestaat tussen het vertrouwen in de politie (trstplc) en de perceptie van criminaliteit in de omgeving (plcpvcr) met een coëfficiënt van 0.38. Dit suggereert dat hoe meer criminaliteit mensen waarnemen, hoe hoger hun vertrouwen in de politie is. Tegelijkertijd zien we een zwakke negatieve correlatie met leeftijd (agea) en opleidingsjaren (eduyrs), wat aangeeft dat oudere en hoger opgeleide respondenten een licht lager vertrouwen tonen. De variabele female (geslacht) vertoont een nauwelijks lineaire correlatie met het vertrouwen, wat suggereert dat er geen verschillen in vertrouwen tussen mannen en vrouwen bestaan. Al met al wijzen deze correlaties erop dat vooral de directe ervaring of perceptie van de leefomgeving (plcpvcr) een sterkere indicator is voor het vertrouwen dan demografische factoren zoals leeftijd of opleidingsniveau.
"""

# 7. Rapportering en reflectie Beschrijf in uw portfolio uw bevindingen.
"""
Dit labo was een diepgaande verkenning van het proces van een realistisch ML-project, waarbij ik van probleemstelling tot initiële analyse ben gegaan met een dataset over het vertrouwen in de Belgische politie. Ik heb geleerd hoe cruciaal de DIPKEVER analyse is voor elke variabele, niet alleen voor technische kwaliteit, maar ook voor privacy, ethiek en relevantie, wat leidde tot weloverwogen beslissingen over welke kolommen best behouden zouden worden. Het vewerken van ontbrekende waarden door middel van imputatie en het verwijderen van redundante variabelen toonde het belang van grondige datacleaning aan voor betrouwbare resultaten. De visualisaties, zoals de histogrammen van vertrouwen en leeftijd, gaven me waardevolle inzichten in de datadistributie (spreiding) en mogelijke verbanden. Vooral de correlatieanalyse was een belangrijk onderdeel, waaruit bleek dat de perceptie van criminaliteit aanzienlijk positief samenhangt met het politieke vertrouwen. Dit labo heeft mijn begrip van data-exploratie, ethische overwegingen bij datagebruik en de interpretatie van statistische verbanden erg verdiept, wat essentieel is voor het analyseren van ML-projecten in de toekomst. Ik zie nu duidelijk dat een sterke basis in deze vroege fasen de kwaliteit van elk model fundamenteel bepaalt.
"""