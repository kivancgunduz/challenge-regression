import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Read raw data from file
raw_data = pd.read_csv("C:/Users/kivan/Desktop/becode_projects/challenge-regression/data/raw_data.csv")

# Remove Unexpected rows
raw_data.drop(raw_data[raw_data['Price'] == 'None' ].index, inplace=True)
raw_data.drop(raw_data[raw_data['Type of property'] == 'new-real-estate-project-apartments'].index,inplace=True)
raw_data.drop(raw_data[raw_data['Type of property'] == 'new-real-estate-project-houses'].index,inplace=True)

# Delete dublicates
raw_data = raw_data.drop_duplicates()


# Price 
raw_data['Price'] = raw_data['Price'].str.replace(',','')
raw_data['Price'] = raw_data['Price'].astype(np.int64)
#print(raw_data.Price.mean())

#print(raw_data.dtypes)

#Bedrooms
raw_data[raw_data['Type of property'] =='flat-studio']['Bedrooms'].all = 0
raw_data.drop(raw_data[raw_data["Bedrooms"] == "None"].index, inplace=True)
raw_data["Bedrooms"] = raw_data['Bedrooms'].astype(int)

# Living Area 

#print(raw_data[raw_data["Living area"] == "None"].value_counts().sum())
raw_data.drop(raw_data[raw_data["Living area"] == "None"].index, inplace=True)
raw_data["Living area"] = raw_data["Living area"].astype(int)

# Building Condition
raw_data['Building condition'].replace({
    'As new': 1,
    'Good': 2,
    'Just renovated': 3,
    'To renovate': 4,
    'To be done up': 5,
    'To restore': 6,
    'None': 2,
}, inplace=True)
raw_data['Building condition'] = raw_data['Building condition'].astype(int)

# Kitchen type
raw_data["Kitchen type"].replace({
    "Installed":2,
    "Hyper equipped":1,
    "None":0,
    "Semi equipped":3,
    "USA hyper equipped":1,
    "Not installed":0,
    "USA installed":2,
    "USA semi equipped":3,
    "USA uninstalled":0,
    }, inplace=True)
raw_data['Kitchen type'] = raw_data['Kitchen type'].astype(int)

# Furnished
raw_data['Furnished'] = raw_data['Furnished'].replace('None', 0).astype(int)


# Number of Frontage
apt_median_value = raw_data[raw_data['Number of frontages'] != 'None']['Number of frontages'].median()
raw_data['Number of frontages'] = raw_data['Number of frontages'].replace('None',2).astype(int)


# Swimming Pool
raw_data['Swimming pool'] = raw_data['Swimming pool'].replace('None', 0).astype(int)

#Type of property
#print(raw_data['Type of property'].unique())
raw_data["Type of property"].replace({
    "apartment":0,
    "house":1,
    "duplex":2,
    "villa":3,
    "mixed-use-building":4,
    "exceptional-property": 5,
    "ground-floor": 6,
    "penthouse": 7,
    "loft": 8,
    "apartment-block": 9,
    "town-house": 10,
    "mansion": 11,
    "service-flat": 12,
    "castle": 13,
    "bungalow": 14,
    "triplex": 15,
    "flat-studio": 16,
    "farmhouse": 17,
    "other-property": 18,
    "kot": 19,
    "manor-house": 20,
    "chalet": 21,
    "country-cottage": 22
    }, inplace=True)
raw_data['Type of property'] = raw_data['Type of property'].astype(int)

# Terrace surface
raw_data['Terrace surface'] = raw_data['Terrace surface'].replace('None', 0).astype(int)

#Garden surface
raw_data['Garden surface'] = raw_data['Garden surface'].replace('None', 0).astype(int)

# surface of the plot 

raw_data['Surface of the plot'] = raw_data['Surface of the plot'].replace('None', None)
raw_data['Surface of the plot'] = raw_data['Surface of the plot'].fillna(raw_data['Living area'])

raw_data['Surface of the plot'] = raw_data['Surface of the plot'].astype(int)

# Locality

locality_stat = raw_data['Locality'].value_counts(ascending=False)

locality_less_10 = locality_stat[locality_stat <= 10]
raw_data['Locality'] = raw_data['Locality'].apply(lambda x: 'other' if x in locality_less_10 else x)
raw_data_dummies = pd.get_dummies(raw_data,columns=['Locality'], drop_first=True)

# Calculate correration and Remove strong correration column
df2 = raw_data.drop(['Locality'], axis=1)
corr_mat = df2.corr().round(2)
#print(corr_mat)

plt.figure(figsize=(10, 5))
sns.heatmap(corr_mat, vmax=1, annot=True, linewidths=.5)
plt.xticks(rotation=30, horizontalalignment='right')
plt.show()


variables = raw_data_dummies[
    [
        "Building condition",
        "Kitchen type",
        "Bedrooms",
        "Furnished",
        "Number of frontages",
        "Swimming pool",
        "Garden surface",
        "Terrace surface",
        "Surface of the plot",
        "Living area",
        "Type of property"
    ]
]

vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
print(vif)


# Saving clean data
#raw_data_dummies.to_csv('./data/cleaned_data.csv')