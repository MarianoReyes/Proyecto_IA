import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
data = pd.read_csv("https://media.githubusercontent.com/media/MarianoReyes/Proyecto_IA/main/creditcard.csv")

print(data.columns)
data.Class.value_counts()

print('No Frauds', round(data['Class'].value_counts()
      [0]/len(data) * 100, 2), '% of the dataset')
print('Frauds', round(data['Class'].value_counts()
      [1]/len(data) * 100, 2), '% of the dataset')

data.info()

data.describe()

# Revisar faltantes
print("Number of missing values:\n", data.isnull().sum())

# Revisar duplicados
print("Number of duplicates:", data.duplicated().sum())
data.drop_duplicates(inplace=True)

# Guardar el dataset limpio
data.to_csv("cleaned_creditcard.csv", index=False)
