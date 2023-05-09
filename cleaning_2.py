import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("fraudTest.csv")


print(data.columns)

print(data.describe())