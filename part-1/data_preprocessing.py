from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing datasets

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# missing data-
imputer_data = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_data = imputer_data.fit(x[:, 1:3])
x[:, 1:3] = imputer_data.transform(x[:, 1:3])
print(x)

from sklearn.preprocessing import LabelEncoder , OneHotEncoder

labelencoder_x = LabelEncoder()

x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categories=[0])
x = onehotencoder.fit_transform(x).toarray()


print(x)
