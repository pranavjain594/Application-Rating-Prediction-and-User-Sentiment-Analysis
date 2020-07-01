
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('googleplaystore3.csv')
newdataset=dataset
dataset.head()
newdataset['Content Rating'] = newdataset['Content Rating'].map({'Everyone':0,'Teen':1,'Everyone 10+':2,'Unrated':0,'Mature 17+':3,
                                                     'Adults only 18+':4}).astype(float)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
newdataset["Category"] = labelencoder_X.fit_transform(newdataset["Category"])
newdataset["Genres"] = labelencoder_X.fit_transform(newdataset["Genres"])

newdataset["Installs"] = [ float(i.replace('+','').replace(',', '')) if '+' in i or ',' in i else float(0) for i in newdataset["Installs"] ]

newdataset['Price'] = [ float(i.split('$')[1]) if '$' in i else float(0) for i in newdataset['Price'] ]

newdataset["Size"] = [ float(i.replace('.',"").replace('M','00').replace('k','')) if '.' in i else float(i.replace('M','000').replace('k','').replace('Varies with device','NaN')) for i in newdataset["Size"] ]

newdataset["Category"].unique()
newdataset["Installs"].unique()
newdataset["Price"].unique()
newdataset["Size"].unique()
newdataset["Content Rating"].unique()
newdataset["Genres"].unique()

X = newdataset.iloc[:, [1,3,5,7,8,9]].values
y = newdataset.iloc[:, 2].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
y = imputer.fit_transform(y.reshape(-1,1))
X = imputer.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#KNN REGRESSOR
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=10,metric='chebyshev')
neigh.fit(X_train, y_train)
knn_pred = neigh.predict(X_test)
mean_squared_error(y_test, knn_pred)

#random forest
from sklearn.ensemble import RandomForestRegressor
regressor1 = RandomForestRegressor(n_estimators =250, random_state = 0)
regressor1.fit(X_train, y_train)
y_pred2 = regressor1.predict(X_test)
mean_squared_error(y_test, y_pred2)

from sklearn.linear_model import LinearRegression
regressor2 = LinearRegression()
regressor2.fit(X_train, y_train)
y_pred3 = regressor2.predict(X_test)
mean_squared_error(y_test, y_pred3)

from sklearn.svm import SVR
regressor3 = SVR(kernel = 'rbf')
regressor3.fit(X_train, y_train)
y_pred4 = regressor3.predict(X_test)
mean_squared_error(y_test, y_pred4)

# Visualising the Training set results
plt.scatter(x=y_test,y=knn_pred,color='c')
plt.title('Google Play Store Rating Prediction(KNN)')
plt.xlabel('REAL')
plt.ylabel('PREDICTED')
plt.show()

plt.scatter(x=y_test,y=y_pred2,color='c')
plt.title('Google Play Store Rating Prediction(Random Forest)')
plt.xlabel('REAL')
plt.ylabel('PREDICTED')
plt.show()

plt.scatter(x=y_test,y=y_pred3,color='c')
plt.title('Google Play Store Rating Prediction(Multi linear)')
plt.xlabel('REAL')
plt.ylabel('PREDICTED')
plt.show()

plt.scatter(x=y_test,y=y_pred4,color='c')
plt.title('Google Play Store Rating Prediction(SVR)')
plt.xlabel('REAL')
plt.ylabel('PREDICTED')
plt.show()

plt.scatter(X_train[:,0], y_train, color = 'red')
plt.title('Google Play Store')
plt.xlabel('Category')
plt.ylabel('Rating')
plt.show()

plt.scatter(X_test[:,0], knn_pred, color = 'red')
plt.scatter(X_test[:,0], y_test, color = 'blue')
plt.title('Google Play Store(KNN)')
plt.xlabel('Category')
plt.ylabel('Rating')
plt.show()

plt.scatter(X_test[:,0], y_pred2, color = 'red')
plt.scatter(X_test[:,0], y_test, color = 'blue')
plt.title('Google Play Store(random forest)')
plt.xlabel('Category')
plt.ylabel('Rating')
plt.show()

plt.scatter(X_test[:,0], y_pred4, color = 'red')
plt.scatter(X_test[:,0], y_test, color = 'blue')
plt.title('Google Play Store(svr)')
plt.xlabel('Category')
plt.ylabel('Rating')
plt.show()

dataset.Category.plot(kind="hist",color="red",figsize=(8,8),bins=35)
plt.xlabel('Category')

dataset.Rating.plot(kind="hist",color="red",figsize=(8,8),bins=34)
plt.show()
