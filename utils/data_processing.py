import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/Users/kivan/Desktop/becode_projects/challenge-regression/data/cleaned_data.csv")
log_price = np.log(df['Price'])

# Then we add it to our data frame
df['log_price'] = log_price

log_living = np.log(df["Living area"])

df['log_living'] = log_living
df.drop(["Price","Living area"], axis=1)

## Simple Linear Regression
"""
price_df = np.array(df['Price'])
living_df = np.array(df['Living area'])

plt.scatter(x=living_df, y=price_df)
plt.xlabel("Living area")
plt.ylabel("Price")
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(living_df, price_df, test_size=0.20, random_state=1)
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
regressor = LinearRegression().fit(X_train, y_train)
#print(regressor.score(X_train, y_train))


X_test = X_test.reshape(-1, 1)
regressor.predict(X_test)
regressor.score(X_test, y_test)

plt.scatter(X_test, y_test)
plt.plot(X_test, regressor.predict(X_test), color="red")
plt.title("Price vs living area (Test set)")
plt.xlabel("Living area")
plt.ylabel("Price")
#plt.show()
"""

# Multi Linear Regression
y = df["log_price"]
X = df.drop(["log_price"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
features_scal = scaler.transform(X_train)


regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_train, y_train))


predict = regressor.predict(X_train)
print(regressor.score(X_test, y_test))

plt.scatter(y_train, predict)
plt.title('Target vs Prediction')
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (predict)',size=18)
plt.show()



