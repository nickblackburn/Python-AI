import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download the dataset
path = "/Users/nicholas/Desktop/Machine Learning/data/FuelConsumptionCo2.csv"
df = pd.read_csv(path)

# Explore the dataset
print(df.head())
print(df.describe())

# Visualize data
viz = df[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

plt.scatter(df.FUELCONSUMPTION_COMB, df.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(df.ENGINESIZE, df.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Fix typo in the next scatter plot
plt.scatter(df.CYLINDERS, df.CO2EMISSIONS, color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()

# Train-test split
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

# Linear regression on Engine Size
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# Visualize the regression line
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Evaluate the model
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, test_y_))

# Linear regression on Fuel Consumption
train_x = train[["FUELCONSUMPTION_COMB"]]
test_x = test[["FUELCONSUMPTION_COMB"]]
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

# Make predictions and evaluate
predictions = regr.predict(test_x)
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(predictions - test_y)))

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)

# use Mean Squared error (MSE)
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Mean Squared error(MSE): %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


# use Residual sum of squares
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
print ('Coefficients: ', regr.coef_)
y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"% np.mean((y_ - y) ** 2))
print('Variance score: %.2f' % regr.score(x, y))



# Features for training
train_features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']

# Convert to NumPy arrays
x_train = np.asanyarray(train[train_features])
y_train = np.asanyarray(train[['CO2EMISSIONS']])

# Features for testing
test_features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']

# Convert to NumPy arrays
x_test = np.asanyarray(test[test_features])
y_test = np.asanyarray(test[['CO2EMISSIONS']])

# Create and fit the Linear Regression model
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

# Print coefficients and metrics
print('Coefficients:', regr.coef_)
y_hat = regr.predict(x_test)
mse = np.mean((y_hat - y_test) ** 2)
variance_score = regr.score(x_test, y_test)

print('Mean Squared Error (MSE): %.2f' % mse)
print('Variance Score: %.2f' % variance_score)
