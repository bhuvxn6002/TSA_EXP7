# Ex.No: 07 AUTO-REGRESSIVE MODEL

## AIM:

To Implementat an Auto Regressive Model using Python

## ALGORITHM :

Step 1 :
Import necessary libraries.

Step 2 :
Read the CSV file into a DataFrame.

Step 3 :
Perform Augmented Dickey-Fuller test.

Step 4 :
Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags.

Step 5 :
Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF).

Step 6 :
Make predictions using the AR model.Compare the predictions with the test data.

Step 7 :
Calculate Mean Squared Error (MSE).Plot the test data and predictions.

## PROGRAM:
### Name : Bhuvaneshawaran H
### Register Number : 212223240018
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

data = pd.read_csv('silver.csv',parse_dates=['Date'],index_col='Date')
data.head()


data['USD'].fillna(method='ffill', inplace=True)

data = data.dropna(subset=['USD'])  
data = data[np.isfinite(data['USD'])] 

result = adfuller(data['USD']) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])

x=int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

lag_order = 13
model = AutoReg(train_data['USD'], lags=lag_order)
model_fit = model.fit()

plt.figure(figsize=(10, 6))
plot_acf(data['USD'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()
plt.figure(figsize=(10, 6))
plot_pacf(data['USD'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

mse = mean_squared_error(test_data['USD'], predictions)
print('Mean Squared Error (MSE):', mse)

plt.figure(figsize=(12, 6))
plt.plot(test_data['USD'], label='Test Data - Price')
plt.plot(predictions, label='Predictions - Price',linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid()
plt.show()
```

## OUTPUT:

Dataset:

![image](https://github.com/user-attachments/assets/f2fd91c7-bfd1-4a2b-86f8-3fa7e70b38b6)


ADF test result:

![image](https://github.com/user-attachments/assets/8d2ee0d1-b72d-4deb-b93d-e2b3144abc09)

PACF plot:

![image](https://github.com/user-attachments/assets/1e326a1e-76f0-4d80-baa5-4a53cf4d3500)


ACF plot:

![image](https://github.com/user-attachments/assets/d20dae10-262b-4fba-ac04-2d4397dab715)


Accuracy:

![image](https://github.com/user-attachments/assets/f48733d2-cd39-4ab4-a331-aabeee394deb)

Prediction vs test data:

![image](https://github.com/user-attachments/assets/95e7801f-2383-4b8c-a380-8bcf63185a01)

## RESULT:

Thus we have successfully implemented the auto regression function using python.
