import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pandas_datareader as data
from pandas_datareader import *
import pandas_datareader.data as web
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import date
import streamlit as st

TODAY = date.today().strftime("%Y-%m-%d")
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter stock ticker', 'TSLA')
tick = user_input
yf.download(tick)
df=yf.download(tick)

st.write(df)
#st.write(df.describe())

list = df.columns.tolist()
choise = st.multiselect("Choose  the feature to predict", list , default=["Close"])
datas= df[choise]
#st.line_chart(data)
st.area_chart(datas)

x = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 0)

st.subheader('Actual Price vs Predicted Price ')
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
#regressor=LinearRegression()
regressor.fit(x_train, y_train)
predicted=regressor.predict(x_test)
dframe=pd.DataFrame(y_test, predicted)
dfr=pd.DataFrame({'Actual Price' :y_test, 'Predicted Price' :predicted})
df_sorted = dfr.sort_values(by='Date')

# Display the DataFrame using Streamlit
st.write(df_sorted)

#st.subheader('Prediction vs Original ')
#graph=dfr.tail(150)
graph=dfr.tail(50)


graph.plot(kind='bar')
st.bar_chart(graph)



#df=yf.download(tick)
start='2010-08-10'
end='2024-04-29'
#TODAY = date.today().strftime("%Y-%m-%d")
#data = yf.download(tick, start, TODAY)
data = yf.download(tick, start, end)
# Function to create lag features
def create_lag_features(data, lag_days):
    for i in range(1, lag_days + 1):
        data[f'lag_{i}'] = data['Close'].shift(i)
    return data

# Function to prepare data for training
def prepare_data(data, lag_days):
    data = create_lag_features(data, lag_days)
    data.dropna(inplace=True)
    X = data.drop(['Close'], axis=1)
    y = data['Close']
    return X, y

# Function to forecast future prices
def forecast_prices(model, last_data_point, num_days):
    forecast = []
    last_data_point = np.array(last_data_point).reshape(1, -1)
    for i in range(num_days):
        prediction = model.predict(last_data_point)
        forecast.append(prediction[0])
        last_data_point = np.roll(last_data_point, -1)
        last_data_point[0, -1] = prediction
    return forecast

# Prepare data
lag_days = 10
X, y = prepare_data(data, lag_days)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
#rf_regressor=LinearRegression()
rf_regressor.fit(X_train, y_train)
# Evaluate model
y_pred_train = rf_regressor.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
print("Training MSE:", mse_train)

y_pred_test = rf_regressor.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
print("Testing MSE:", mse_test)
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Forecast future prices
last_date = data.index[-1]
#last_date = data['Close'].iloc[-1]
date_range = pd.date_range(start=last_date, periods=30)  # Include Saturdays and Sundays
date_range = date_range[date_range.weekday < 5]  # Filter out Saturdays and Sundays
import holidays
from datetime import date

# Define the country for which you want to identify public holidays
country = 'IN'  # Example: United States

# Create a Holiday object for the specified country
us_holidays = holidays.CountryHoliday(country)

# Define the date range for which you want to check public holidays
start_date = date(2024, 1, 1)
end_date = date(2024, 12, 31)

# Iterate through the date range and check if each date is a public holiday
public_holidays = []
for single_date in pd.date_range(start_date, end_date):
    if single_date in us_holidays:
        public_holidays.append(single_date)

print("Public holidays in", country, "for the year 2024:", public_holidays)

date_range = [d for d in date_range if d not in public_holidays]

date_range = date_range[1:]  # Exclude last date and start from the next day (Monday)

last_data_point = data.tail(1).drop(['Close'], axis=1).values.flatten()
forecast_cp = forecast_prices(rf_regressor, last_data_point, num_days=len(date_range))
#forecast_cp = forecast_prices(rf_regressor, last_date, num_days=len(date_range))

# Combine dates with forecasted values
forecast_df = pd.DataFrame({'Date': date_range, 'Forecasted_CP': forecast_cp})
print(forecast_df)
#st.subheader('30 days forecasted price')
#st.write(forecast_df)

st.title('Stock Price Forecasting')
st.subheader('30 days forecasted price')
#st.line_chart(forecast_df.set_index('Date')['Forecasted_CP'])
#st.bar_chart(forecast_df.set_index('Date')['Forecasted_CP'])
st.area_chart(forecast_df.set_index('Date')['Forecasted_CP'])