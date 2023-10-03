import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df1 = pd.read_csv('/Users/YashChavan/Desktop/additional_work/Data/trip_data/trip_data_1.csv')
df1.columns = df1.columns.str.replace(' ', '')
df2 = pd.read_csv('/Users/YashChavan/Desktop/additional_work/Data/trip_data/trip_data_2.csv')
df2.columns = df2.columns.str.replace(' ', '')
df3 = pd.read_csv('/Users/YashChavan/Desktop/additional_work/Data/trip_data/trip_data_3.csv')
df3.columns = df3.columns.str.replace(' ', '')

df = pd.concat([df1, df2, df3],axis=0)
del df1 
del df2
del df3

# Convert pickup_datetime and dropoff_datetime to datetime objects
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

