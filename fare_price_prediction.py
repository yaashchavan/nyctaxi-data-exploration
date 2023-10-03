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

# Extract date-related features for analysis
df['pickup_date'] = df['pickup_datetime'].dt.date
df['pickup_month'] = df['pickup_datetime'].dt.month
df['pickup_weekday'] = df['pickup_datetime'].dt.weekday  # Monday=0, Sunday=6
df['pickup_hour'] = df['pickup_datetime'].dt.hour



# Remove trips with zero passengers and more than 9 passengers
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] < 10)]

# Feature selection (e.g., distance, time features, location features)
features = df[['passenger_count', 'trip_time_in_secs', 'trip_distance','pickup_month','pickup_weekday','pickup_hour']]

# Target variable
target = df['rate_code']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# List of regression models
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    # RandomForestRegressor(),
    # SVR()
]

# Train and evaluate each model
for model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model: {type(model).__name__}")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)
    print("\n")