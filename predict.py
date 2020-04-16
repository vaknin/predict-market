import pandas as pd
import datetime as dt
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('C:/Users/avivvak/Desktop/Projects/predict/5118047.csv')
test_data = pd.read_csv('C:/Users/avivvak/Desktop/Projects/predict/test.csv')

# Takes a dataframe and converts the date string column, to three seperate columns(Y/M/D)
def date_to_columns(df):

    days = []
    months = []
    years = []

    # Prepare the dates (X)
    for date in df.date:
        # Indices
        index1 = date.find('/')
        index2 = date.rfind('/')

        # Slice
        day = int(date[0:index1])
        month = int(date[index1+1:index2])
        year = int(date[index2+1:index2+5])

        # Add to the array
        days.append(day)
        months.append(month)
        years.append(year)

    # Drop the date column
    df = df.drop(columns=['date'])

    # Add the new date columns
    df['year'] = years
    df['month'] = months
    df['day'] = days

    return df

# Convert the dataframes date to columns
data = date_to_columns(data)
test_data = date_to_columns(test_data)

# Define features
features = ['year', 'month', 'day']

# Fitting data
y = data.price
X = data[features]

# Building the model
model = DecisionTreeRegressor().fit(X, y)

# Predict
model.predict(test_data[features])