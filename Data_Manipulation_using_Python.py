import pandas as pd
import numpy as np

# import data from an external source
url = ("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
       "IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv")

# creating and filling a dataframe using pandas library
df = pd.read_csv(url, header=None)

# Adding headers to the dataframe
headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
           "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
           "peak-rpm", "city-mpg", "highway-mpg", "price"]
df.columns = headers

# Viewing first 10 rows of the data
df.head(10)

# Viewing bottom 10 row of data
df.tail(10)

# creating a statistical summary of the quantitative data
df.describe()

# creating a statistical summary of all the data (including categorical)
df.describe(include='all')

# concise summary of data
df.info()

# checking each columns datatypes
df.dtypes

# Data wrangling

# identifying and replacing missing values with np.NaN
df.replace('?', np.NAN, inplace=True)

# creating a new dataframe for storing ture for null and false for not null values
missing_data = df.isnull()
missing_data.head(5)

# counting number of missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print(" ")

# Since none of the column in the data has significant number of missing values
# we will not drop them instead replace them with mean or mode

avg_norm_loss = df['normalized-losses'].astype("float").mean(axis=0)
df['normalized-losses'].replace(np.NaN, avg_norm_loss, inplace=True)

avg_bore = df['bore'].astype('float').mean(axis=0)
df["bore"].replace(np.nan, avg_bore, inplace=True)

avg_stroke = df['stroke'].astype('float').mean(axis=0)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

avg_peakrpm = df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

# for categorical data
# replacing missing values with mode

df['num-of-doors'].value_counts()
# or
mode_num_of_doors = df['num-of-doors'].value_counts().idxmax()
# mode = four

# Rechecking for missing values
df['num-of-doors'].replace(np.NAN, mode_num_of_doors, inplace=True)

missing_data = df.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())

# The above command has given no result
# Which means missing values has been tackled

# Let's check datatypes of each column
df.dtypes

# Convert datatypes of some columns into relevant datatype
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# Now we finally obtained the cleansed data set with no missing values and with all data in its proper format.

# DATA STANDARDIZATION
# Converting city-mpg and highway-mpg to L/100km

df['city-mpg'] = 235 / df['city-mpg']
df.rename(columns={'city-mpg': 'city-L/100km'}, inplace=True)

df['highway-mpg'] = 235 / df['highway-mpg']
df.rename(columns={'highway-mpg': 'highway-L100km'}, inplace=True)

# DATA NORMALIZATION
# Normalizing the values of length, width and height so that they fall in the same range

df['length'] = df['length'] / df['length'].max()
df['width'] = df['width'] / df['width'].max()
df['height'] = df['height'] / df['height'].max()

df[["length", "width", "height"]].head()

# Binning data
# Binning data of horsepower column into three categories low, medium and high

df['horsepower'] = df['horsepower'].astype('int', copy=True)

# creating bins using numpy linspace function
bins = np.linspace(min(df['horsepower']), max(df['horsepower']), 4)
bins

group_name = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_name, include_lowest=True)
df[['horsepower', 'horsepower-binned']].head()

df['horsepower-binned'].value_counts()

# DUMMY VARIABLES:
# Creating dummy variables for fuel-type column

dummy_variable1 = pd.get_dummies(df['fuel-type'])
dummy_variable1.head()
dummy_variable2 = pd.get_dummies(df['aspiration'])
dummy_variable2.head()
# Changing the column name for dummy variable
dummy_variable1.rename(columns={'gas': "fuel-type-gas", 'diesel': 'fuel-type-diesel'}, inplace=True)
dummy_variable1.head()

# Merging df and dummy_variable1
df = pd.concat([df, dummy_variable1], axis=1)
df.drop('fuel-type', axis=1, inplace=True)

df = pd.concat([df, dummy_variable2], axis=1)
df.drop("aspiration", axis=1, inplace=True)

# exporting cleaned data file as csv
df.to_csv('cleaned_data.csv')
