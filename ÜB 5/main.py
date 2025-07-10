import pandas as pd

df = pd.read_csv('data/SPY_15min_2002-01_to_2006-08.csv')
df2 = pd.read_csv('data/SPY_15min_2020-01_to_2022-01.csv')
df_news = pd.read_csv('data/WhatMovesMarkets_eventdatabase.csv')

df['datetime'] = pd.to_datetime(df['Unnamed: 0'], utc=True)
df.set_index('datetime', inplace=True)
df.index = pd.DatetimeIndex(df.index)  # <--- This ensures time-aware index
df.drop(columns=['Unnamed: 0'], inplace=True)

# Now these will work
df["day_of_week"] = df.index.dayofweek
df["hour_of_day"] = df.index.hour

day_dummies = pd.get_dummies(df["day_of_week"], prefix="day")[["day_1", "day_2", "day_3", "day_4"]]
hour_dummies = pd.get_dummies(df["hour_of_day"], prefix="hour")[["hour_11", "hour_12", "hour_13", "hour_14", "hour_15", "hour_16"]]
print(hour_dummies)

