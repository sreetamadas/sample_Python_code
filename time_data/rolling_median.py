import pandas as pd

df = ...

df['median'] = df['y'].rolling(window=30, center=True).median().fillna(method='bfill').fillna(method='ffill')
