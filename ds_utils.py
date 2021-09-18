import pandas as pd
import seaborn as sns
import numpy as np

DS_PATH = 'datasets/'

#load dataset

def getDSFuelConsumptionCo2():
    return pd.read_csv(DS_PATH + 'FuelConsumptionCo2.csv')

def getDSPriceHousing():
    return pd.read_csv(DS_PATH + 'USA-priceHousing.csv')

def getCorrHeatMap(ds):
    return sns.heatmap(ds.corr(), cmap='coolwarm', fmt='.2f', linewidths=0.1,vmax=1.0, square=True, linecolor='white', annot=True)

