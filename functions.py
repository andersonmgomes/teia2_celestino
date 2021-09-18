import pandas as pd

DS_PATH = 'datasets/'

#load dataset
#download IBM data
def getDSFuelConsumptionCo2():
    return pd.read_csv(DS_PATH + 'FuelConsumptionCo2.csv')

