import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def MoveActivity(fileName , location , saveName):
    dataframe = pd.read_csv(location + fileName)
    activity = dataframe['Activity']
    dataframe.drop(labels=['Activity'], axis=1,inplace = True)
    dataframe.insert(len(dataframe.columns.values) , 'Activity', activity)
    dataframe.to_csv(location + saveName,index=False)
    pass

if __name__ == '__main__':
    MoveActivity(fileName='train.csv' , location='../../Data/' , saveName='newtrain.csv')
    MoveActivity(fileName='test.csv' , location='../../Data/' , saveName='newtest.csv')