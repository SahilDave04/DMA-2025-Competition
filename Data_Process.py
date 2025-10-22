import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def labeller(data,colNames):
    lbls = dict()
    tempData = pd.DataFrame()
    for colName in colNames:
        tempData[colName] = data[colName].astype('category').cat.codes
        lbls[colName] = dict(enumerate(data[colName].astype('category').cat.categories))
    #print(tempData.info())
    #print(lbls)
    return tempData,lbls

def dataMaker(data,yColName):
    #print(data.head(10))
    tempX = data.drop(yColName,axis=1)
    tempY = data[yColName]
    #print("tempX")
    #print(tempX.info())
    #print("tempY")
    #print(tempY.info())
    return tempX,tempY

def saviour(sr,pred):
    final = pd.DataFrame({'SR_no':sr,'Annual_salary':pred})
    final.to_csv("preddy.csv",index=False)

def displayer(self,yPred,yTest):
    #print(yPred)
    mse = mean_squared_error(np.exp(yTest), np.exp(yPred))
    print("---------------------------------------")
    print(f'MSE : {mse:.2f}')
    rmse = np.sqrt(mse)
    print(f'RMSE : {rmse:.2f}')
    #acc = accuracy_score(yTest.astype('float'),pd.Series(yPred))
    #print(f'Accuracy : {acc:.2f}')
