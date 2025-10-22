import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import ElasticNet
import Data_Process as dp

main_trainData = pd.read_csv("Placement_Train.csv")
main_testData = pd.read_csv("Placement_Test.csv")

nonDigitCols = ["gender","ssc_b","hsc_b","hsc_s","degree_t","workex","specialisation"]

trainData, trainLabels = dp.labeller(main_trainData,nonDigitCols)
trainData = pd.concat([trainData,main_trainData.drop(nonDigitCols,axis=1)],axis=1)
#print("trainData")
#print(trainData.info())

testData,testLabels = dp.labeller(main_testData,nonDigitCols)
testData = pd.concat([testData,main_testData.drop(nonDigitCols,axis=1)],axis=1)
#print("testData")
#print(main_testData.info())

x,y = dp.dataMaker(trainData,'Annual_salary')
#print(testData['Annual_salary'])
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
elastic = ElasticNet(alpha=1,l1_ratio=0.5)
elastic.fit(X_train,y_train)
print(elastic.score(X_test,y_test))
guess = elastic.predict(testData)
print(guess)
dp.saviour(testData['SR_no'],guess)