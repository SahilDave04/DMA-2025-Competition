import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class Trainer:
    def __init__(self,train,test,yColName):
        self.train = train
        self.test = test
        self.yColName = yColName

    def dataMaker(self,data,yColName):
        print(data.head(10))
        tempX = data.drop(yColName,axis=1)
        tempY = data[yColName]
        #print("tempX")
        #print(tempX.info())
        #print("tempY")
        #print(tempY.info())
        return tempX,tempY

    def saviour(self,sr,pred):
        final = pd.DataFrame({'SR_no':sr,'Annual_salary':pred})
        final.to_csv("preddy.csv",index=False)

    def displayer(self,yPred,yTest):
        #print(yPred)
        mse = mean_squared_error(yTest, yPred)
        print("---------------------------------------")
        print(f'MSE : {mse:.2f}')
        rmse = np.sqrt(mse)
        print(f'RMSE : {rmse:.2f}')
        #acc = accuracy_score(yTest.astype('float'),pd.Series(yPred))
        #print(f'Accuracy : {acc:.2f}')

    def important_feats(self,model_fit,X_train,X_test):
        importances = pd.DataFrame({"Gini Importance":model_fit.feature_importances_})
        colums = pd.DataFrame({"Top Vars":X_train.columns.tolist()})
        #print(colums)
        sorted_imps = pd.concat([colums,importances],axis=1).sort_values('Gini Importance', ascending=False)
        #print(sorted_imps)
        top = sorted_imps["Top Vars"][:10]
        #print(top.tolist())
        X_train = X_train[top]
        #print("Filtered_X-train")
        #print(X_train.info())
        X_test = X_test[top]
        return X_train,X_test,top.tolist()

    def kFold(self):
        something = None

    def randomRegress(self,feature_sel=False):
        X, Y = self.dataMaker(self.train,self.yColName)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
        #X_test, y_test = self.dataMaker(self.test,self.yColName)
        clf1 = RandomForestRegressor(n_estimators=150, random_state=21)
        clf1.fit(X_train, y_train)
        if feature_sel == True:
            filter_X_train, X_test,topper = self.important_feats(clf1,X_train, X_test)
            #print("y-train")
            #print(y_train.info())
            clf1.fit(filter_X_train, y_train)
        y_pred1 = clf1.predict(X_test)
        print("y_pred1")
        print(y_pred1)
        print("y_test")
        print(y_test)
        truePred = clf1.predict(self.test[topper])
        #truePred = np.exp(clf1.predict(self.test[topper]))
        print(truePred)
        self.displayer(y_pred1,y_test)
        self.saviour(self.test['SR_no'],truePred)
        

def labeller(data,colNames):
    lbls = dict()
    tempData = pd.DataFrame()
    for colName in colNames:
        tempData[colName] = data[colName].astype('category').cat.codes
        lbls[colName] = dict(enumerate(data[colName].astype('category').cat.categories))
    #print(tempData.info())
    #print(lbls)
    return tempData,lbls


main_trainData = pd.read_csv("Placement_Train.csv")
main_testData = pd.read_csv("Placement_Test.csv")
y_testData = pd.read_csv("Placement_Sample_Submission.csv")
main_testData = pd.concat([main_testData,y_testData["Annual_salary"]],axis=1)

nonDigitCols = ["gender","ssc_b","hsc_b","hsc_s","degree_t","workex","specialisation"]

trainData, trainLabels = labeller(main_trainData,nonDigitCols)
trainData = pd.concat([trainData,main_trainData.drop(nonDigitCols,axis=1)],axis=1)
#print("trainData")
#print(trainData.info())

testData,testLabels = labeller(main_testData,nonDigitCols)
testData = pd.concat([testData,main_testData.drop(nonDigitCols,axis=1)],axis=1)
#print("testData")
#print(testData.info())

trainer1 = Trainer(trainData,testData,"Annual_salary")
run1 = trainer1.randomRegress(True)

trainData2 = trainData.copy()
trainData2['Annual_salary'] = np.log(trainData2['Annual_salary'])
trainer2 = Trainer(trainData2,testData,"Annual_salary")
#run2 = trainer2.randomRegress(True)

preprocessor = None
pipeline = Pipeline([])
