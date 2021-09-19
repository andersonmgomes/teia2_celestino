from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import linear_model
import numpy as np
from itertools import chain, combinations

METRICS_F1 = 'F1'
METRICS_MAE = 'MAE'
METRICS_MSE = 'MSE'

class ModelResult:
    def __init__(self, X_cols, order, model, metrics_detail) -> None:
        self.X_cols = X_cols
        self.order = order
        self.model = model
        self.metrics_detail = metrics_detail
    
    def __str__(self):
        return str(self.X_cols) + ' -> ' + str(self.metrics_detail)
    
    def __repr__(self):
        return str(self)
    
    def getMetric(self, metric):
        return self.metrics_detail[metric]

    def getF1(self):
        return self.getMetric(METRICS_F1)

    def getMAE(self):
        return self.getMetric(METRICS_MAE)

    def getMSE(self):
        return self.getMetric(METRICS_MSE)

class LinearRegression:
    def __init__(self, ds, y_colname, metric_order=METRICS_F1) -> None:
        self.__ds_full = ds
        self.__ds_onlynums = self.__ds_full.select_dtypes(exclude=['object'])
        self.__X_full = self.__ds_onlynums.drop(columns=[y_colname])
        self.__Y_full = self.__ds_onlynums[[y_colname]]
        self.__results = None
        self.metric_order = metric_order
        
    def getBestModel(self):
        return self.getResults()[0]
    
    def getResults(self, buffer=True):
        if buffer and self.__results is not None:
            return self.__results
        #else to get results
        self.__results = []
        
        for col_tuple in all_subsets(self.__X_full.columns):
            if len(col_tuple) == 0:
                continue
            col_list = list(col_tuple)
            self.__results.append(self.__score_dataset(col_list))
        
        self.__results = sorted(self.__results, key=lambda x: x.order)
                            
        return self.__results           
        
    def __score_dataset(self, x_cols):
        X = self.__ds_onlynums[x_cols]
        y = self.__Y_full
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
        
        model = linear_model.LinearRegression()

        X_train2 = X_train
        X_valid2 = X_valid
        y_train2 = y_train
        y_valid2 = y_valid
        
        if len(x_cols)==1:
            X_train2 = np.asanyarray(X_train).reshape(-1, 1)
            X_valid2 = np.asanyarray(X_valid).reshape(-1, 1)
            y_train2 = np.asanyarray(y_train).reshape(-1, 1)
            y_valid2 = np.asanyarray(y_valid).reshape(-1, 1)

        model.fit(X_train2, y_train2)
        preds = model.predict(X_valid2)
        
        mae = mean_absolute_error(y_valid2, preds)
        f1 = r2_score(y_valid2, preds)
        mse = mean_squared_error(y_valid2, preds)
        
        return ModelResult(x_cols, (1-f1), model, {METRICS_MAE: mae, METRICS_F1: f1, METRICS_MSE: mse})
        #return [x_cols, f1, {METRICS_MAE: mae, METRICS_F1: f1, METRICS_MSE: mse}, model]

#util methods
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))


'''
from ds_utils import getDSPriceHousing
lr = LinearRegression(getDSPriceHousing(), 'Price')
model = lr.getBestModel()
print(lr.getResults())
'''
