import logging
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.model_selection import train_test_split
import dvc.api
import mlflow
import mlflow.sklearn
import logging
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(50)
df_train_store = pd.read_csv('data/train_store.csv',index_col=0)


def loss_function(y, yhat):
    rmspe = np.sqrt(np.mean((y - yhat)**2))
    return rmspe
class RandomForestModelPipeline:
    
    def __init__(self, df, model_name):  
       
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data(df)
        print("Training and testing split was successful.")
        mlflow.log_param('input_rows', self.X_train.shape[0])
        mlflow.log_param('input_cols', self.X_train.shape[1])
        self.model_name = model_name
    
    def prepare_data(self, df):
        X = df.copy().drop(columns=["Sales", "Customers"], axis=1)
        y = df["Sales"]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    def Preproccessor(self):  
        cols = self.X_train.columns
        numeric_cols = ["CompetitionDistance", "Promo2SinceWeek",'SalesperCustomer']
        categorical_cols = self.X_train.drop(columns=numeric_cols, axis=1).columns.to_list()
        
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('encoder', OrdinalEncoder())])
        
        preprocessor = ColumnTransformer(
            transformers=[('numric', numeric_transformer, numeric_cols),
                          ('category', categorical_transformer, categorical_cols)])
        return preprocessor
    
    def train(self):
        preprocessor = self.Preproccessor()
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', RandomForestRegressor(n_estimators=10,
                                criterion='mse',
                                max_depth=5,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_features='auto',
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                min_impurity_split=None,
                                bootstrap=True,
                                oob_score=False,
                                n_jobs=4,
                                random_state=31,
                                verbose=0,
                                warm_start=False))])
        
        
        model = pipeline.fit(self.X_train, self.y_train)
                
        return pipeline, model
    
    
    def test(self, model):
         filename = 'model1.sav'
         pickle.dump(pipeline, open(filename, 'wb'))
         yhat = model.predict(self.X_test)
         error = loss_function(self.y_test, yhat)
         print(error)
         mlflow.log_param('error', error)
randomforestPipeLine = RandomForestModelPipeline(df_train_store, "RandomForset")
pipeline, model = randomforestPipeLine.train()
randomforestPipeLine.test(model)