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


path = 'data/train_store.csv'
repo = '/home/neba/Desktop/forecasting  sales'
version = 'mergedv1'

data_url = dvc.api.get_url(
    path=path,
    repo=repo,
    rev= version
)
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(50)
df_train_store = pd.read_csv('data/train_store.csv',index_col=0)
mlflow.log_param('df_train_store_data_url', data_url)
mlflow.log_param('data_version', version)
mlflow.log_param('model_type', 'Random Forest')

#test = pd.read_csv('data/test.csv',index_col="Date", parse_dates=True)

class TransformingTrainStoreData:
   
    def __init__(self):
        pass

    def to_category(self,df):    
        df["Open"] = df["Open"].astype("category")
        df["DayOfWeek"] = df["Open"].astype("category")
        df["Promo"] = df["Promo"].astype("category")
        df["StateHoliday"] = df["StateHoliday"].astype("category")
        df["SchoolHoliday"] = df["SchoolHoliday"].astype("category")
        df['StateHoliday'] = df['StateHoliday'].astype("str").astype("category")
        df["StoreType"] = df["StoreType"].astype("category")
        df["Assortment"] = df["Assortment"].astype("category")
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].astype("category")
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].astype("category")
        df["Promo2"] = df["Promo2"].astype("category")
        df["Promo2SinceYear"] = df["Promo2SinceYear"].astype("category")
        df["PromoInterval"] = df["PromoInterval"].astype("category")
        df['Year'] = df['Year'].astype("category")
        df['Month'] = df['Month'].astype("category")
        return df
    
        
    def convert_to_datetime(self, df):  
        try:
            df['Date'] = pd.to_datetime(df_train_store['Date'])
            return df
        except:
            pass
    
    def sort_by_date(self, df):
        return df.sort_values(by=["Date"], ascending=False)     
    
        
    def Transformed(self, df):
        df = self.to_category(df)
        df = self.convert_to_datetime(df)

        return df

class ExtractingCOlumns:

    def __init__(self):
        pass
   
    # Let's get Days from Date and delete Date since we already have its Year and Month:
    def transform_date(self, df):
        df['Day']=df.Date.dt.day
        df['Day'] = df['Day'].astype("category")
        del df["Date"]
        return df
    
    def to_month_category(self, df):
       df["Monthcategory"] = df["Day"].apply(lambda x: 'BegMonth' if x < 11 else ('Midmonth' if x<21 else 'EndMonth'))
       return df
    def add_weekday_col(self, df):
      
        df["Weekends"] = df["DayOfWeek"].apply(lambda x: 1 if x > 5 else 0)
        df["Weekdays"] = df["DayOfWeek"].apply(lambda x: 1 if x <= 5 else 0)
        return df
    def process(self, df): 
        df = self.transform_date(df)
        df = self.add_weekday_col(df)
        df = self.to_month_category(df)
        
        return df

class Preprocess:
    
    def __init__(self):
        pass
    
    def encode_train_store_data(self, df):
        
        StateHolidayEncoder = preprocessing.LabelEncoder()
        DayInMonthEncoder = preprocessing.LabelEncoder()
        StoreTypeEncoder = preprocessing.LabelEncoder()
        AssortmentEncoder = preprocessing.LabelEncoder()
        PromoIntervalEncoder = preprocessing.LabelEncoder()
        MonthcategoryEncoder = preprocessing.LabelEncoder()

        df['StateHoliday'] = StateHolidayEncoder.fit_transform(df['StateHoliday'])
        df['StoreType'] = StoreTypeEncoder.fit_transform(df['StoreType'])
        df['Assortment'] = AssortmentEncoder.fit_transform(df['Assortment'])
        df['PromoInterval'] = PromoIntervalEncoder.fit_transform(df['PromoInterval'])
        df['Monthcategory'] = MonthcategoryEncoder.fit_transform(df['Monthcategory'])

        return df
    def process(self, df):
        df = self.encode_train_store_data(df)        
        return df
df_train_store = TransformingTrainStoreData().Transformed(df_train_store)
df_train_store = ExtractingCOlumns().process(df_train_store)
df_train_store = Preprocess().process(df_train_store)

# cleaning the data set
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
clean_dataset(df_train_store)

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
                                   ('regressor', RandomForestRegressor(n_estimators=124,
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