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