import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
import logger



class Plot():

    def plot_bar(self, df:pd.DataFrame, x_col:str, y_col:str, title:str, xlabel:str, ylabel:str)->None:
        """
            Plots BarGraph with given dataframe, x, y columns, title, and label 
        """
        # logger.info(f"Plotting bar with title: {title}, and xlabel: {xlabel}")
        plt.figure(figsize=(12, 7))
        sns.barplot(data = df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks( fontsize=14)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        # logger.debug("Plot the bar")
        plt.show()

    def plot_hist(self, df:pd.DataFrame, column:str, color:str)->None:
        """
            Plots Histogram with given dataframe, columns, and color 
        """
        # logger.info(f"PLotting Histogram of color : {color}")
        # plt.figureself, figsize=(15, 10))
        # fig, ax = plt.subplots(1, figsize=(12, 7))
        sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        # logger.debug("Plot the Histogram")
        plt.show()
    def plot_sub (self,df,df2,col,title1,title2)->None:
        fig, axes = plt.subplots(1, 2, figsize=(17,3.5))
        axes[0].hist(df[col], cumulative=False, bins=20)
        axes[0].set_title(f'Promotion histogram for {title1}')
        axes[1].hist(df2[col], cumulative=False, bins=20)
        axes[1].set_title(f'Promotion histogram for {title2}')
    
    def plot_count(self, df:pd.DataFrame, column:str) -> None:
        """
            Plots the counts  
        """
        # logger.info(f"PLotting Columms ")
        plt.figure(figsize=(12, 7))
        sns.countplot(data=df, x=column)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        # logger.debug("Display the columns")
        plt.show()
        

    def plot_heatmap(self, df:pd.DataFrame, title:str, cbar=False)->None:
        """
            Plots heatmaps with given title  
        """
        # logger.info(f"plotting heatmap with title: {title} ")
        plt.figure(figsize=(12, 7))
        sns.heatmap(df, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=cbar )
        plt.title(title, size=18, fontweight='bold')
        # logger.debug("Plot the Hitmap")
        plt.show()

    def plot_box(self, df:pd.DataFrame, x_col:str, title:str) -> None:
        """
            Plots plot box with given title  
        """
        # logger.info(f"PLotting box with title: {title} ")
        plt.figure(figsize=(12, 7))
        sns.boxplot(data = df, x=x_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        # logger.debug("Plot the Box")
        plt.show()

    def plot_box_multi(self, df:pd.DataFrame, x_col:str, y_col:str, title:str) -> None:
        """
            Plots a box plot with given column, column and title
        """
        # logger.info(f"Plotting Boxplot with given xcolumn, ycolumn, and title: {title} ")
        plt.figure(figsize=(12, 7))
        sns.boxplot(data = df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks( fontsize=14)
        # logger.debug("Plot the Box")
        plt.show()

    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
        """
            Plots a scatter plot with given column, column and title
        """
        # logger.info(f"Plotting scatter plot with given xcolumn, ycolumn, and title: {title} ")
        plt.figure(figsize=(12, 7))
        sns.scatterplot(data = df, x=x_col, y=y_col, hue=hue, style=style)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.yticks( fontsize=14)
        # logger.debug("Plot the scatter plot")
        plt.show()
    def fill_median (self,df,col1) -> DataFrame:
        """
            fill missing value with median
        """
        df[col1].fillna(df[col1].median(), inplace = True)
        return df