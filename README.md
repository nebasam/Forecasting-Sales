# Forecasting-Sales

## Overview
The finance team at Rossmann Pharmaceuticals wants to forecast sales in all their stores across several cities six weeks ahead of time. The data team identified factors such as promotions, competition, school and state holidays, seasonality, and locality as necessary for predicting the sales across the various stores.
The main objective of this project is to forecast sales in all the stores found across several cities six weeks ahead of time and serve an end-to-end product that delivers this prediction to analysts in the finance team
## Install
git clone https://github.com/nebasam/Forecasting-Sales
pip install -r requirements.txt
## Model tracking
cd notebooks
mlflow ui
## Data
train.csv: This is a dataset that holds data of sales at Rossman stores. It contains   sale information from 2013 to 2015. There are 1017209 sales data in this dataset
test.csv: This dataset holds test to check performance model
store.csv: This dataset holds information about each stores. 

## Directory Structure

## notebooks
Exploratory data analysis and different models in notebook are found here.
## Data
The dvc version of data is found in this directory
## scripts
Test.py and other function used for Plotting graphs are found in plots.py module
## model
model in pickle and python format is found here

