from flask import Flask, render_template, send_from_directory, request
import os   
import pandas as pd
import pickle
import sys
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route("/assets/<path:path>")
def static_dir(path):
    return send_from_directory("static/assets", path)

@app.route("/")
def index():
    return render_template("index.html")

FEATURES = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday',
            'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekOfYear']

def make_prediction(df):
    print(df.shape)
    loaded_model = pickle.load(open("./model/model.pkl", 'rb'))
    df = df[FEATURES]
    result = loaded_model.predict(df)
    print("First result: ", result)
    print("RESULT:", np.exp(result))
    
    return np.exp(result)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    global class_names
    if request.method == 'POST':
        print('called')
        # Get the file from post request
        print('Request files', request.files['csv'])
        f = request.files['csv']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads/', secure_filename(f.filename))
        f.save(file_path)
        print(f)
        df = pd.read_csv(file_path, parse_dates=True, index_col="Date")
        # print(df.head())
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['WeekOfYear'] = df.index.weekofyear
        print(df.head())
        print("Index", df.index)
        # TODO: feed into sklearn pipeline
        # TODO: make prediction

        results = make_prediction(df)
        dates = df.index.values
        dates = dates.astype(str).tolist()

        return '<h1>Uploaded</h1>'

@app.route('/analysis')
def predict():
    return render_template('analysis.html')


@app.route("/about")
def about():
    return "<h1>About</h1>"


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.run(host="0.0.0", debug=True,port=port)