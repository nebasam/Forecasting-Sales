from flask import Flask, render_template, send_from_directory, request
import os   
import pandas as pd
import pickle
import sys
import numpy as np
from werkzeug.utils import secure_filename, send_file

app = Flask(__name__)

@app.route("/assets/<path:path>")
def static_dir(path):
    return send_from_directory("static/assets", path)

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/download', methods=['GET', 'POST'])
def download_file():
    return send_from_directory('downloads', 'result.csv')

FEATURES = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday',
            'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekOfYear']

def make_prediction(df):
    if os.path.exists('downloads/result.csv'):
        os.remove("downloads/result.csv")
    print(df.shape)
    loaded_model = pickle.load(open("./model/model.pkl", 'rb'))
    df = df[FEATURES]
    result = loaded_model.predict(df)
    print('The Type is: ',type(result))
    df['Result'] = result
    # save_file = pd.DataFrame(np.exp(result), columns=['Sales'])
    df.to_csv('downloads/result.csv')
    print("First result: ", result)
    print("RESULT:", np.exp(result))


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
        results = make_prediction(df)
        print('Printing result',results[0])
        return str(int(results[0]))

@app.route('/analysis')
def predict():
    return render_template('analysis.html')


@app.route("/about")
def about():
    return "<h1>About</h1>"


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.run(host="0.0.0", debug=True,port=port)
