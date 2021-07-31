from flask import Flask, render_template, send_from_directory, request
import os   
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route("/assets/<path:path>")
def static_dir(path):
    return send_from_directory("static/assets", path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    global class_names
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads/', secure_filename(f.filename))
        f.save(file_path)
        print(f)
        df = pd.read_csv(file_path)
        print(df.head())

@app.route('/analysis')
def predict():
    return render_template('file-upload.html')


@app.route("/about")
def about():
    return "<h1>About</h1>"


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.run(host="0.0.0", debug=True,port=port)