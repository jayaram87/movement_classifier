from flask import Flask, render_template, request, redirect, url_for
from flask_cors import cross_origin
import pickle
import numpy as np
from model import Model
from prediction_transformer import Predictor_Data_Transformer
from logger import Logger

app = Flask(__name__)

@cross_origin
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@cross_origin
@app.route('/pandas_profiling')
def profile():
    return render_template('profile.html')

@cross_origin
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    #model = Model(os.path.join(os.getcwd(), 'sample.csv')).best_model()
    model = pickle.load(open('model.sav', 'rb'))
    if request.method == 'POST':
        try:
            le = pickle.load(open("label_encoder.sav", 'rb'))
            avg_rss12 = float(request.form.get('avg_rss12'))
            var_rss12 = float(request.form.get('var_rss12'))
            avg_rss13 = float(request.form.get('avg_rss13'))
            var_rss13 = float(request.form.get('var_rss13'))
            avg_rss23 = float(request.form.get('avg_rss23'))
            var_rss23 = float(request.form.get('var_rss23'))
            data = Predictor_Data_Transformer(avg_rss12, var_rss12, avg_rss13, var_rss13, avg_rss23, var_rss23).data()
            predictor = le.inverse_transform(model.predict(data))[0]
            return render_template('prediction.html', temp=predictor)
        except Exception as e:
            Logger('test.log').logger('ERROR', str(e))
            return redirect(url_for('home'))


if __name__ == '__main__':
    app.run()