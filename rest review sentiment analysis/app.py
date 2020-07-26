from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('restaurant-sentiment.pkl', 'rb'))


cv = pickle.load(open('cv-transform.pkl','rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':

        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = model.predict(vect)
        if my_prediction == 1:
            return render_template('index.html',prediction_text="Positive Review")
        elif my_prediction == 0:
            return render_template('index.html',prediction_text="Negative Review ")

    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

