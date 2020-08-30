from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model,model_from_json
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import requests

# Define a flask app
app = Flask(__name__)

json_file = open('model/model_9979.json','r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
#load weights into new model
model.load_weights("model/model_9979.h5")
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256), color_mode='grayscale')


    x = image.img_to_array(img).reshape(-1,256,256,1)

    preds = model.predict(x)
    print(preds)
    return preds[0][0]


@app.route('/', methods=['GET'])
def index():
    # Main page
    link = 'https://api.covidindiatracker.com/total.json'
    response = requests.get(link)
    res_json = response.json()
    active = res_json['active']
    recovered = res_json['recovered']
    confirmed = res_json['confirmed']

    return render_template('index.html',active=active,recovered=recovered,confirmed=confirmed)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        if preds == 1.0:
            result = "Positive"  
        else:
            result = "Negative"   
                
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

