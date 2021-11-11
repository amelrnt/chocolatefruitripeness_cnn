import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub

# Some utilites
import numpy as np

app = Flask(__name__)

# Load your own trained model
MODEL_PATH = os.path.join(app.root_path, 'assets/model-choco.h5')
model = load_model((MODEL_PATH),custom_objects={'KerasLayer':hub.KerasLayer})
print('Model loaded. Start serving...')   

def model_predict(img):
    # Preprocessing the image
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    img = image.load_img(img, target_size=(150, 150))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.    

    preds = model.predict(img)
    return preds

def decode(pred):
    pred_class = ['immature', 'mature', 'overmature']
    index = np.argmax(pred)
    pred_value = pred_class[index]
    return pred_value    

UPLOAD_PATH = os.path.join(app.root_path, 'static/test')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['UPLOAD_PATH'] = UPLOAD_PATH

@app.route("/")
def index():
    return render_template('V_home.html')

@app.route("/testing")
def testing():
    return render_template('V_result.html')

@app.route('/display/<filename>')
def upload(filename):
    return send_from_directory(app.config["UPLOAD_PATH"], filename)

@app.route("/ensiklopedia")
def ensiklopedia():
    return render_template('V_ensik.html')

@app.route("/about")
def about():
    return render_template('V_about.html')

@app.route("/demo")
def demo():
    return render_template('V_demo.html')    

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        # Get the image from post request
        f = request.files['file']
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            # Save the image to ./uploads
            f.save(os.path.join(app.config['UPLOAD_PATH'], filename))

            # Make prediction
            preds = model_predict(os.path.join(app.config['UPLOAD_PATH'], filename))

            # Process your result for human
            pred_proba = "{:.3f}".format(np.amax(preds)*100)    # Max probability
            pred_class = decode(preds)   # Decode
            
            return render_template('V_result.html', result=pred_class, filename=os.path.join('test/',filename), pred_prob = pred_proba)

    return render_template('V_demo.html')

if __name__ == '__main__':
    app.debug = True
    app.run()    