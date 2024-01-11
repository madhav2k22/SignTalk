from flask import Flask,render_template,request,jsonify
import tensorflow as tf
import numpy as np


app = Flask(__name__)


#load the model
model= tf.keras.models.load('copy model path here ')



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
  
    img_data = request.get_json()['image']

   
    result = model.predict(np.array(img_data).reshape(1, height, width, channels))

  
    return jsonify({'prediction': result.tolist()})

if __name__ == '__main__':
    app.run(debug=True)