import flask
import werkzeug
import base64
import io
import os
import time

from flask import jsonify
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from keras.models import load_model
from subprocess import check_output
from io import BytesIO

app = flask.Flask(__name__)

@app.route('/check', methods=['GET'])
def check():
    return "Flask Server Working Successfully"
@app.route('/splitImage', methods = ['POST','GET'])
def splitImage():
    print("Request received")
    imagefile = flask.request.json['image']

    imageData = base64.b64decode(imagefile)
    imageArray = np.fromstring(imageData, np.uint8)
    print(imageArray.shape)
    imageArray = cv2.resize(imageArray, (28,28), interpolation=cv2.INTER_NEAREST)
    print(imageArray.shape)
    height = imageArray.shape[0]
    width = imageArray.shape[1]
    width_cutoff = width // 2

    left1 = imageArray[:, :width_cutoff]
    right1 = imageArray[:, width_cutoff:]

    img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)

    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    q3 = img[:, :width_cutoff]
    q1 = img[:, width_cutoff:]

    q3 = cv2.rotate(q3, cv2.ROTATE_90_COUNTERCLOCKWISE)
    q1 = cv2.rotate(q1, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)

    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2

    q4 = img[:, :width_cutoff]
    q2 = img[:, width_cutoff:]
    q4 = cv2.rotate(q4, cv2.ROTATE_90_COUNTERCLOCKWISE)
    q2 = cv2.rotate(q2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    print("Jsonifier")
    return jsonify(
        quad1 = q1.tolist(),
        quad2 = q2.tolist(),
        quad3 = q3.tolist(),
        quad4 = q4.tolist()
    )

@app.route('/masterClassify', methods = ['POST','GET'])
def masterClassify():
    print("Received Request")
    input = flask.request.json['value']
    trainedModel = load_model('finalModel')
    pred = trainedModel.predict(input)
    print(pred)
    return pred

@app.route('/slaveClassify', methods = ['POST','GET'])
def slaveClassify():
    print("Received Request")
    quadrant = flask.request.json['quadrant']
    if(quadrant==1):
        trainedModel = load_model('model_q1')
    elif(quadrant==2):
        trainedModel = load_model('model_q2')
    elif(quadrant==3):
        trainedModel = load_model('model_q3')
    elif(quadrant==4):
        trainedModel = load_model('model_q4')

    imagefile = flask.request.json['image']
    imageData = base64.b64decode(imagefile)

    imageArray = np.fromstring(imageData, np.uint8)
    image = cv2.imdecode(imageArray, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28,28), interpolation=cv2.INTER_NEAREST)
    res = Image.fromarray(image)
    res = np.array(res).flatten()
    # res = np.array(res.resize((28,28)))

    blackThresh = 128
    numOfBlack = 0
    for pixel in res:
        if pixel < blackThresh:
            numOfBlack += 1
    size = len(res)

    if numOfBlack/float(size) < 0.5:
        _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
        res = thresh
    else:
        _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        res = thresh

    res = res.astype('float32')/255
    res = res.reshape(-1, 14, 14, 1)
    pred = trainedModel.predict(res)
    category = np.argmax(pred, axis=None, out=None)
    print(category)
    return category

@app.route('/upload', methods = ['GET', 'POST'])
def handle_request():
    print("Request received")
    imagefile = flask.request.json['image']
    imageData = base64.b64decode(imagefile)

    imageArray = np.fromstring(imageData, np.uint8)
    image = cv2.imdecode(imageArray, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28,28), interpolation=cv2.INTER_NEAREST)
    res = Image.fromarray(image)
    res = np.array(res).flatten()
    # res = np.array(res.resize((28,28)))

    blackThresh = 128
    numOfBlack = 0
    for pixel in res:
        if pixel < blackThresh:
            numOfBlack += 1
    size = len(res)
    
    if numOfBlack/float(size) < 0.5:
        _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
        res = thresh
    else:
        _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        res = thresh

    # res = np.reshape(res,[-1,784])
    res = res.astype('float32')/255
    res = res.reshape(-1, 28, 28, 1)
 
    trainedModel = load_model('cnnModel')
    pred = trainedModel.predict(res)
    category = np.argmax(pred, axis=None, out=None)
    print(category)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = 'Img_'+timestr
    imagePath = (str(category)+"/"+filename)
    os.makedirs(os.path.dirname(imagePath), exist_ok=True)
    img = Image.open(io.BytesIO(imageData))
    img.save(imagePath, 'png')
    return str(category)

app.run(host="0.0.0.0", port=8081, debug=True)
