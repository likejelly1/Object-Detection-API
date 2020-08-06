import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
import os
import base64
from yolov3_tf2.models import (
    YoloV3,
    YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from flask import Flask, request, Response, jsonify, send_from_directory, abort, url_for, send_file
from werkzeug.utils import secure_filename
from urllib.parse import quote
from base64 import b64encode
from io import BytesIO
from PIL import Image


UPLOAD_FOLDER = './newmoney/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
# customize your API through the following parameters
classes_path = './data/labels/obj.names'
weights_path = './weights/yolov3-final.tf'
tiny = False                    # set to True if using a Yolov3 Tiny model
size = 416                      # size images are resized to for model
output_path = './result/'   # path to output folder where images with detections are saved
num_classes = 4                # number of classes in model

# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tiny:
    yolo = YoloV3Tiny(classes=num_classes)
else:
    yolo = YoloV3(classes=num_classes)

yolo.load_weights(weights_path).expect_partial()
print('weights loaded')

class_names = [c.strip() for c in open(classes_path).readlines()]
print('classes loaded')

# Initialize Flask application
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "welcome to image detection api"

def get_response_image(path):
    pil_image = Image.open(path, mode='r')
    byte_arr = BytesIO()
    pil_image.save(byte_arr,format='JPEG')
    encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii')
    return encoded_img
# API that returns JSON with classes found in images
@app.route('/detections', methods=['POST'])
def get_detections():
    raw_images = []
    images = request.files.getlist("images")
    image_names = []
    for image in images:
        image_name = image.filename
        image_names.append(image_name)
        image.save(os.path.join(os.getcwd(), image_name))
        img_raw = tf.image.decode_image(
            open(image_name, 'rb').read(), channels=3)
        raw_images.append(img_raw)
        
    num = 0
    
    # create list for final response

    for j in range(len(raw_images)):
        # create list of responses for current image
        responses = []
        confidence = []
        raw_img = raw_images[j]
        num+=1
        img = tf.expand_dims(raw_img, 0)
        img = transform_images(img, size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        print('detections:')
        
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))
            confidence.append(np.array(scores[0][i])*100)
            responses.append({
                "class": class_names[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i])*100))
            })
        original_value = np.sum(confidence)/3
        if(original_value<80):
            originality = False
        else:
            originality = True
        img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(output_path + 'detection' + str(num) + '.jpg', img)
        # retval, buffer = cv2.imencode('.jpg',img)
        # data = b64encode(buffer)
        # data = data.decode('ascii')
        # dataurl = 'data:image/jpg;base64,{}'.format(quote(data))
        response = {
            "image": image_names[j],
            "detections": responses,
            "originality": originality,
            "original_value": original_value,
            "imageUrl": get_response_image('./result/detection1.jpg')
        }
        print('output saved to: {}'.format(output_path + 'detection' + str(num) + '.jpg'))

    #remove temporary images
    for name in image_names:
        os.remove(name)
    try:
        return jsonify({"response":response}), 200
    except FileNotFoundError:
        abort(404)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/upload-new-image', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            # flash('No file part')
            return jsonify({"response":"No image Selected"})
        image = request.files['image']
        # if user does not select file, browser also
        # submit an empty part without filename
        if image.filename == '':
            # flash('No selected file')
            return jsonify({"response":"file Selected is no name"})
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(UPLOAD_FOLDER, filename))
            # return redirect(url_for('uploaded_file',
            #                         filename=filename))
    return jsonify({"response":"Upload success"})



if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)

