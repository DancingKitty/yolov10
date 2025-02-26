
from HelperFunction import YOLOv10
from flask import Flask, request
from flask_cors import CORS
import configparser
import cv2
from PIL import Image
import numpy as np
def run_initialize(detector,img_path):
    img = cv2.imread(img_path)
    class_ids, boxes, confidences = detector(img)
    return None
app = Flask(__name__)
CORS(app)

current_model_name = ''

@app.route('/api/predict_img', methods=['GET', 'POST'])
def predict_type():
    img = np.array(Image.open(request.files['file']))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    class_ids, boxes, confidences = detector(img)
    output_dict = {"class_ids":class_ids.tolist(),"boxes":boxes.tolist(),"confidences":confidences.tolist()}
    return output_dict

if __name__ == '__main__':

    conf = configparser.ConfigParser()
    conf.read('online_server.ini', encoding='utf-8')
    ip = conf.get('server', 'ip')
    port = conf.get('server', 'port')

    model_path = conf.get('setting', 'model_path')
    img_path = conf.get('setting', 'img_path')
    detector = YOLOv10(model_path, conf_thres=0.2)
    run_initialize(detector,img_path)
    app.run(
        host=ip,
        port=port,
        debug=False,
        use_reloader=False,
        threaded=False
    )