import requests
import os
import time
import cv2
import configparser
import numpy as np
from tqdm import tqdm
from HelperFunction import draw_detections

conf = configparser.ConfigParser()
conf.read('online_server.ini', encoding='utf-8')
ip = conf.get('server', 'ip')
port = conf.get('server', 'port')
num_send = int(conf.get('setting', 'num_send'))
server_url = f"http://{ip}:{port}/api/predict_img"


def send_image(image_path):
    with open(image_path, "rb") as img_file:
        files = {"file": (os.path.basename(image_path), img_file, "image/png")}
        response = requests.post(server_url, files=files)
    return response


if __name__ == "__main__":
    response = False
    start = time.time()
    for i in tqdm(range(num_send)):
        image_path = "models/img_input.png"
        response = send_image(image_path)
    time_cost = time.time()-start
    print(f"FPS : {num_send/time_cost}")
    if response:
        class_ids, boxes, confidences = eval(response.text)['class_ids'], eval(response.text)['boxes'], eval(response.text)['confidences']
        class_ids = np.array(class_ids)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        img = cv2.imread(image_path)
        combined_img = draw_detections(img, boxes, confidences, class_ids)
        cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
        cv2.imshow("Detected Objects", combined_img)
        cv2.waitKey(0)