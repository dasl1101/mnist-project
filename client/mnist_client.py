import requests
import cv2
import json
import base64
import numpy as np

# URL = 'http://127.0.0.1:8000/train_mnist/'


# params = {'epochs': 10, 'batch_size': 100}
# res = requests.get(URL, params=params)



URL = 'http://127.0.0.1:8000/test_mnist/'

imgFile = 'C:\\projects\\mysite\\1.jpg'

img = cv2.imread(imgFile)
jpg_img = cv2.imencode('.jpg', img)
b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
date = input("모델 날짜를 입력하세요 : ")

files = {
            "img": b64_string,
            "date": date,
        }



response = requests.post(URL, json=json.dumps(files))
print(type(response.text))
list = eval(response.text)
list = list['prediction']
res = np.argmax(list)
print(res)

