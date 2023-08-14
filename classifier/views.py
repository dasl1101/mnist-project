from django.shortcuts import render
from .models import MnistModel
from django.http import JsonResponse
import tensorflow as tf
from glob import glob
import os
import base64
from PIL import Image
import json
from django.views.decorators.csrf import csrf_exempt
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import cv2

@csrf_exempt
def train_mnist(request):
    epochs = request.GET.get('epochs')
    print(type(epochs), epochs)
    epochs = int(epochs)
    batch_size = request.GET.get('batch_size')
    print(type(batch_size), batch_size)
    batch_size = int(batch_size)
    MnistModel.mnist_model(epochs, batch_size)
    print('asdasdasdasd')
    return JsonResponse({'message': 'Model training이 끝났습니다.'})

@csrf_exempt
def test_mnist(request) -> JsonResponse :
    PATH = os.getcwd()
    print(type(request.body))
    data = eval(json.loads(request.body.decode('utf-8')))
    print(type(data))

    raw_date = data['date']
    if '-' in raw_date:
        date =  raw_date.replace('-','')
    else : 
        date = raw_date    
 


    img = data['img']
    img = base64.b64decode(img)
    img = BytesIO(img)
    image = Image.open(img)
    image.save('test.jpg')



    prediction = None
    model_path = None
    for mn_date in  glob(os.path.join(PATH,"mnist","*")):
        if os.path.basename(mn_date) == date:
            model_path = mn_date
            print("for in mn_date" + mn_date)
            break

    if model_path is not None:
        print(model_path)
        

    # 이미지 데이터 변환
        num_img = np.array(image)
        print(num_img.shape)
        image = image.convert('L')
        image = image.resize((28, 28))
        num_img = np.array(image)
        print(num_img.shape)
        image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
        print(image_array.shape)
        # 모델 로드 및 예측
        model = tf.keras.models.load_model(model_path)
        prediction = model(image_array).numpy().tolist()
        print(prediction)
    return JsonResponse({'prediction': prediction})