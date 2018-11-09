import cv2
import h5py
import os
import argparse
import json
import time
from keras.utils import np_utils, conv_utils
from keras.models import Sequential  # 卷积核越多提取的特征越细化
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
import cv2
import numpy as np
import h5py
import os
import keras
from PIL import Image
from keras.utils import np_utils, conv_utils
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.optimizers import Adagrad
from keras.optimizers import  Adam
from keras.optimizers import  sgd
from keras.optimizers import  RMSprop ,Adagrad
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import load_model

from keras import backend as K
from keras import regularizers
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import time
def translate(image, x, y):
    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # 返回转换后的图像
    return shifted
'''
shifted = translate(image, 0, 100)
shifted = translate(image, 0, -100)
shifted = translate(image, 50, 0)
shifted = translate(image, -50, 0)
'''
cv2.waitKey(0)
def eachFile(filepath):                 #将目录内的文件名放入列表中
    pathDir =os.listdir(filepath)
    out = []
    for allDir in pathDir:
        #  child = allDir.decode('gbk')    # .decode('gbk')是解决中文显示乱码问题
        out.append( allDir[0:-4])
    return out

def get_data(data_name, train_left=0.0, train_right=0.7, train_all=0.7, resize=True, data_format=None, t=''):  # 从文件夹中获取图像数据
    file_name = os.path.join(pic_dir_out+ data_name+'_' + str(train_left) + '_' + str(train_right) + t + '_' + str(Width) + "X" + str(Height) + ".h5")
    print(file_name)
    if t == 'train':
        reference_file = "./ai_challenger_pdr2018_train_annotations_20181021.json"
        user_result_list = json.load(open(reference_file, encoding='utf-8'))
        f = open(reference_file, encoding='utf-8')
        ref_list = json.load(f)
        f.close()
        print("the total image is :")
        print(len(ref_list))
        user_result_dict = {}
        for each_item in user_result_list:
            image_id = each_item['image_id']
            if image_id[-4:].lower() == '.jpg':
                image_id = image_id[0:-4]
            label_id = each_item['disease_class']
            user_result_dict[image_id] = label_id
        data_format = conv_utils.normalize_data_format(data_format)
        pic_dir_set = eachFile(pic_dir_data)
        for i in range(61):
            os.makedirs("./out_60/"+str(i))

        count = 0
        for pic_dir in pic_dir_set:
            count = count + 1
            print(count)
            image_class=user_result_dict[pic_dir]
            for j in range(61):
                if  image_class ==j:
                    img = cv2.imdecode(np.fromfile(os.path.join(pic_dir_data, pic_dir + ".jpg"), dtype=np.uint8), -1)  # 解决不能读中文路径的问题
                    cv2.imwrite("./out_60/"+str(j)+"/"+str(count)+".jpg", img)


def main():
    global Width, Height, pic_dir_out, pic_dir_data,pic_dir_val
    Width = 224
    Height = 224
    apple_classes = 6
    yingtao_classes=3
    yumi_classes=8
    putao_classes=7
    ganju_classes=3
    tao_classes=3
    lajiao_classes=3
    malingshu_classes=5
    caomei_classes=3
    fanqie_classes=20
    epoch_num=8
    pic_dir_out = './out_60/'
    pic_dir_data = './ai_challenger_pdr2018_trainingset_20181023/AgriculturalDisease_trainingset/images/'
    pic_dir_val = './ai_challenger_pdr2018_validationset_20181023/AgriculturalDisease_validationset/images/'

    get_data( "Caltech101_color_test_", 0.0, 0.7, data_format='channels_last', t='train')


if __name__ == '__main__':
    main()






