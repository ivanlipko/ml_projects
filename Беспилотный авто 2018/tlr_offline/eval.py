import numpy as np
import cv2

GLOB_INIT = True
clf = 0      # classifier

clf_data = """"""

def get_hog_vector(im):
    """
    Функция, возвращающая вектор HOG для одного grayscale-изображения размера (64 x 64). 
    Возвращаемый вектор имеет вид (1 x длина вектора HOG).
    """
    hog_descriptor = cv2.HOGDescriptor((64, 64), #winsize
                                       (32, 32), #blocksize
                                       (16, 16), #blockstride
                                       (8,8),   #cellsize
                                       9)        #nbins
    return hog_descriptor.compute(im).T

def one_hot_encode(label):
    """ Функция осуществляет перекодировку текстового входного сигнала
     в массив элементов, соответствующий выходному сигналу
     Входные параметры: текстовая метка
     Выходные параметры: метка ввиде массива
     Пример:
        one_hot_encode("red") должно возвращать: [1, 0, 0]
        one_hot_encode("yellow") должно возвращать: [0, 1, 0]
        one_hot_encode("green") должно возвращать: [0, 0, 1]
     """
    one_hot_encoded = []
    if label == "red":
        one_hot_encoded = [1, 0, 0]
    elif label == "yellow":
        one_hot_encoded = [0, 1, 0]
    elif label == "green":
        one_hot_encoded = [0, 0, 1]
    return one_hot_encoded

def one_hot_encode_num(num):
    one_hot_encoded = []
    if num == 1:
        one_hot_encoded = [1, 0, 0]
    elif num == 2:
        one_hot_encoded = [0, 1, 0]
    elif num == 3:
        one_hot_encoded = [0, 0, 1]
    return one_hot_encoded

# приведение входного изображения к стандартному виду
def standardize_input(image):
    im = image
    """Приведение изображений к стандартному виду. 
    Входные данные: изображение
    Выходные данные: стандартизированное изображений.
    """

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (64, 64))
    standard_im = im
    return standard_im

def init_eval():
    IS_SERVER = 1
    if IS_SERVER == 1:   # load from memory
        with open("traffic_light_images/clf_data.xml.mem", "w") as text_file:
            text_file.write(clf_data)
        clf = cv2.ml.SVM_create()
        clf = clf.load('traffic_light_images/clf_data.xml.mem')
        return clf
    if IS_SERVER == 2:  # load from file
        clf = cv2.ml.SVM_create()
        clf = clf.load('traffic_light_images/clf_data.xml')
        return clf



# Определение сигнала светофора по изображению
def predict_label(image):
    """
     функция определения сигнала светофора по входному изображению
     Входные данные: rgb изображение
     Выходные данные:
    """
    global GLOB_INIT
    global clf
    
    if GLOB_INIT:
        GLOB_INIT = False
        clf = init_eval()
        print('svm, train, hog(64,32,16,8,9)')
        print('init finish')
       
    hog = get_hog_vector(image)
    
    result = clf.predict(hog)[1][0][0]
    encoded_label = one_hot_encode_num(result)

    return encoded_label
