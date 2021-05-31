# -*- coding: utf-8 -*-

import cv2  # computer vision library
import helpers  # helper functions
import numpy as np
import random
import eval_dev
import glob

GLOB_CLF = 0   # classifier

def save_im_table(MISCLASSIFIED, path):
    im_tab_size = int(np.sqrt(np.ceil(np.sqrt(len(MISCLASSIFIED)))**2))
#    print(im_tab_size)
    if im_tab_size>0:
        im_cur = 1
        blank_image = np.zeros( (64*im_tab_size, 64*im_tab_size), np.uint8)
#        blank_image = np.zeros( (64*im_tab_size, 64*im_tab_size, 3), np.uint8)
        for im,pl,tl  in MISCLASSIFIED:
            pos = ((im_cur-1) % im_tab_size*64, int((im_cur-1) / im_tab_size)*64)
            cv2.putText(im,str(one_hot_encode_inv(pl)),(1,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255), 1,cv2.LINE_AA)
            cv2.putText(im,str(one_hot_encode_inv(tl)),(1,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 1,cv2.LINE_AA)
#            blank_image[pos[0]:pos[0]+64, pos[1]:pos[1]+64, :] = im
            blank_image[pos[0]:pos[0]+64, pos[1]:pos[1]+64] = im
            im_cur +=1
#            print(str(one_hot_encode_inv_num(tl)))
        cv2.imwrite(path, blank_image)
        
def save_folder(MISCLASSIFIED, path):
    i = 0
    for im,pl,tl  in MISCLASSIFIED:
#        cv2.putText(im,str(one_hot_encode_inv(pl)),(1,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255), 1,cv2.LINE_AA)
#        cv2.putText(im,str(one_hot_encode_inv(tl)),(1,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 1,cv2.LINE_AA)
        cv2.imwrite(path+str(i)+str('.jpg'), im)
        i += 1
        print(i,one_hot_encode_inv(tl))

def read_images(path, number):
    """
    Функция для загрузки изображений из папки по пути path.
    """
    names = glob.glob(path + "*.*")
    data = []
    for name in names[:number]:
        image = eval_dev.standardize_input(cv2.imread(name))
        data.append(image)
    return np.array(data)

def train_classifier(clf):
    # load data and set labels
    """
    paths = ['traffic_light_images/training/red/',
                 'traffic_light_images/training/yellow/',
                 'traffic_light_images/training/green/']
    data = np.concatenate((read_images(paths[0], 424), 
                            read_images(paths[1], 663),
                            read_images(paths[2], 2288)), axis=0)
    
    labels = np.concatenate( (np.ones((424, 1), np.int64) * 1,
                            np.ones((663, 1), np.int64) * 2,
                            np.ones((2288, 1), np.int64) * 3))
    """
    paths = ['traffic_light_images/training/red/',
                 'traffic_light_images/training/yellow/',
                 'traffic_light_images/training/green/']
    data = np.concatenate((read_images(paths[0], 507), 
                            read_images(paths[1], 26),
                            read_images(paths[2], 301)), axis=0)
    
    labels = np.concatenate( (np.ones((507, 1), np.int64) * 1,
                            np.ones((26, 1), np.int64) * 2,
                            np.ones((301, 1), np.int64) * 3))
#    """
    # train new classifier
    trainData = eval_dev.get_hog_matrix(data)
    print('data loaded...', trainData.shape)
    
    type_clf = type(clf)
    if type_clf == cv2.ml_SVM:
        clf.setType(cv2.ml.SVM_C_SVC)
        clf.setKernel(cv2.ml.SVM_POLY)
        clf.setDegree(5)
        clf.setC(2.67)
        clf.setGamma(5.383)
        clf.trainAuto(trainData, cv2.ml.ROW_SAMPLE, labels)
    
    if type_clf == cv2.ml_RTrees:
        clf.train(trainData, cv2.ml.ROW_SAMPLE, labels)
    
    
    clf.save('traffic_light_images/clf_data_'+str(type_clf)+'.xml')
    print('train ready...')
    return clf

# Image data directories
def load_data():
    IMAGE_DIR_TRAINING = "traffic_light_images/training/"
    IMAGE_DIR_VALIDATION = "traffic_light_images/val/"
    TRAINING_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)
    VALIDATION_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_VALIDATION)
    return TRAINING_IMAGE_LIST, VALIDATION_IMAGE_LIST


# Перекодировка из текстового названия в массив данных
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

def one_hot_encode_inv(one_hot_encoded):
    label = []
    if one_hot_encoded == [1, 0, 0]:
        label = "red"
    elif one_hot_encoded ==[0, 1, 0]:
        label = "yellow"
    elif one_hot_encoded == [0, 0, 1]:
        label = "green"
    return label

# приведение всего набора изображений к стандартному виду
def standardize(image_list):
    """Функция осуществляет приведение всего набора изображений к стндартному виду

    Входные данные: блок изображений (массив)

    Выходные данные: стандартизированный блок изображений
    """

    # Empty image data array
    standard_list = []
    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]
        # Standardize the image
        standardized_im = eval_dev.standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))

    return standard_list


# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    """Определение точности
    Сравните результаты вашего алгоритма классификации
    с истинными метками и определите точность.
    Входные данные: массив с тестовыми изображениями
    Выходные данные: массив с неправильно классифицированными метками
    Этот код используется для тестирования и не должен изменяться
    """
    global GLOB_CLF
    misclassified_images_labels = []
    for image in test_images:
        im = image[0]
        true_label = image[1]
        assert (len(true_label) == 3), "Метка имеет не верную длинну (3 значения)."

        predicted_label = eval_dev.predict_label(im, GLOB_CLF)
        assert (len(predicted_label) == 3), "Метка имеет не верную длинну (3 значения)."

        if predicted_label != true_label:
            misclassified_images_labels.append((im, predicted_label, true_label))

    return misclassified_images_labels


def main():
    TRAIN_IMAGE_LIST, VALIDATION_IMAGE_LIST = load_data()
    # Standardize the test data
    STANDARDIZED_TRAIN_LIST = standardize(TRAIN_IMAGE_LIST)
    STANDARDIZED_VAL_LIST = standardize(VALIDATION_IMAGE_LIST)

    # Shuffle the standardized test data
    random.shuffle(STANDARDIZED_TRAIN_LIST)
    random.shuffle(STANDARDIZED_VAL_LIST)

#    clf_array = [cv2.ml.SVM_create(), cv2.ml.RTrees_create()]
    clf_array = [cv2.ml.SVM_create()]

    global GLOB_CLF
    for clf in clf_array:
        clf = train_classifier(clf)
        type_clf = type(clf)
        
        GLOB_CLF = clf
        # Find all misclassified images in a given test set
        MISCLASSIFIED = get_misclassified_images(STANDARDIZED_VAL_LIST)
    
        # Accuracy calculations
        total = len(STANDARDIZED_VAL_LIST)
        num_correct = total - len(MISCLASSIFIED)
        accuracy = num_correct / total
    
        print('Accuracy: ' + str(accuracy))
        print("Number of misclassified images = " + str(len(MISCLASSIFIED)) + ' out of ' + str(total))
        
        save_im_table(MISCLASSIFIED, 'traffic_light_images/bad_predict_'+str(type_clf)+'.jpg')
#        save_im_table(MISCLASSIFIED, 'traffic_light_images/good_predict_'+str(type_clf)+'.jpg')


if __name__ == '__main__':
    main()