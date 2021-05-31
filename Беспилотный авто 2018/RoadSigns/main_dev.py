import helpers
import cv2
import random
import numpy as np
import os
import eval_dev as eval

from imgaug import augmenters as iaa

GLOB_INIT = True

# Загрузка данных из учебных, проверочных и валидационных данных
def load_data():
    """ Формирование обучающего, тренировочного и тестового массива изображений.
    Вспомогательный файл helpers.py формирует массив изображений по заданному пути.

    Выходные данные:
    IMAGE_LIST - массив тренировочных изображений
    TEST_IMAGE_LIST - массив тестовых изображений
    VALIDATION_IMAGE_LIST - массив валидационных изображений (по этому массиву осуществляется
    проверка работы алгоритма
    """

    IMAGE_DIR_TRAINING = "data/training/"
    IMAGE_DIR_VALIDATION = "data/val/"
#    IMAGE_DIR_VALIDATION = "data/aug/"
    IMAGE_DIR_TEST = "data/test/"

    IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)
    TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)
    VALIDATION_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_VALIDATION)

    return IMAGE_LIST, TEST_IMAGE_LIST, VALIDATION_IMAGE_LIST


# приведение входного изображения к стандартному виду
def standardize_input(image):
    """Приведение изображений к стандартному виду. Если вы хотите преобразовать изображение в
    формат, одинаковый для всех изображений, сделайте это здесь. В примере представлено приведение размера к одинаковому для каждого изображения

    Входные данные: изображение

    Выходные данные: стандартизированное изображений.

    """
    ## TODO: Выполните необходимые преобразования изображения для стандартизации, если это необходимо (обрезка, поворот, изменение размера)
    standard_im = np.copy(image)

    standard_im = cv2.resize(standard_im, (64, 64))
    return standard_im


# Перекодировка из текстового названия в массив данных
def one_hot_encode(label):
    """ Функция осуществляет перекодировку текстового входного сигнала
     в массив элементов, соответствующий выходному сигналу

     Входные параметры: текстовая метка (прим.  pedistrain)

     Выходные параметры: метка ввиде массива
     """
    one_hot_encoded = []
    if label == "none":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 0]
    elif label == "pedistrain":
        one_hot_encoded = [1, 0, 0, 0, 0, 0, 0, 0]
    elif label == "no_drive":
        one_hot_encoded = [0, 1, 0, 0, 0, 0, 0, 0]
    elif label == "stop":
        one_hot_encoded = [0, 0, 1, 0, 0, 0, 0, 0]
    elif label == "way_out":
        one_hot_encoded = [0, 0, 0, 1, 0, 0, 0, 0]
    elif label == "no_entry":
        one_hot_encoded = [0, 0, 0, 0, 1, 0, 0, 0]
    elif label == "road_works":
        one_hot_encoded = [0, 0, 0, 0, 0, 1, 0, 0]
    elif label == "parking":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 1, 0]
    elif label == "a_unevenness":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 1]
    return one_hot_encoded

def one_hot_encode_inv(one_hot_encoded):
    label = []
    if one_hot_encoded == [0, 0, 0, 0, 0, 0, 0, 0]:
        label = "none"
    elif one_hot_encoded == [1, 0, 0, 0, 0, 0, 0, 0]:
        label = "pedistrain"
    elif one_hot_encoded == [0, 1, 0, 0, 0, 0, 0, 0]:
        label = "no_drive"
    elif one_hot_encoded == [0, 0, 1, 0, 0, 0, 0, 0]:
        label = "stop"
    elif one_hot_encoded == [0, 0, 0, 1, 0, 0, 0, 0]:
        label = "way_out"
    elif one_hot_encoded == [0, 0, 0, 0, 1, 0, 0, 0]:
        label = "no_entry"
    elif one_hot_encoded == [0, 0, 0, 0, 0, 1, 0, 0]:
        label = "road_works"
    elif one_hot_encoded == [0, 0, 0, 0, 0, 0, 1, 0]:
        label = "parking"
    elif one_hot_encoded == [0, 0, 0, 0, 0, 0, 0, 1]:
        label = "a_unevenness"
    return label

def one_hot_encode_inv_num(one_hot_encoded):
    label = []
    if one_hot_encoded == [0, 0, 0, 0, 0, 0, 0, 0]:
        label = 0
    elif one_hot_encoded == [1, 0, 0, 0, 0, 0, 0, 0]:
        label = 1
    elif one_hot_encoded == [0, 1, 0, 0, 0, 0, 0, 0]:
        label = 2
    elif one_hot_encoded == [0, 0, 1, 0, 0, 0, 0, 0]:
        label = 3
    elif one_hot_encoded == [0, 0, 0, 1, 0, 0, 0, 0]:
        label = 4
    elif one_hot_encoded == [0, 0, 0, 0, 1, 0, 0, 0]:
        label = 5
    elif one_hot_encoded == [0, 0, 0, 0, 0, 1, 0, 0]:
        label = 6
    elif one_hot_encoded == [0, 0, 0, 0, 0, 0, 1, 0]:
        label = 7
    elif one_hot_encoded == [0, 0, 0, 0, 0, 0, 0, 1]:
        label = 8
    return label


# приведение всего набора изображений к стандартному виду
def standardize(image_list):
    """Функция осуществляет приведение всего набора изображений к стндартному виду

    Входные данные: блок изображений (массив)

    Выходные данные: стандартизированный блок изображений
    """

    standard_list = []

    for item in image_list:
        image = item[0]
        label = item[1]

        # стандартизация каждого изображения
        standardized_im = standardize_input(image)

        # перекодировка из названия в массив
        one_hot_label = one_hot_encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))

    return standard_list


# Получение списка неклассифицированных изображений
def get_misclassified_images(test_images):
    """Определение точности
    Сравните результаты вашего алгоритма классификации
    с истинными метками и определите точность.

    Входные данные: массив с тестовыми изображениями
    Выходные данные: массив с неправильно классифицированными метками

    Этот код используется для тестирования и не должен изменяться
    """
    misclassified_images_labels = []
    classified_images_labels = []
    # Классификация каждого изображения и сравенение с реальной меткой
    for image in test_images:
        # получение изображения и метки
        im = image[0]
        true_label = image[1]
        
        # вывожу только нужные мне классы
#        if not (true_label == one_hot_encode('parking') or true_label == one_hot_encode('no_drive') ):
#        if true_label != one_hot_encode('a_unevenness'):
#            continue
        
        # метки должны быть в виде массива
        assert (len(true_label) == 8), "Метка имеет не верную длинну (8 значений)"

        # Получение метки из написанного Вами классификатора
#        predicted_label, dist, len_good = eval.predict_label(im)
        predicted_label = eval.predict_label(im)
#        global GLOB_INIT
#        predicted_label = eval.predict_label(im, GLOB_INIT)
#        if GLOB_INIT:
#            GLOB_INIT = False
        assert (len(predicted_label) == 8), "Метка имеет не верную длинну (8 значений)"

        # Сравнение реальной и предсказанной метки
        if (predicted_label != true_label):
            # Если значения меток не совпадают, то изображение помечается как неклассифицированное
            misclassified_images_labels.append((im, predicted_label, true_label))
#            print('0', one_hot_encode_inv_num(predicted_label), dist, len_good)
        else:
            classified_images_labels.append((im, predicted_label, true_label))
#            print('1', one_hot_encode_inv_num(predicted_label), dist, len_good)
    
    # Возвращение неклассифицированных изображений [image, predicted_label, true_label] values
    return misclassified_images_labels, classified_images_labels

def save_im_table(MISCLASSIFIED, path):
    im_tab_size = int(np.sqrt(np.ceil(np.sqrt(len(MISCLASSIFIED)))**2))
#    print(im_tab_size)
    if im_tab_size>0:
        im_cur = 1
        blank_image = np.zeros( (64*im_tab_size, 64*im_tab_size, 3), np.uint8)
        for im,pl,tl  in MISCLASSIFIED:
            pos = ((im_cur-1) % im_tab_size*64, int((im_cur-1) / im_tab_size)*64)
            cv2.putText(im,str(one_hot_encode_inv(pl)),(1,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255), 1,cv2.LINE_AA)
            cv2.putText(im,str(one_hot_encode_inv(tl)),(1,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 1,cv2.LINE_AA)
            blank_image[pos[0]:pos[0]+64, pos[1]:pos[1]+64, :] = im
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

def main():
    ## загрузка учебных изображений
    IMAGE_LIST, TEST_IMAGE_LIST, VALIDATION_IMAGE_LIST = load_data()
    ## Отображение изображения, приведенного к стандартному виду (размер 32х32) и его метки (массива чисел)
    STANDARDIZED_LIST = standardize(IMAGE_LIST)

    STANDARDIZED_VAL_LIST = standardize(VALIDATION_IMAGE_LIST)
    random.seed(a=1)
    random.shuffle(STANDARDIZED_VAL_LIST)

    # [1]
    # Пример работы с изображением из списка тренировочных изображений

    # Получим изображение и его метку
#    standart_image = STANDARDIZED_LIST[1520][0]
#    standart_image_label = STANDARDIZED_LIST[1520][1]
#    #
#    predict_image_label = eval.predict_label(standart_image)
#
#    print("Реальный класс изображения: {} Предсказанный класс изображения {}".format(standart_image_label,
#                                                                                      predict_image_label))
    # Чтобы отобразить изображение раскомментируйте 2 строки ниже:
#    cv2.imshow("standart_test_im", standart_image)
#    cv2.waitKey(0)

    # [2]
    ## сравнение учебных и тестовых изображений
    # поиск неклассифицированных изображений в тестовой выборке
    MISCLASSIFIED, CLASSIFIED = get_misclassified_images(STANDARDIZED_VAL_LIST)
    # Вы также можете увидеть изображения, которые не удалось классифицировать раскомментировав 2 строчки ниже:
#    cv2.imshow("MISCLASSIFIED", MISCLASSIFIED[10][0])
#    cv2.waitKey(0)
    
    save_folder(MISCLASSIFIED, 'data/bad_predict/')
    blank_image = np.zeros( (2, 2), np.uint8)
    cv2.imwrite('data/bad_predict_img.jpg', blank_image)
    cv2.imwrite('data/good_predict_img.jpg', blank_image)
    save_im_table(MISCLASSIFIED, 'data/bad_predict_img.jpg')
    save_im_table(CLASSIFIED, 'data/good_predict_img.jpg')

    # вычисление точности
    total = len(STANDARDIZED_VAL_LIST)
    num_correct = total - len(MISCLASSIFIED)
    accuracy = num_correct / total

    print('Точность: ' + str(accuracy))
    print("Число не распознанных изображений = " + str(len(MISCLASSIFIED)) + ' из ' + str(total))

def augment_images(path):
    IMAGE_LIST, TEST_IMAGE_LIST, VALIDATION_IMAGE_LIST = load_data()
#    seq = iaa.Sequential([ iaa.PiecewiseAffine(scale=(0,1))])
    seq = iaa.PiecewiseAffine(scale=(0,0.1))
    aug_count = 5 # сколько нужно аугментированных изображений
    
    i = 0
    for im in VALIDATION_IMAGE_LIST:
        for j in range(0,aug_count):
            im_aug = seq.augment_image(im[0])
#            cv2.imwrite(path+str(im[1])+str(i)+'_'+str(j)+'.jpg', im_aug)
#            save to path/label_of_image/filename.jpg
            cv2.imwrite(path+str(im[1])+'/'+str(i)+'_'+str(j)+'.jpg', im_aug)
        i += 1
    print('augment finished')
    """
    for j in range(0,5):
        seq = iaa.PiecewiseAffine(scale=(0,0.1))
        im = cv2.imread('data/standards/pedistrain.jpg')
        im = seq.augment_image(im)
        cv2.imshow('im', im)
        cv2.waitKey(0)
    """

if __name__ == '__main__':
#    """
    #    for i in range(1,5):
    main()
#        GLOB_INIT =True
    cv2.destroyAllWindows()
#    """
    augment_images('data/aug/')
