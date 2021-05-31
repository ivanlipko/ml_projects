import helpers
import cv2
import random
import numpy as np
import glob

GLOB_INIT = True
GLOB_ITER = 2

hog_file = """_"""

def get_features_SIFT(img):
    detector = cv2.xfeatures2d.SIFT_create()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    kp, des = detector.detectAndCompute(img, None)
    img = cv2.drawKeypoints(img,kp,img, color=(0,255,0), flags=0)
    return kp, des, img

def predict_one_SIFT(des_ideal, des_dest, label, MATCH_THRESHOLD):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_ideal,des_dest, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])

#    print('len of good', len(good))
    dist = 0;
    for g in good: #range(1, MATCH_THRESHOLD):
        dist += g[0].distance

    predicted_label = 0
    if(len(good) >= MATCH_THRESHOLD):
        predicted_label = one_hot_encode(label)
#        print("Threshold!")
#    cv2.waitKey(0)
    return predicted_label, good, dist

def predict_no_entry(des_a_unevenness, des):
    predicted_label, good, dist = predict_one_SIFT(des_a_unevenness, des, "road_works", 5)
    if predicted_label == 0 or len(good)==0:
        predicted_label = [0, 0, 0, 0, 0, 0, 0, 0]
    elif (dist/len(good))>10:
        predicted_label = one_hot_encode("road_works")
    return predicted_label, dist, len(good)

#def plot_hist(hist):
##    len_hist = len(hist)
#    cv2.calcHist()
#    blank= np.zeros( (64, 128), np.uint8)
#    
#    cv2.rectangle()

def deskew(img):
#    from https://docs.opencv.org/3.4.1/dd/d3b/tutorial_py_svm_opencv.html
    SZ=64
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_NEAREST)
    return img

def filter_image(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (64, 64))
#    cv2.imshow('im', im)

#    im = np.concatenate((im[:,:,0], im[:,:,1], im[:,:,2]), axis=0)

#    cv2.imshow('im_t', im_t)
#    im = deskew(im)
    
#    cv2.imshow('deskew', im)

#    im = cv2.blur(im, (5,5))
#    cv2.waitKey(0)
    return im

def read_images(path, number):
    """
    Функция для загрузки изображений из папки по пути path.
    """
    names = glob.glob(path + "*.*")
    data = []
    for name in names[:number]:
#        image = cv2.resize(cv2.imread(name, cv2.IMREAD_GRAYSCALE), (64, 64))
        image = filter_image(cv2.imread(name))
#        data.append(cv2.pyrUp(image[16:48, 16:48]))
        data.append(image)
    return np.array(data)

def get_hog_vector(im):
    """
    Функция, возвращающая вектор HOG для одного grayscale-изображения размера (64 x 64). 
    Возвращаемый вектор имеет вид (1 x длина вектора HOG).
    """
#    hog_descriptor = cv2.HOGDescriptor((64, 64), #winsize
#                                       (32, 32), #blocksize
#                                       (16, 16), #blockstride
#                                       (8, 8),   #cellsize
#                                       9)        #nbins
    hog_descriptor = cv2.HOGDescriptor((64, 64), #winsize
                                       (32, 32), #blocksize
                                       (16, 16), #blockstride
                                       (8,8),   #cellsize
                                       12)        #nbins
    return hog_descriptor.compute(im).T

def get_hog_matrix(array_images):
    """
    Функция, возвращающая матрицу, i-я строка которой является вектором HOG для i-го изображения 
    входного массива.
    Возвращаемая матрица имеет вид (число изображений x длина вектора HOG)
    """
    matrix = np.array([get_hog_vector(image.reshape(64, 64)).flatten() for image in array_images]) 
#    matrix = np.array([get_hog_vector(image.reshape(192, 64)).flatten() for image in array_images]) 
    return matrix

clf = cv2.ml.SVM_create()
#clf = cv2.ml.RTrees_create()
#clf = cv2.ml.KNearest_create()
#clf = cv2.ml.Boost_create()

def train_HOG():
    '''
    Загружаем тренировочную выборку и обучаем SVM-классификатор
    '''
    global clf
    
    IS_SERVER = 1
    if IS_SERVER == 1:   # load from memory
        with open("data/svm_data.xml.mem", "w") as text_file:
            text_file.write(hog_file)
        clf = cv2.ml.SVM_create()
        clf = clf.load('data/svm_data.xml.mem')
        return clf
    if IS_SERVER == 2:  # load from file
        clf = cv2.ml.SVM_create()
        clf = clf.load('data/svm_data.xml')
        return clf
    
    # train new classifier
    """
    paths = [    'data/training_val/none/',
                 'data/training_val/pedistrain/',
                 'data/training_val/no_drive/',
                 'data/training_val/stop/',
                 'data/training_val/way_out/',
                 'data/training_val/no_entry/',
                 'data/training_val/road_works/',
                 'data/training_val/parking/',
                 'data/training_val/a_unevenness/']
    data = np.concatenate((read_images(paths[0], 449), 
                            read_images(paths[1], 683),
                            read_images(paths[2], 2320),
                            read_images(paths[3], 2357),
                            read_images(paths[4], 2474),
                            read_images(paths[5], 2609),
                            read_images(paths[6], 2445),
                            read_images(paths[7], 59),
                            read_images(paths[8], 27)    ), axis=0)
    
    labels = np.concatenate( (np.ones((449, 1), np.int64) * 0,
                            np.ones((683, 1), np.int64) * 1,
                            np.ones((2320, 1), np.int64) * 2,
                            np.ones((2357, 1), np.int64) * 3,
                            np.ones((2474, 1), np.int64) * 4,
                            np.ones((2609, 1), np.int64) * 5,
                            np.ones((2445, 1), np.int64) * 6,
                            np.ones((59, 1), np.int64) * 7,
                            np.ones((27, 1), np.int64) * 8 ))
    "" "
    paths = ['data/training/none/',
                 'data/training/pedistrain/',
                 'data/training/no_drive/',
                 'data/training/stop/',
                 'data/training/way_out/',
                 'data/training/no_entry/',
                 'data/training/road_works/',
                 'data/training/parking/',
                 'data/training/a_unevenness/']
    data = np.concatenate((read_images(paths[0], 424), 
                            read_images(paths[1], 663),
                            read_images(paths[2], 2288),
                            read_images(paths[3], 2333),
                            read_images(paths[4], 2443),
                            read_images(paths[5], 2585),
                            read_images(paths[6], 2413),
                            read_images(paths[7], 49),
                            read_images(paths[8], 22)    ), axis=0)
    
    labels = np.concatenate( (np.ones((424, 1), np.int64) * 0,
                            np.ones((663, 1), np.int64) * 1,
                            np.ones((2288, 1), np.int64) * 2,
                            np.ones((2333, 1), np.int64) * 3,
                            np.ones((2443, 1), np.int64) * 4,
                            np.ones((2585, 1), np.int64) * 5,
                            np.ones((2413, 1), np.int64) * 6,
                            np.ones((49, 1), np.int64) * 7,
                            np.ones((22, 1), np.int64) * 8 ))
#    """
    paths = [    'data/aug/none/',
                 'data/aug/pedistrain/',
                 'data/aug/no_drive/',
                 'data/aug/stop/',
                 'data/aug/way_out/',
                 'data/aug/no_entry/',
                 'data/aug/road_works/',
                 'data/aug/parking/',
                 'data/aug/a_unevenness/']
    data = np.concatenate((read_images(paths[0], 9375), 
                            read_images(paths[1], 3600),
                            read_images(paths[2], 5760),
                            read_images(paths[3], 4680),
                            read_images(paths[4], 5580),
                            read_images(paths[5], 3548),
                            read_images(paths[6], 960),
                            read_images(paths[7], 300),
                            read_images(paths[8], 150)    ), axis=0)
    
    labels = np.concatenate( (np.ones((9375, 1), np.int64) * 0,
                            np.ones((3600, 1), np.int64) * 1,
                            np.ones((5760, 1), np.int64) * 2,
                            np.ones((4680, 1), np.int64) * 3,
                            np.ones((5580, 1), np.int64) * 4,
                            np.ones((3548, 1), np.int64) * 5,
                            np.ones((960, 1), np.int64) * 6,
                            np.ones((300, 1), np.int64) * 7,
                            np.ones((150, 1), np.int64) * 8 ))
    
    
    trainData = get_hog_matrix(data)
    print('hog ready...', trainData.shape)

#    """
    clf = cv2.ml.SVM_create()
#        clf.setKernel(cv2.ml.SVM_LINEAR)

    clf.setKernel(cv2.ml.SVM_POLY)
    clf.setDegree(5)
    
#    global GLOB_ITER
#    print('GLOB_ITER is', GLOB_ITER)
#    clf.setDegree(GLOB_ITER)


#    clf.setKernel(cv2.ml.SVM_RBF) # очень медленно и ошибок 126 на 9 nbins? 106 on 3 nbins
#    clf.setKernel(cv2.ml.SVM_INTER)
    
    clf.setType(cv2.ml.SVM_C_SVC)
    clf.setC(2.67)
    clf.setGamma(5.383)
#    """

#    clf = cv2.ml.RTrees_create()
    
    clf.train(trainData, cv2.ml.ROW_SAMPLE, labels)
#    clf.trainAuto(trainData, cv2.ml.ROW_SAMPLE, labels)
    
#    Cgrid = cv2.ml.ParamGrid_create(0.1, 5, 1.1)
#    gammaGrid = cv2.ml.ParamGrid_create(0,0,0) #cv2.ml.SVM_getDefaultGridPtr(cv2.ml.SVM_GAMMA)
#    pGrid = cv2.ml.ParamGrid_create(0,0,0)
#    nuGrid = cv2.ml.ParamGrid_create(0,0,0)
#    coeffGrid = cv2.ml.ParamGrid_create(0,0,0)
#    degreeGrid = cv2.ml.ParamGrid_create(0,0,0)
#    clf.trainAuto(trainData, cv2.ml.ROW_SAMPLE, labels, kFold=10, Cgrid=Cgrid, gammaGrid=gammaGrid, pGrid=pGrid, nuGrid=nuGrid, coeffGrid=coeffGrid, degreeGrid=degreeGrid)
    
    clf.save('data/svm_data.xml')
#    clf.save('data/svm_data.xml' + str(GLOB_ITER))
#    GLOB_ITER += 1
    print('train ready...')
    return clf

def one_hot_encode(label):
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

def one_hot_encode_num(num):
    one_hot_encoded = []
    if num == 0:
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 0]
    elif num == 1:
        one_hot_encoded = [1, 0, 0, 0, 0, 0, 0, 0]
    elif num == 2:
        one_hot_encoded = [0, 1, 0, 0, 0, 0, 0, 0]
    elif num == 3:
        one_hot_encoded = [0, 0, 1, 0, 0, 0, 0, 0]
    elif num == 4:
        one_hot_encoded = [0, 0, 0, 1, 0, 0, 0, 0]
    elif num == 5:
        one_hot_encoded = [0, 0, 0, 0, 1, 0, 0, 0]
    elif num == 6:
        one_hot_encoded = [0, 0, 0, 0, 0, 1, 0, 0]
    elif num == 7:
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 1, 0]
    elif num == 8:
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 1]
    return one_hot_encoded

def predict_validation():
    VAL_PATH = 'data/val/stop/'
    VAL_COUNT = 10
    data_val = read_images(VAL_PATH, VAL_COUNT)
    mat_val = get_hog_matrix(data_val)
    
    labels_val = np.ones(shape=(len(data_val), 1)) # np.array c единичками 
    print("Read %d images, \ndata_pos shape is %s" % (POS_COUNT, data_pos.shape))
    result = clf.predict(mat_val)[1]

def predict_label(image):
#def predict_label(image, GLOB_INIT):
    global GLOB_INIT
    global clf
    
    if GLOB_INIT:
        GLOB_INIT = False
        clf = train_HOG()
        print('init end')
        
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    image = cv2.resize(image, (64, 64))
    image = filter_image(image)
    hog = get_hog_vector(image)
    
#    print(type(hog), hog.shape)
    result = clf.predict(hog)[1][0][0]
#    print(result)

    predicted_label = one_hot_encode_num(result)
    return predicted_label#, dist, len_good
    