import numpy as np
import cv2

actions = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain', 'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 'ta', 'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay']

arabic_dict = {
    'ain': 'ع',
    'al': 'ال',
    'aleff': 'أ',
    'bb': 'ب',
    'dal': 'د',
    'dha': 'ذ',
    'dhad': 'ض',
    'fa': 'ف',
    'gaaf': 'غ',
    'ghain': 'غ',
    'ha': 'هـ',
    'haa': 'ح',
    'jeem': 'ج',
    'kaaf': 'ك',
    'khaa': 'خ',
    'la': 'ل',
    'laam': 'ل',
    'meem': 'م',
    'nun': 'ن',
    'ra': 'ر',
    'saad': 'ص',
    'seen': 'س',
    'sheen': 'ش',
    'ta': 'ت',
    'taa': 'ط',
    'thaa': 'ث',
    'thal': 'ذ',
    'toot': 'ت',
    'waw': 'و',
    'ya': 'ي',
    'yaa': 'ي',
    'zay': 'ز'
}


from keras.models import load_model
model = load_model("cnn_model.h5")


def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)     # CONVERT TO GRAYSCALE
    img = equalize(img)      # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img

def predict_alphabet(path):
    img = cv2.imread(path)
     
    # PROCESS IMAGE
    img = cv2.resize(img, (64, 64))
    img = preprocessing(img)
    img = img.reshape(1, 64, 64, 1)

    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)[0]
    res = actions[classIndex]

    return {"result_alphabet": arabic_dict[res]}
