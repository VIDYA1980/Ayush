import cv2
import numpy as np
from skimage import feature
import pandas as pd
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
def dataset(img_path):
    names = ['area', 'perimeter', 'physiological_length', 'physiological_width', 'aspect_ratio', 'rectangularity', 'circularity',
             'mean_r', 'mean_g', 'mean_b', 'stddev_r', 'stddev_g', 'stddev_b',
             'contrast', 'correlation', 'inverse_difference_moments', 'entropy'
            ]
    data = []  

    main_img = cv2.imread(img_path)
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (25, 25), 0)
    ret_otsu, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((50, 50), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    if len(contours) > 0:
        for cnt in contours:
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if area != 0:  
                rectangularity = w * h / area
                circularity = ((perimeter) ** 2) / area
            else:
                rectangularity = 0
                circularity = 0
    else:
        rectangularity = 0
        circularity = 0

    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0
        
    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)
        
    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)
        
    # Calculate LBP texture features
    lbp = feature.local_binary_pattern(gs, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    vector = [area, perimeter, w, h, aspect_ratio, rectangularity, circularity,
            red_mean, green_mean, blue_mean, red_std, green_std, blue_std,
            hist[0], hist[1], hist[2], hist[3]
            ]
        
    data.append(vector)
    
    df = pd.DataFrame(data, columns=names)
    return df