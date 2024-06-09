import cv2 as cv
import numpy as np

import os
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import pickle
from keras.models import load_model
from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_json

old = None 
vgg_model = load_model('model/model.h5')
class_labels = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)',
'Speed limit (80km/h)', 'Speed limit(100km/h)', 'End of speed limit (80 km/h)', 'Speed limit(100km/h)', 'Speed limit(120km/h)', 'No passing', 'Stop', 'No Entry', 
'General caution', 'Traffic signals' ]

def cannyDetection(img):
    grayImg = cv.cvtColor(img, cv.COLOR_RGB2GRAY) 
    blurImg = cv.GaussianBlur(grayImg, (5,5), 0)
    cannyImg = cv.Canny(blurImg, 50, 150)
    return cannyImg


def segmentDetection(img):
    height = img.shape[0]
    polygons= np.array([[(0, height), (800, height), (380, 290)]])
    maskImg= np.zeros_like(img)
    cv.fillPoly(maskImg, polygons ,255)
    segmentImg =cv.bitwise_and(img, maskImg)
    return segmentImg


def calculateLines(frame, lines):
    left = []
    right = []
    for line in lines:
        x1, x2, y1, y2 = line.reshape(4)
        parameters = np.polyfit((x1, y1), (x2, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        if slope < 0:
            left.append((slope, y_intercept))
        else :
            right.append((slope, y_intercept))
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    left_line = calculateCoordinates(frame, left_avg)
    right_line = calculateCoordinates(frame, right_avg)
    return np.array([left_line, right_line])


def calculateCoordinates(frame, parameters):
    global old
    if old is None :
        old = parameters
    if np.isnan(parameters.any()) == False:
        parameters = old
    slope, intercept= parameters
    y1 = frame.shape[0]
    y2 = int(y1- 150)
    x1 = int((y1- intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def visualizeLines(frame, lines):
    lines_visualize = np.zeros_like(frame)
    if lines is not None :
        for x1, y1, x2, y2 in lines:
            cv.line(lines_visualize, (x1, y1), (x2, y2),(0, 255, 0), 5)
    return lines_visualize

def detectPotholes(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    edges = cv.Canny(blur,50,150)
    _, binary = cv.threshold(blur, 60, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(edges, cv.RETR_TREE ,cv.CHAIN_APPROX_SIMPLE)
    potholes = []
    for contour in contours:
        if cv.contourArea(contour) > 500:
            x, y, w, h = cv.boundingRect(contour)
            potholes.append((x, y, w, h))
    return potholes

def detectUnstructuredRoad(frame):
    height,width = frame.shape[:2]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150)
    roi_edges = segmentDetection(edges)
    lines = cv.HoughLinesP(roi_edges,1,np.pi/180,threshold = 100, minLineLength = 100, maxLineGap = 50)

    if lines is None:
        return True
    
    left_lines = [line for line in lines if line[0][0] < width // 2 and line [0][2] < width //2]
    right_lines = [line for line in lines if line[0][0] > width // 2 and line [0][2] > width //2]

    if len(left_lines) < 5 or len(right_lines) < 5:
        return True
    
    return False