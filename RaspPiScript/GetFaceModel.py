import os
import json
import requests
from PIL import Image
from keras.models import load_model
from keras_facenet import FaceNet
import numpy as np
from numpy import asarray
from numpy import expand_dims
import pickle
import cv2
import numpy as np
import datetime
import requests
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#remember to change ngrok url every time 
url = "'https://fcb0-27-125-249-39.ngrok-free.app/api/" #url for accessing the custom RestAPIs by exposing the localhost to ngrok and using the provided links given by ngrok
url2 = "https://fcb0-27-125-249-39.ngrok-free.app/api/"
#get faces data files from server
faceurl = url + "api/faces'"
faceembedurl = url + "api/facesembeds'"
facesdownload = "sudo curl -X 'GET' " + faceurl + " -H 'accept: application/json' --output registered-faces-db.npz"
facesembeddownload = "sudo curl -X 'GET' " + faceembedurl + " -H 'accept: application/json' --output registered-faces-db-embeddings.npz"
print(facesdownload)
print(facesembeddownload)
os.popen(facesdownload)
os.popen(facesembeddownload)
