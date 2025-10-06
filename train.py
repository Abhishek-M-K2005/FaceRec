from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
import pickle
import cv2
import numpy as np

X = []
Y = []

with open("./data/names.pkl") as f:
    X = pickle.load(f)
    
with open (".data/names.pkl") as f:
    Y = pickle.load(f)
    
train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=42)

model = KNeighborsClassifier
model.fit(train_x, train_y)

