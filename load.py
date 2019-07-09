
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import xlsxwriter
#loading the model
model = load_model('C:/Users/SamSung/Desktop/projects2/trained_model/new2dcnn.h5')

model.compile(loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=['accuracy'])
dir = "C:/Users/SamSung/Downloads/output_videos_for_testing/output_videos_for_testing"

test=[]
X=[]

wb = xlsxwriter.Workbook('C:/Users/SamSung/Desktop/test_results.xlsx')
ws = wb.add_worksheet()

row2 = 1
def testset():
    #print(os.listdir(dir)) list of the file names in dir
    row=1
    for img in os.listdir(dir):
        ws.write(row, 1, img)
        row+=1
        #print (img) img is the name of the file
        img_array = cv2.imread(os.path.join(dir, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array,(108,108))#new_array is one image
        test.append([new_array])


        #classes = model.predict(i)

testset()
for i in test:
    X.append(i)
X=np.asarray(X, dtype=np.int32)
X=X.reshape(-1,108,108,1)
X=np.divide(X,255)
# X is an array of images, 1 img = 1 matrix 108x108

classes = model.predict(X)
#print(classes.shape) shape is 71,3

for i in classes:
    #print(i) i is a probability vector
    #prediction = np.argmax(i)
    #print(np.argmax(i))

    if np.argmax(i) == 0:
        ws.write(row2, 2, "Angry")
        row2+=1
    if np.argmax(i) == 1:
        ws.write(row2, 2, "Hungry")
        row2+=1
    if np.argmax(i) == 2:
        ws.write(row2, 2, "What")
        row2+=1

wb.close()
"""
    if prediction == 0:
        print('Angry')
    if prediction == 1:
        print('Hungry')
    if prediction == 2:
        print('What')
"""

"""
#img = cv2.imread('C:/Users/SamSung/Desktop/projects2/Hungry/Hungry_25.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('C:/Users/SamSung/Desktop/projects2/Angry/Angry_75.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('C:/Users/SamSung/Desktop/projects2/What/What_122.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.resize(img,(108,108))
"""
"""
classes = model.predict(img)
prediction = np.argmax(classes[0])
"""
