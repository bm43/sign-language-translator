from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import array
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import os, cv2, random
import matplotlib.pyplot as plt

dir = "C:/Users/SamSung/Desktop/projects2"
labels = ["Angry", "Hungry", "What"]
# angry = 0, hungry = 1 , what = 2

X=[]
y=[]
tdata=[]
def cr_tdata():
    for label in labels:
        path = os.path.join(dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(108,108))
                tdata.append([new_array, class_num])
            except:
                pass

cr_tdata()
random.shuffle(tdata)
for features, label in tdata:
    X.append(features)
    y.append(label)
X=np.asarray(X, dtype=np.int32)
X=X.reshape(-1,108,108,1)
X=np.divide(X,255)
# normalisation

model= Sequential()

model.add( Conv2D(32,(3,3), input_shape = X.shape[1:]) )
model.add(Activation("relu"))
model.add( Conv2D(32, (3,3)) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(units=128))
model.add(Dropout(0.22))
model.add(Dense(units=3))
model.add(Activation('softmax'))

model.compile(loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=['accuracy'])

#sparse_categorical = integer labels

history=model.fit(X,y, batch_size=32, epochs=2, validation_split=0.1)

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#model.save('C:/Users/SamSung/Desktop/projects2/trained_model/new2dcnn.h5')
