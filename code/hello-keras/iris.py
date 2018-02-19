import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, Callback
from keras.metrics import mae, categorical_accuracy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


iris = datasets.load_iris()
X = iris.data
Y = to_categorical(iris.target)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print(len(X_train), len(X_test), len(Y_train), len(Y_test))

model = Sequential()
model.add(Dense(32, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[mae, categorical_accuracy])

model.fit(X_train, Y_train, epochs=160, batch_size=100, validation_data=(X_test, Y_test))

# Print predictions vs. targets for test data
for py, y in zip(model.predict(X_test), Y_test):
    print(np.argmax(py), np.argmax(y))
