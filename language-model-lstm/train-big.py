import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint

"""
Create data + model
"""

with open('alice.txt') as fin:
    raw_text = fin.read().lower()
print (raw_text[0:30], '...')
distinct_chars = sorted(list(set(raw_text)))
n_chars = len(raw_text)
n_distinct = len(distinct_chars)
print ("Total Characters: ", n_chars)
print ("Total distinct: ", n_distinct)

int_to_char = dict([(i, c) for i, c in enumerate(distinct_chars)])
char_to_oh = dict([(c, np.identity(n_distinct)[i: i+1][0]) for i, c in enumerate(distinct_chars)])

window_size = 100
data_X = []
data_y = []
for i in range(0, n_chars - window_size, 1):
    seq_in = [char_to_oh[c] for c in raw_text[i: i + window_size]]
    seq_out = char_to_oh[raw_text[i+window_size]]
    data_X.append(seq_in)
    data_y.append(seq_out)

n_patterns = len(data_X)
print ("Total Patterns: ", n_patterns)

# Use one-hot encoded
X = np.reshape(data_X, (n_patterns, window_size, n_distinct))
y = np.reshape(data_y, (n_patterns, n_distinct))

print('Shape X:', X.shape)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

"""
Training
"""

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=200, batch_size=128, callbacks=callbacks_list)

"""
Use model

import sys
import random
# load the network weights
filename = "weights-improvement-13-0.9761.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

def oh_to_char(oh):
    return int_to_char[np.argmax(oh)]

# pick a random seed

pattern = [oh for oh in random.sample(data_X, 1)[0]]
print ('---')
print ('Seed:')
print ('"', ''.join([oh_to_char(oh) for oh in pattern]), '"')

print('Generated text:')
# generate characters
for i in range(1000):
    X_next = np.reshape(pattern, (1, window_size, n_distinct))
    prediction = model.predict(X_next, verbose=0)
    index = np.argmax(prediction)
    predicted_char = int_to_char[index]
    sys.stdout.write(predicted_char)
    padding = char_to_oh[predicted_char]
    pattern.append(padding)
    pattern = pattern[1:]
print ()
print ('Done')
"""
