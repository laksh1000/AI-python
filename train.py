from os import read
import random
import numpy as np
import pickle
import json

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras import models
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

lematizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_words = [',','.','!','?']

for intent in intents['intents']:
    for pattren in intent['pattrens']:
        word_list = nltk.word_tokenize(pattren)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lematizer.lemmatize(word) for word in words if word not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open("words.pkl","wb"))
pickle.dump(classes, open("classes.pkl","wb"))

training = []
output_emp = [0]*len(classes)

for document in documents:
    bag = []
    word_pattrens = document[0]
    word_pattrens = [lematizer.lemmatize(word.lower()) for word in word_pattrens]

    for word in words:
        bag.append(1) if word in word_pattrens else bag.append(0)

    output_row = list(output_emp)
    output_row[classes.index(document[1])] = 1
    training.append([bag,output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=400, batch_size=5, verbose=1)
model.save('ai_model.h5',hist)
print("DONE")