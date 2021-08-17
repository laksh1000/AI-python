import random
import pickle
import json
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from datetime import datetime,date
import calendar

day_value = False

lematizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('ai_model.h5')

def clean_up_sent(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lematizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sent(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    global day_value
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    # ---------
    dic = return_list[0]
    hmm = dic['intent'] # date is yes or not
    if hmm == "day-date":
        day_value = True
    return return_list


def get_response(intent_list,intents_json):
    tag = intent_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("RUNNING...")

while True:
    message = input("> ")
    ints = predict_class(message)
    if day_value == True:
        date_time = datetime.now()
        my_date = date.today()
        date_string = date_time.strftime("%d/%m/%Y %H:%M:%S \n")
        day_string = calendar.day_name[my_date.weekday()]
        res = "Date and Time : "+date_string +"   Day : "+day_string
        day_value = False

    else:
        res = get_response(ints,intents)

    print(res)


