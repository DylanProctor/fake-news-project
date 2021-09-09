import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import h5py
import pickle

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
model = pickle.load(open('pac_model.p', 'rb'))
df = pd.read_csv('news.csv')
labels = df.label
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size = 0.2, random_state = 7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)

print()
print('Welcome to the fake news detector')
print('---------------------------------')
print('I am a bot that can help you determine the difference between real and fake news')
print('Give me an article you are suspicious of and I will determine its authenticity')
print('I take a text file of the the text in the article')
print('Make sure the text file is in the same directy or you know the path to it')
print('---------------------------------')
print()
time.sleep(2)

while True:
    key = cv2.waitKey(1) & 0xFF
    print("Please enter name of the text file or path to it")
    print("If you wish to exit the program type 'q' instead")
    print()
    file_name = input("")
    print()

    if file_name == 'q':
        break
    try:
        with open(file_name, 'r') as f:
            data = f.read().replace('\n', '')
        text_data = tfidf_vectorizer.transform([data])
        pred = model.predict(text_data)
        print(f'This article is {pred[0]}')
        print()
    except Exception as e:
        print(f'Unable to find file: {e}')
        print()

    time.sleep(2)
    

print()
print('Thank you for using the fake news detector, have a great day')
print()
