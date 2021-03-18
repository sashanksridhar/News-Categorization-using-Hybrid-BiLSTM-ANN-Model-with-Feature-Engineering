import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import scale

from sklearn.metrics import classification_report, confusion_matrix
import csv
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter

rowsx = []


with open("trainWords.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(1, len(row)):

            for j in row[i].split("\n"):
                # s+=j+" "

                    # print(j)
                    rows1.append(j)



        del (row[0])
    # print(rows1)

        rowsx.append(rows1)

        # print(len(rows))





t = Tokenizer()

t.fit_on_texts(rowsx)
vocab_size = len(t.word_index) + 1
print(vocab_size)



model = load_model("OnlyWords10.h5")

rowsx_t_1 = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []


with open("testWords.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(1, len(row)):

            for j in row[i].split("\n"):
                # s+=j+" "

                    # print(j)
                    rows1.append(j)


        if row[0] == "cinema":
            y1.append(1)
            y2.append(0)
            y3.append(0)
            y4.append(0)
            y5.append(0)
            y6.append(0)

        elif row[0] == "lifestyle":
            y1.append(0)
            y2.append(1)
            y3.append(0)
            y4.append(0)
            y5.append(0)
            y6.append(0)

        elif row[0] == "crime":
            y1.append(0)
            y2.append(0)
            y3.append(1)
            y4.append(0)
            y5.append(0)
            y6.append(0)

        elif row[0] == "politics":
            y1.append(0)
            y2.append(0)
            y3.append(0)
            y4.append(1)
            y5.append(0)
            y6.append(0)

        elif row[0] == "science":
            y1.append(0)
            y2.append(0)
            y3.append(0)
            y4.append(0)
            y5.append(1)
            y6.append(0)

        elif row[0] == "business":
            y1.append(0)
            y2.append(0)
            y3.append(0)
            y4.append(0)
            y5.append(0)
            y6.append(1)

        del (row[0])
    # print(rows1)

        rowsx_t_1.append(rows1)

        # print(len(rows))






encoded_train_set = t.texts_to_sequences(rowsx_t_1)
SEQ_LEN = 80


padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

train_docs = [list(doc) for doc in padded_train]
x_train = np.array([np.array(token) for token in train_docs])

count = 0


# score = model.evaluate([x_train,x_train1], [y1,y2,y3,y4,y7,y8,y9,y10])
# print(score)
pred = []
# for i in range(0,8):
#     pred.append([])
predicted = model.predict([x_train])
print(len(predicted))
ytrue = []

for i in range(0,len(y1)):
    # ytrue = 0
    values = []
    for j in range(0,6):
        if predicted[j][i][0] > 0.5:
            if j==0:
                values.append("Cinema&Sports")
            elif j==1:
                values.append("Lifestyle")
            elif j == 2:
                values.append("Crime&Legal")
            elif j==4:
                values.append("Politics")
            elif j==5:
                values.append("Science&Engg")
            elif j==6:
                values.append("Business&Economy")
            # pred[j].append(1)
        # else:
            # pred[j].append(0)
    # print(values)
    flag = 0
    if not values:
        flag =1
    if y1[i] == 1:
        ytrue.append("Cinema&Sports")
        if flag == 1:
            values.append("Cinema&Sports")
    if y2[i] == 1:
        ytrue.append("Lifestyle")
        if flag == 1:
            values.append("Lifestyle")
    if y3[i] == 1:
        ytrue.append("Crime&Legal")
        if flag == 1:
            values.append("Crime&Legal")
    if y4[i] == 1:
        ytrue.append("Politics")
        if flag == 1:
            values.append("Politics")
    if y5[i] == 1:
        ytrue.append("Science&Engg")
        if flag == 1:
            values.append("Science&Engg")
    if y6[i] == 1:
        ytrue.append("Business&Economy")
        if flag == 1:
            values.append("Business&Economy")

    pred.append(values[0])
    if ytrue[i] == pred[i]:
        count+=1
print(count)
print((count/len(y1))*100)


import scikitplot
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss
print("=== Confusion Matrix ===")
print(confusion_matrix(ytrue, pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(ytrue, pred))
print('\n')
print("hamming loss")
print(hamming_loss(ytrue, pred))
print('\n')
scikitplot.metrics.plot_confusion_matrix(ytrue, pred)
plt.xticks(rotation=25)
plt.show()