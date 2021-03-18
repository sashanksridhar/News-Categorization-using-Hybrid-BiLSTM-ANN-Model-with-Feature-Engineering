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




rowsx1 = []


with open("trainHypernyms.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)



        # del (row[0])
    # print(rows1)

        rowsx1.append(rows1)

rowsx2 = []


with open("trainHyponyms.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)



        # del (row[0])
    # print(rows1)

        rowsx2.append(rows1)

rowsx1L2 = []


with open("trainHypernymsL2.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)



        # del (row[0])
    # print(rows1)

        rowsx1L2.append(rows1)

rowsx2L2 = []


with open("trainHyponymsL2.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)



        # del (row[0])
    # print(rows1)

        rowsx2L2.append(rows1)

rowsx1L3 = []


with open("trainHypernymsL3.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)



        # del (row[0])
    # print(rows1)

        rowsx1L3.append(rows1)

rowsx2L3 = []


with open("trainHyponymsL3.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)



        # del (row[0])
    # print(rows1)

        rowsx2L3.append(rows1)


t = Tokenizer()

t.fit_on_texts(rowsx)
vocab_size = len(t.word_index) + 1
print(vocab_size)

embeddings = Word2Vec(size=200, min_count=3)
embeddings.build_vocab([sentence for sentence in rowsx1])
embeddings.train([sentence for sentence in rowsx1],
                 total_examples=embeddings.corpus_count,
                 epochs=embeddings.epochs)
# print(embeddings.wv.most_similar('economy'))

gen_tfidf = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
matrix = gen_tfidf.fit_transform([sentence   for sentence in rowsx1])
tfidf_map = dict(zip(gen_tfidf.get_feature_names(), gen_tfidf.idf_))
print(len(tfidf_map))

embeddings1 = Word2Vec(size=200, min_count=3)
embeddings1.build_vocab([sentence for sentence in rowsx2])
embeddings1.train([sentence for sentence in rowsx2],
                 total_examples=embeddings1.corpus_count,
                 epochs=embeddings1.epochs)
# print(embeddings.wv.most_similar('economy'))

gen_tfidf1 = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
matrix1 = gen_tfidf1.fit_transform([sentence   for sentence in rowsx2])
tfidf_map1 = dict(zip(gen_tfidf1.get_feature_names(), gen_tfidf1.idf_))
print(len(tfidf_map1))

embeddingsL2 = Word2Vec(size=200, min_count=3)
embeddingsL2.build_vocab([sentence for sentence in rowsx1L2])
embeddingsL2.train([sentence for sentence in rowsx1L2],
                 total_examples=embeddingsL2.corpus_count,
                 epochs=embeddingsL2.epochs)
# print(embeddings.wv.most_similar('economy'))

gen_tfidfL2 = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
matrixL2 = gen_tfidfL2.fit_transform([sentence   for sentence in rowsx1L2])
tfidf_mapL2 = dict(zip(gen_tfidfL2.get_feature_names(), gen_tfidfL2.idf_))
print(len(tfidf_mapL2))

embeddings1L2 = Word2Vec(size=200, min_count=3)
embeddings1L2.build_vocab([sentence for sentence in rowsx2L2])
embeddings1L2.train([sentence for sentence in rowsx2L2],
                 total_examples=embeddings1L2.corpus_count,
                 epochs=embeddings1L2.epochs)
# print(embeddings.wv.most_similar('economy'))

gen_tfidf1L2 = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
matrix1L2 = gen_tfidf1L2.fit_transform([sentence   for sentence in rowsx2L2])
tfidf_map1L2 = dict(zip(gen_tfidf1L2.get_feature_names(), gen_tfidf1L2.idf_))
print(len(tfidf_map1L2))

embeddingsL3 = Word2Vec(size=200, min_count=3)
embeddingsL3.build_vocab([sentence for sentence in rowsx1L3])
embeddingsL3.train([sentence for sentence in rowsx1L3],
                 total_examples=embeddingsL3.corpus_count,
                 epochs=embeddingsL3.epochs)
# print(embeddings.wv.most_similar('economy'))

gen_tfidfL3 = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
matrixL3 = gen_tfidfL3.fit_transform([sentence   for sentence in rowsx1L3])
tfidf_mapL3 = dict(zip(gen_tfidfL3.get_feature_names(), gen_tfidfL3.idf_))
print(len(tfidf_mapL3))

embeddings1L3 = Word2Vec(size=200, min_count=3)
embeddings1L3.build_vocab([sentence for sentence in rowsx2L3])
embeddings1L3.train([sentence for sentence in rowsx2L3],
                 total_examples=embeddings1L3.corpus_count,
                 epochs=embeddings1L3.epochs)
# print(embeddings.wv.most_similar('economy'))

gen_tfidf1L3 = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
matrix1L3 = gen_tfidf1L3.fit_transform([sentence   for sentence in rowsx2L3])
tfidf_map1L3 = dict(zip(gen_tfidf1L3.get_feature_names(), gen_tfidf1L3.idf_))
print(len(tfidf_map1L3))



def encode_sentence(tokens, emb_size):
    _vector = np.zeros((1, emb_size))
    length = 0
    for word in tokens:
        try:
            _vector += embeddings.wv[word].reshape((1, emb_size)) * tfidf_map[word]
            length += 1
        except KeyError:
            continue
        break

    if length > 0:
        _vector /= length

    return _vector

def encode_sentence1(tokens, emb_size):
    _vector = np.zeros((1, emb_size))
    length = 0
    for word in tokens:
        try:
            _vector += embeddings1.wv[word].reshape((1, emb_size)) * tfidf_map1[word]
            length += 1
        except KeyError:
            continue
        break

    if length > 0:
        _vector /= length

    return _vector

def encode_sentenceL2(tokens, emb_size):
    _vector = np.zeros((1, emb_size))
    length = 0
    for word in tokens:
        try:
            _vector += embeddingsL2.wv[word].reshape((1, emb_size)) * tfidf_mapL2[word]
            length += 1
        except KeyError:
            continue
        break

    if length > 0:
        _vector /= length

    return _vector

def encode_sentence1L2(tokens, emb_size):
    _vector = np.zeros((1, emb_size))
    length = 0
    for word in tokens:
        try:
            _vector += embeddings1L2.wv[word].reshape((1, emb_size)) * tfidf_map1L2[word]
            length += 1
        except KeyError:
            continue
        break

    if length > 0:
        _vector /= length

    return _vector

def encode_sentenceL3(tokens, emb_size):
    _vector = np.zeros((1, emb_size))
    length = 0
    for word in tokens:
        try:
            _vector += embeddingsL3.wv[word].reshape((1, emb_size)) * tfidf_mapL3[word]
            length += 1
        except KeyError:
            continue
        break

    if length > 0:
        _vector /= length

    return _vector

def encode_sentence1L3(tokens, emb_size):
    _vector = np.zeros((1, emb_size))
    length = 0
    for word in tokens:
        try:
            _vector += embeddings1L3.wv[word].reshape((1, emb_size)) * tfidf_map1L3[word]
            length += 1
        except KeyError:
            continue
        break

    if length > 0:
        _vector /= length

    return _vector




model = load_model("L123MultiHybridMultioutput20.h5")

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





rowsx_t_2 = []


with open("testHypernyms.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)




    # print(rows1)

        rowsx_t_2.append(rows1)

rowsx_t_3 = []

with open("testHyponyms.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):
                rows1.append(j)

        # print(rows1)

        rowsx_t_3.append(rows1)


rowsx_t_2L2 = []


with open("testHypernymsL2.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)




    # print(rows1)

        rowsx_t_2L2.append(rows1)

rowsx_t_3L2 = []

with open("testHyponymsL2.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):
                rows1.append(j)

        # print(rows1)

        rowsx_t_3L2.append(rows1)

rowsx_t_2L3 = []


with open("testHypernymsL3.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)




    # print(rows1)

        rowsx_t_2L3.append(rows1)

rowsx_t_3L3 = []

with open("testHyponymsL3.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):
                rows1.append(j)

        # print(rows1)

        rowsx_t_3L3.append(rows1)


encoded_train_set = t.texts_to_sequences(rowsx_t_1)
SEQ_LEN = 80


padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

train_docs = [list(doc) for doc in padded_train]
x_train = np.array([np.array(token) for token in train_docs])
x_train1 = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, rowsx_t_2)]))
x_train2 = scale(np.concatenate([encode_sentence1(ele, 200) for ele in map(lambda x: x, rowsx_t_3)]))
x_train3 = scale(np.concatenate([encode_sentenceL2(ele, 200) for ele in map(lambda x: x, rowsx_t_2L2)]))
x_train4 = scale(np.concatenate([encode_sentence1L2(ele, 200) for ele in map(lambda x: x, rowsx_t_3L2)]))
x_train5 = scale(np.concatenate([encode_sentenceL3(ele, 200) for ele in map(lambda x: x, rowsx_t_2L3)]))
x_train6 = scale(np.concatenate([encode_sentence1L3(ele, 200) for ele in map(lambda x: x, rowsx_t_3L3)]))


count = 0


# score = model.evaluate([x_train,x_train1], [y1,y2,y3,y4,y7,y8,y9,y10])
# print(score)
pred = []
# for i in range(0,8):
#     pred.append([])
predicted = model.predict([x_train,x_train1,x_train2,x_train3,x_train4,x_train5,x_train6])
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