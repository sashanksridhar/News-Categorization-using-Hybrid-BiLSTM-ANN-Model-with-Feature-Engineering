import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model
from keras.layers import Dense, Input, Concatenate
from keras.layers import LSTM,Embedding,Bidirectional
from keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import scale

rowsx = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []


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


t = Tokenizer()

t.fit_on_texts(rowsx)
vocab_size = len(t.word_index) + 1
print(vocab_size)
encoded_train_set = t.texts_to_sequences(rowsx)
SEQ_LEN = 80


padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

train_docs = [list(doc) for doc in padded_train]

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



x_train1 = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, rowsx1)]))
# x_train = np.array([encode_sentence_lstm(ele, 200) for ele in map(lambda x: x, rowsx)])
print(x_train1.shape)

x_train2 = scale(np.concatenate([encode_sentence1(ele, 200) for ele in map(lambda x: x, rowsx2)]))
# x_train = np.array([encode_sentence_lstm(ele, 200) for ele in map(lambda x: x, rowsx)])
print(x_train2.shape)

x_train3 = scale(np.concatenate([encode_sentenceL2(ele, 200) for ele in map(lambda x: x, rowsx1L2)]))
# x_train = np.array([encode_sentence_lstm(ele, 200) for ele in map(lambda x: x, rowsx)])
print(x_train3.shape)

x_train4 = scale(np.concatenate([encode_sentence1L2(ele, 200) for ele in map(lambda x: x, rowsx2L2)]))
# x_train = np.array([encode_sentence_lstm(ele, 200) for ele in map(lambda x: x, rowsx)])
print(x_train4.shape)

input_tensor = Input(shape=(SEQ_LEN,), dtype='int32')
e = Embedding(vocab_size, 300, input_length=SEQ_LEN, trainable=True)(input_tensor)
x = Bidirectional(LSTM(128, return_sequences=True))(e)
x = Bidirectional(LSTM(64, return_sequences=False))(x)
x = Dense(64, activation='relu')(x)
output_tensor = Dense(6, activation='relu')(x)
model = Model(input_tensor, output_tensor)


# model.compile(optimizer=Adam(lr=1e-3),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# x_train = np.array([np.array(token) for token in train_docs])
# model.fit(x_train, yx, epochs=50,batch_size=512)
# model.save("NewBD.h5")









visible = Input(shape=(200,))

c1 = Dense(256,activation='relu')(visible)
c2 = Dense(256,activation='relu')(c1)
c3 = Dense(512,activation='relu')(c2)
c4 = Dense(1024,activation='relu')(c3)
c5 = Dense(2048,activation='relu')(c4)
c6 = Dense(1024,activation='relu')(c5)
c7 = Dense(512,activation='relu')(c6)
c8 = Dense(256,activation='relu')(c7)
s1 = Dense(6,activation='relu')(c8)
model1 = Model(inputs=visible,outputs=s1)

visible1 = Input(shape=(200,))

c12 = Dense(256,activation='relu')(visible1)
c22 = Dense(256,activation='relu')(c12)
c32 = Dense(512,activation='relu')(c22)
c42 = Dense(1024,activation='relu')(c32)
c52 = Dense(2048,activation='relu')(c42)
c62 = Dense(1024,activation='relu')(c52)
c72 = Dense(512,activation='relu')(c62)
c82 = Dense(256,activation='relu')(c72)
s12 = Dense(6,activation='relu')(c82)
model2 = Model(inputs=visible1,outputs=s12)

visible2 = Input(shape=(200,))

c122 = Dense(256,activation='relu')(visible2)
c222 = Dense(256,activation='relu')(c122)
c322 = Dense(512,activation='relu')(c222)
c422 = Dense(1024,activation='relu')(c322)
c522 = Dense(2048,activation='relu')(c422)
c622 = Dense(1024,activation='relu')(c522)
c722 = Dense(512,activation='relu')(c622)
c822 = Dense(256,activation='relu')(c722)
s122 = Dense(6,activation='relu')(c822)
model3 = Model(inputs=visible2,outputs=s122)

visible3 = Input(shape=(200,))

c1222 = Dense(256,activation='relu')(visible3)
c2222 = Dense(256,activation='relu')(c1222)
c3222 = Dense(512,activation='relu')(c2222)
c4222 = Dense(1024,activation='relu')(c3222)
c5222 = Dense(2048,activation='relu')(c4222)
c6222 = Dense(1024,activation='relu')(c5222)
c7222 = Dense(512,activation='relu')(c6222)
c8222 = Dense(256,activation='relu')(c7222)
s1222 = Dense(6,activation='relu')(c8222)
model4 = Model(inputs=visible3,outputs=s1222)


combined = Concatenate()([model.output, model1.output,model2.output,model3.output,model4.output])
mix1 = Dense(100,activation='relu')(combined)
mix2 = Dense(50,activation='relu')(mix1)
out1 = Dense(1,activation="sigmoid")(mix2)
out2 = Dense(1,activation="sigmoid")(mix2)
out3 = Dense(1,activation="sigmoid")(mix2)
out4 = Dense(1,activation="sigmoid")(mix2)
out5 = Dense(1,activation="sigmoid")(mix2)
out6 = Dense(1,activation="sigmoid")(mix2)


model3 = Model(inputs=[input_tensor,visible,visible1,visible2,visible3],outputs=[out1,out2,out3,out4,out5,out6])

model3.compile(optimizer=Adam(lr=1e-3),
               loss='binary_crossentropy',
              metrics=['accuracy'])
x_train = np.array([np.array(token) for token in train_docs])
model3.fit([x_train,x_train1,x_train2,x_train3,x_train4], [y1,y2,y3,y4,y5,y6], epochs=10, batch_size=128)
model3.save("L12MultiHybridMultioutput10.h5")