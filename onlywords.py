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




t = Tokenizer()

t.fit_on_texts(rowsx)
vocab_size = len(t.word_index) + 1
print(vocab_size)
encoded_train_set = t.texts_to_sequences(rowsx)
SEQ_LEN = 80


padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

train_docs = [list(doc) for doc in padded_train]




input_tensor = Input(shape=(SEQ_LEN,), dtype='int32')
e = Embedding(vocab_size, 300, input_length=SEQ_LEN, trainable=True)(input_tensor)
x = Bidirectional(LSTM(128, return_sequences=True))(e)
x = Bidirectional(LSTM(64, return_sequences=False))(x)
x = Dense(64, activation='relu')(x)
out1 = Dense(1,activation="sigmoid")(x)
out2 = Dense(1,activation="sigmoid")(x)
out3 = Dense(1,activation="sigmoid")(x)
out4 = Dense(1,activation="sigmoid")(x)
out5 = Dense(1,activation="sigmoid")(x)
out6 = Dense(1,activation="sigmoid")(x)
model = Model(input_tensor, [out1,out2,out3,out4,out5,out6])


# model.compile(optimizer=Adam(lr=1e-3),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# x_train = np.array([np.array(token) for token in train_docs])
# model.fit(x_train, yx, epochs=50,batch_size=512)
# model.save("NewBD.h5")




model.compile(optimizer=Adam(lr=1e-3),
               loss='binary_crossentropy',
              metrics=['accuracy'])
x_train = np.array([np.array(token) for token in train_docs])
model.fit([x_train], [y1,y2,y3,y4,y5,y6], epochs=10, batch_size=128)
model.save("OnlyWords10.h5")