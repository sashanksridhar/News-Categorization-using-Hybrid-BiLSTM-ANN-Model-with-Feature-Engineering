from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from os import walk
import nltk
import csv
import pandas as pd
stop_words = set(stopwords.words('english'))

df = pd.read_csv("news_test.csv",encoding='latin1')

labels = df["class"].values.tolist()

news = df["content"].values.tolist()
count = 0

for i in range(0,len(labels)):
    print(count)
    count+=1
    label = labels[i]

    words = news[i].split()
    with open("testWordsL10.csv", 'a', encoding='latin1') as csv_file, open("testHypernymsL10.csv", 'a', encoding='latin1') as hypernym_file, open("testHyponymsL10.csv", 'a', encoding='latin1') as hyponym_file:
        filewriter = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        hyperwriter = csv.writer(hypernym_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        hypowriter = csv.writer(hyponym_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        filtered_sentence = []

        x = []
        for r in words:

            if not r in stop_words:
                # print(r)

                filtered_sentence.append(r)
        tagged = nltk.pos_tag([word for word in filtered_sentence if word])
        for i in tagged:
            if len(i[0]) != 0 or len(i[0]) != 1:

                if i[1] == 'IN' or i[1] == 'DT' or i[1] == 'CD' or i[1] == 'CC' or i[1] == 'EX' or i[1] == 'MD' or i[
                    1] == 'WDT' or i[1] == 'WP' or i[1] == 'UH' or i[1] == 'TO' or i[1] == 'RP' or i[1] == 'PDT' or i[
                    1] == 'PRP' or i[1] == 'PRP$' or i[0] == 'co':
                    # print(i[0])
                    continue
                else:

                    x.append(i[0].strip(".,?!"))
        listitem = [] #words
        hyper = []
        hypo = []
        listitem.append(label)

        for it in x:
            # print(i)
            hy = ""
            hp = ""
            token = it
            try:
                for k in range(10): #0-9
                    for i, j in enumerate(wn.synsets(token)):
                        for l in j.hypernyms():
                            token =  l.lemma_names()[0]
                            if k==9:
                                hyper.append(l.lemma_names()[0])
                                hy = l.lemma_names()[0]
                            break
                        break
            except IndexError:
                hyper.append([it])
                hy = it

            try:
                for k in range(10):
                    for i, j in enumerate(wn.synsets(token)):
                        for l in j.hyponyms():
                            token = l.lemma_names()[0]
                            if k==9:
                                hypo.append(l.lemma_names()[0])
                                hp = l.lemma_names()[0]
                            break
                        break
            except IndexError:
                hypo.append([it])
                hp = it

            if hy == it or hp == it:
                continue

            listitem.append(it)
        filewriter.writerow(listitem)
        hyperwriter.writerow(hyper)
        hypowriter.writerow(hypo)
