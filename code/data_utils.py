import re
import numpy as np
import nltk
nltk.download('punkt')
import pandas as pd
import numpy as np

class2label = {'Other': 0,
               'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
               'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
               'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
               'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
               'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
               'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
               'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
               'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
               'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}

label2class = {0: 'Other',
               1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
               3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
               5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
               7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
               9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
               11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
               13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
               15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
               17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}


def load_glove(embedding_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.word_to_idx_dict), embedding_dim).astype(np.float32) / np.sqrt(len(vocab.word_to_idx_dict))
    # load any vectors from the word2vec
    print("Load glove file {0}".format(embedding_path))
    f = open(embedding_path, 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.word_to_idx(word)
        if idx == 22389:
            print(word, vocab.word_to_idx(word), len(vocab.word_to_idx_dict))
        if idx != 0:
            initW[idx] = embedding
    return initW

def clean_str(text):
    text = text.lower()
    # Clean the text
    # text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    # text = re.sub(r"what's", "what is ", text)
    # text = re.sub(r"that's", "that is ", text)
    # text = re.sub(r"there's", "there is ", text)
    # text = re.sub(r"it's", "it is ", text)
    # text = re.sub(r"\'s", " ", text)
    # text = re.sub(r"\'ve", " have ", text)
    # text = re.sub(r"can't", "can not ", text)
    # text = re.sub(r"n't", " not ", text)
    # text = re.sub(r"i'm", "i am ", text)
    # text = re.sub(r"\'re", " are ", text)
    # text = re.sub(r"\'d", " would ", text)
    # text = re.sub(r"\'ll", " will ", text)
    # text = re.sub(r",", " ", text)
    # text = re.sub(r"\.", " ", text)
    # text = re.sub(r"!", " ! ", text)
    # text = re.sub(r"\/", " ", text)
    # text = re.sub(r"\^", " ^ ", text)
    # text = re.sub(r"\+", " + ", text)
    # text = re.sub(r"\-", " - ", text)
    # text = re.sub(r"\=", " = ", text)
    # text = re.sub(r"'", " ", text)
    # text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    # text = re.sub(r":", " : ", text)
    # text = re.sub(r" e g ", " eg ", text)
    # text = re.sub(r" b g ", " bg ", text)
    # text = re.sub(r" u s ", " american ", text)
    # text = re.sub(r"\0s", "0", text)
    # text = re.sub(r" 9 11 ", "911", text)
    # text = re.sub(r"e - mail", "email", text)
    # text = re.sub(r"j k", "jk", text)
    # text = re.sub(r"\s{2,}", " ", text)

    return text.strip()

# 解析数据集文件
def load_data_and_labels(path):
    data = []
    nums = []
    lines = [line.strip() for line in open(path)]
    max_sentence_length = 0
    for idx in range(0, len(lines), 4):
        id = lines[idx].split("\t")[0]
        nums.append(id)
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = sentence.replace('<e1>', ' # ')
        sentence = sentence.replace('</e1>', ' # ')
        sentence = sentence.replace('<e2>', ' $ ')
        sentence = sentence.replace('</e2>', ' $ ')

        sentence = clean_str(sentence)
        tokens = nltk.word_tokenize(sentence)
        # distrib[int(len(tokens) / 10)] += 1
        if max_sentence_length < len(tokens):
            max_sentence_length = len(tokens)
        sentence = " ".join(tokens)

        data.append([id, sentence, relation])
    # print(distrib)
    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation"])
    df['label'] = [class2label[r] for r in df['relation']]

    # Text Data
    x_text = df['sentence'].tolist()

    # Label Data
    y = df['label']
    labels = y.values.ravel()
    labels = labels.astype(np.uint8)

    return x_text, labels, nums


if __name__ == "__main__":
    trainFile = '../data/TRAIN_FILE.TXT'
    testFile = '../data/TEST_FILE_FULL.TXT'
    # generate_vocab(trainFile, testFile, '../data/vocab.txt')
    x, labels, nums = load_data_and_labels(testFile)
    print(x)
