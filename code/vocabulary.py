import numpy as np
import pickle

class Vocab(object):
    def __init__(self, texts, maxlen):
        self.word_to_idx_dict = {}
        self.idx_to_word_dict = {}
        self.oov = set()
        self.generate_vocab(texts)
        self.maxlen = maxlen

    # 输入为sentence list
    def generate_vocab(self, texts):
        words_set = set()
        for sentence in texts:
            for word in sentence.split():
                words_set.add(word)
        index = 1
        for word in words_set:
            self.idx_to_word_dict[index] = word
            self.word_to_idx_dict[word] = index
            index += 1
        self.word_to_idx_dict[''] = 0
        self.idx_to_word_dict[0] = ''

    def word_to_idx(self, word):
        if word in self.word_to_idx_dict:
            return self.word_to_idx_dict[word]
        else:
            self.oov.add(word)
            return 0

    def sentence_to_idxs(self, sentence):
        idxs = []
        for word in sentence.split():
            idxs.append(self.word_to_idx(word))
        return idxs

    def sentence_list_to_np(self, sentences):
        np_list = []
        mlen = 0
        for sentence in sentences:
            if len(sentence) > mlen:
                mlen = len(sentence)
            np_list.append(self.sentence_to_idxs(sentence))
        if self.maxlen != 0:
            mlen = self.maxlen
        res = np.zeros((len(sentences), mlen))
        for i in range(len(sentences)):
            for j in range(len(np_list[i])):
                if j >= mlen:
                    break
                res[i, j] = np_list[i][j]
        return res