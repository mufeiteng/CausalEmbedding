# -*- coding: utf-8 -*-
import codecs
import numpy as np
from tools import WordCounter


def load_train_samples(data_path):
    """
    eg: earthquake----many people died
    """
    _response = []
    with codecs.open(data_path, 'r', 'utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            items = line.strip().split('----')
            
            s1 = items[0].split(' ')
            s2 = items[1].split(' ')
            if '' in s1 or '' in s2:
                continue
            _response.append((s1, s2, 1.0))
    return _response


def convert2indices(data, left_vocab, right_vocab):
    _response, max_len = [], 0
    for (left, right, label) in data:
        cause_indices = [left_vocab[w] for w in left if w in left_vocab]
        effect_indices = [right_vocab[w] for w in right if w in right_vocab]
        if cause_indices and effect_indices:
            max_len = max([max_len, len(cause_indices), len(effect_indices)])
            _response.append((cause_indices, effect_indices, label))
    return _response, max_len


def complete_data(batch, left_vocab, right_vocab, max_len, pad):
    _left_w, _right_w, _len1, _len2, _label = [], [], [], [], []

    for (left, right, label) in batch:
        l1, l2 = len(left), len(right)
        _len1.append(l1)
        _len2.append(l2)
        _label.append(label)
        _left_w.append(left + [left_vocab[pad]] * (max_len - l1))
        _right_w.append(right + [right_vocab[pad]] * (max_len - l2))
    return _left_w, _right_w, _len1, _len2, _label


def sample_negative(data, amount):
    L, _response = len(data), []
    for i in range(amount):
        k = np.random.randint(0, L)
        _left = data[k][0]
        j = np.random.randint(0, L)
        _right = data[j][1]
        _response.append((_left, _right, 0.0))
    return _response


class Data(object):
    def __init__(self):
        self.train, self.test, self.max_len = None, None, 0
        self.vocab_left, self.vocab_rev_left, self.vocab_left_size = ['<pad>'], {}, 0
        self.vocab_right, self.vocab_rev_right, self.vocab_right_size = ['<pad>'], {}, 0
        self.c2e_visualize, self.e2c_visualize = [], []

    def load_test_data(self, test_data_path):
        cause_dev = ['accident', 'explosion', 'earthquake', 'virus', 'storm', 'war']
        effect_dev = ['pollution', 'death', 'loss', 'failure', 'disease', 'illness', 'flood']
        
        for w in cause_dev:
            try:
                self.c2e_visualize.append(self.vocab_rev_left[w])
            except KeyError as e:
                print('{} is not existed in cause vocab!'.format(e))
        for w in effect_dev:
            try:
                self.e2c_visualize.append(self.vocab_rev_right[w])
            except KeyError as e:
                print('{} is not existed in effect vocab!'.format(e))
        """
        eg:
        earthquake died 1
        flood wood 0
        """
        _sharp_word_pair = []
        with codecs.open(test_data_path, 'r', 'utf-8') as f:
            lines = f.readlines()
            for line in lines:
                res = line.strip().split(' ')
                w1, w2, label = res
                _sharp_word_pair.append((w1, w2, float(label)))
            print('total number of wp test set is {}.'.format(len(_sharp_word_pair)))
        return _sharp_word_pair
    
    def build_vocab(self, data, min_count):
        counter_left, counter_right = WordCounter(), WordCounter()
        for (left, right, _) in data:
            counter_left.add(left)
            counter_right.add(right)
        self.vocab_left += [w for w in counter_left if counter_left[w] >= min_count]
        self.vocab_right += [w for w in counter_right if counter_right[w] >= min_count]
        self.vocab_rev_left = {x: i for i, x in enumerate(self.vocab_left)}
        self.vocab_rev_right = {x: i for i, x in enumerate(self.vocab_right)}
        self.vocab_left_size = len(self.vocab_left)
        self.vocab_right_size = len(self.vocab_right)

    def prepare_data(self, train_data_path, test_data_path, min_count):
        samples = load_train_samples(train_data_path)
        self.build_vocab(samples, min_count)
        self.train, self.max_len = convert2indices(samples, self.vocab_rev_left, self.vocab_rev_right)
        self.test = self.load_test_data(test_data_path)
