# -*- coding: utf-8 -*-
from pyltp import Segmentor, Postagger, Parser, SentenceSplitter
from msworks.tools import ltp_path
from msworks.extraction.cn.cueset import *
import logging
import re


class Node:
    def __init__(self, word, index, postag, head, relation):
        self.index, self.word, self.postag = index, word, postag
        self.parent, self.relation, self.children = head, relation, []

    def __str__(self):
        return 'id: {}, word: {}, postag: {}, parent: {}, relation: {}, children: {}'.format(
            self.index, self.word, self.postag, self.parent, self.relation, ','.join(map(str, self.children))
        )


class LTPModel:
    def __init__(self):
        self.sentencesplitter = SentenceSplitter()
        self.segmentor = Segmentor()
        self.segmentor.load(ltp_path + 'cws.model')
        self.postagger = Postagger()
        self.postagger.load(ltp_path + 'pos.model')
        self.parser = Parser()
        self.parser.load(ltp_path + 'parser.model')
        # self.recognizer = NamedEntityRecognizer()
        # self.recognizer.load(ltp_path + "ner.model")


all_causal_clues = cn_cause_cue_set | cn_effect_cue_set | cn_reverse_cause_cue_set | cn_reverse_effect_cue_set
all_causal_clues = all_causal_clues | cn_single_cause_cue_set | cn_single_effect_cue_set | cn_causal_verb_set


class CnParseTree:
    def __init__(self, ltp_model):
        self._splitter_ = ltp_model.sentencesplitter
        self._segmentor_ = ltp_model.segmentor
        self._postagger_ = ltp_model.postagger
        self._parser_ = ltp_model.parser
        self.tree, self.root, self.sentence, self.bow, self.isCausal = [], -1, None, set(), False

    def create(self, sentence):
        del self.tree[:]
        self.sentence, self.isCausal = None, True
        if not (0 < len(sentence) < 200):
            raise Exception('sent length is too long!')
        self.sentence, self.tree, self.root = sentence, [], -1
        words = self._segmentor_.segment(sentence)
        if len(words) > 60:
            raise Exception('words length out of memory!')
        self.bow = set(words)
        if len(self.bow & all_causal_clues) == 0:
            self.isCausal = False
        postags = self._postagger_.postag(words)
        arcs = self._parser_.parse(words, postags)
        L, i = len(words), 0
        while i < L:
            node = Node(words[i], i, postags[i], arcs[i].head - 1, arcs[i].relation)
            self.tree.append(node)
            i += 1
        for node in self.tree:
            if node.parent == -1:
                self.root = node.index
            self.tree[node.parent].children.append(node.index)

    def split_sentence(self, sent, efface_number=False):
        results, response = self._splitter_.split(sent), []
        for r in results:
            if efface_number and bool(re.search(r'\d', r)):
                continue
            r = re.sub('[^，。；:、（）《》“”\n\u4e00-\u9fa5]', '', r)
            if 7 < len(r) < 300:
                response.append(r)
        return response

    def subtree(self, index):
        queue = [index]
        res = []
        # queue.append(index)
        while queue:
            top = self.tree[queue[0]]
            queue.extend([child for child in top.children if child != self.root])
            res.append(queue.pop(0))
        res.sort()
        return res

