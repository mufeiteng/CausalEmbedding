# -*- coding: utf-8 -*-
import codecs
from tools import project_source_path
from extraction.zh.cueset import global_cn_comma_words
import os

# global_cn_comma_words = ['，', '；']
# used for extracting negative sample
path = os.path.join(project_source_path, 'cue_words/')
single_fin = codecs.open(os.path.join(path, 'neg_single.txt'), 'r', 'utf-8')
parallel_fin = codecs.open(os.path.join(path, 'neg_parallel.txt'), 'r', 'utf-8')
parallel_words = [line.strip() for line in parallel_fin.readlines()]
temp = [line.split('...') for line in parallel_words]
all_neg_words = set([w for t in temp for w in t])
parallel_words = set(parallel_words)
single_words = set([line.strip() for line in single_fin.readlines()])
single_fin.close()
parallel_fin.close()


# rules of extracting negative samples
class NegativeRules(object):

    def __init__(self):
        return

    @staticmethod
    def extract_use_two_conj(words, postags, cue1, cue2):
        try:
            L = len(words)
            comma = [i for i in range(cue1, cue2) if words[i] in global_cn_comma_words]

            if len(comma) >= 3:
                return {}
            if 3 > len(comma) > 0:
                pre = comma[-1]
            else:
                pre = cue2

            left, left_pos = words[cue1 + 1:pre], postags[cue1 + 1:pre]

            comma = [i for i in range(cue2, L) if words[i] in global_cn_comma_words]
            if len(comma) < 2:
                end = L
            else:
                end = comma[1]

            right, right_pos = words[cue2 + 1:end], postags[cue2 + 1:end]

            if 10 < len(left) < 25 and 10 < len(right) < 25:
                return {'left': left, 'right': right, 'left_pos': left_pos,
                        'right_pos': right_pos, 'cue': '_'.join([words[cue1], words[cue2]])}
        except Exception:
            pass
        return {}

    @staticmethod
    def extract_use_single_conj(words, postags, cue):
        try:
            L = len(words)
            if len([i for i in range(0, L) if words[i] in global_cn_comma_words]) > 3:
                return {}
            start, end = -1, L
            if words[cue - 1] in global_cn_comma_words:
                i = cue - 2
            else:
                i = cue - 1
            while i >= 0:
                if words[i] in global_cn_comma_words:
                    start = i
                    break
                i -= 1
            i = cue + 1
            while i < L:
                if words[i] in global_cn_comma_words:
                    end = i
                    break
                i += 1

            left, left_pos = words[start + 1:cue], postags[start + 1:cue]
            right, right_pos = words[cue + 1:end], postags[cue + 1:end]
            if 6 < len(left) < 25 and 6 < len(right) < 25:
                return {'left': left, 'right': right, 'left_pos': left_pos,
                        'right_pos': right_pos, 'cue': words[cue]}
        except Exception:
            print('error occurred in \'boost use single conj\'')
        return {}

    @staticmethod
    def extract_use_verb(ptree, words, postags, cue):
        try:
            sbv, vob = [], []
            for i in ptree.tree[cue].children:
                if ptree.tree[i].relation == 'SBV':
                    sbv = ptree.sub_tree(i)
                if ptree.tree[i].relation == 'VOB':
                    vob = ptree.sub_tree(i)
            if not sbv or not vob:
                return {}
            left, left_pos = words[sbv[0]:sbv[-1] + 1], postags[sbv[0]:sbv[-1] + 1]
            right, right_pos = words[vob[0]:vob[-1] + 1], postags[vob[0]:vob[-1] + 1]
            if 5 < len(left) < 31 and 5 < len(right) < 31:
                return {'left': left, 'right': right, 'left_pos': left_pos,
                        'right_pos': right_pos, 'cue': words[cue]}
        except Exception:
            print('error occurred in \'boost use verb\'')
        return {}

    @staticmethod
    def extract(ptree):
        words, postags = [node.word for node in ptree.tree], [node.postag for node in ptree.tree]
        negative_pairs, L = [], len(words)
        # use two conj
        i = 0
        while i < L:
            if words[i] in all_neg_words:
                j = i + 1
                while j < L:
                    if words[j] in all_neg_words:
                        if '...'.join([words[i], words[j]]) in parallel_words:
                            res = NegativeRules.extract_use_two_conj(words, postags, i, j)
                            if res:
                                negative_pairs.append(res)
                    j += 1
            i += 1

        if not negative_pairs:
            i = 0
            while i < L:
                if words[i] in single_words and words[i-1] in global_cn_comma_words:
                    res = NegativeRules.extract_use_single_conj(words, postags, i)
                    if res:
                        negative_pairs.append(res)
                i += 1
        # # use verb
        # i = 0
        # while i < L:
        #     if postags[i] == 'v' and len(words[i]) > 1 and words[i] not in how_net:
        #         res = NegativeRules.extract_use_verb(ptree, words, postags, i)
        #         if res:
        #             negative_pairs.append(res)
        #     i += 1
        return negative_pairs
