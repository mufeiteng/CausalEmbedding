# -*- coding: utf-8 -*-
from extraction.cn.cueset import *
import codecs
import re
from tools import project_source_path
import os

# rules of extracting candidate pairs
path = os.path.join(project_source_path, 'cue_words/')
lines = codecs.open(os.path.join(path, 'all_conj_left.txt'), 'r', 'utf-8').readlines()
all_conj_left = set([line.strip() for line in lines])
lines = codecs.open(os.path.join(path, 'all_conj_right.txt'), 'r', 'utf-8').readlines()
all_conj_right = set([line.strip() for line in lines])
all_conjunction = all_conj_left | all_conj_right


class CandidateRules(object):
    # 要高考了，我心乱如麻
    # 因身材较矮，显得帽翅格外长
    # 整个风景区庄严肃穆，环境优美；
    # 神奇至极，令人不由慨叹大自然的鬼斧神工
    # 牛郎无望，只能停止追赶
    # 天山英雄传的动作很新颖，不单调、枯燥
    # 解脱之法用尽，终郁郁不得，故长叹与君知
    # 君令人之侧目，在于文笔生花，话语如珠
    # 山茶油能改变食用单一油类所造成的营养不均，能较充分地平衡人体营养，有利身体健康
    def __int__(self):
        return

    @staticmethod
    def filter(ptree, start, end):
        tag = False
        words, postags = [node.word for node in ptree.tree], [node.postag for node in ptree.tree]
        for i in range(start, end):
            if postags[i] == 'v':
                for j in ptree.tree[i].children:
                    if postags[j] in ['a', 'v', 'n', 'd', 'i']:
                        tag = True
            if postags[i] in ['n', 'i']:
                for j in ptree.tree[i].children:
                    if postags[j] in ['a', 'n', 'i', 'r']:
                        tag = True
            if postags[i] == 'a':
                for j in ptree.tree[i].children:
                    if postags[j] in ['a', 'n', 'i', 'd']:
                        tag = True
            if end-start < 2:
                if postags[i] in ['a', 'n', 'i', 'v']:
                    tag = True
        return tag

    @staticmethod
    def extract_use_two_conj(ptree, words, postags, cue1, cue2):
        if words[cue2-1] != '，':
            return {}
        if len(words[cue2]) == 2 and len(words[cue1]) == 1:
            return {}
        if postags[cue1] == 'v' and len(words[cue1]) == 1:
            return {}
        if postags[cue2] == 'v' and len(words[cue2]) == 1:
            return {}
        if postags[cue1] not in ['d', 'c'] or postags[cue2] not in ['d', 'c']:
            return {}
        L = len(words)
        comma = [i for i in range(cue1, cue2) if words[i] == '，']
        left, left_pos, right, right_pos = [], [], [], []

        if len(comma) >= 3:
            return {}
        if 3 > len(comma) > 0:
            pre = comma[-1]
        else:
            pre = cue2
        tag_left = CandidateRules.filter(ptree, cue1+1, pre)
        if tag_left:
            left, left_pos = words[cue1+1:pre], postags[cue1+1:pre]

        comma = [i for i in range(cue2, L) if words[i] == '，']
        if len(comma) < 2:
            end = L
        else:
            end = comma[1]
        tag_right = CandidateRules.filter(ptree, cue2+1, end)
        if tag_right:
            right, right_pos = words[cue2+1:end], postags[cue2+1:end]

        if 1 < len(left) < 21 and 1 < len(right) < 21:
            return {'left': left, 'right': right, 'left_pos': left_pos,
                    'right_pos': right_pos, 'cue': '_'.join([words[cue1], words[cue2]])}
        return {}

    @staticmethod
    def extract_use_single_conj(ptree, words, postags, cue):
        L = len(words)
        if len([i for i in range(0, L) if words[i] == '，']) > 4:
            return {}
        start, end = -1, L
        if words[cue-1] == '，':
            i = cue-2
        else:
            i = cue-1
        while i >= 0:
            if words[i] == '，':
                start = i
                break
            i -= 1
        i = cue+1
        while i < L:
            if words[i] == '，':
                end = i
                break
            i += 1
        tag_left = CandidateRules.filter(ptree, start+1, cue)
        tag_right = CandidateRules.filter(ptree, cue+1, end)
        if tag_left and tag_right:
            left, left_pos = words[start+1:cue], postags[start+1:cue]
            right, right_pos = words[cue+1:end], postags[cue+1:end]
            if 1 < len(left) < 21 and 1 < len(right) < 21:
                return {'left': left, 'right': right, 'left_pos': left_pos,
                        'right_pos': right_pos, 'cue': words[cue]}
        return {}

    @staticmethod
    def extract_use_verb(ptree, words, postags, cue):
        L = len(words)
        left, left_pos, right, right_pos = [], [], [], []
        sbv, vob = [], []
        for i in ptree.tree[cue].children:
            if ptree.tree[i].relation == 'SBV':
                sbv.extend(ptree.sub_tree(i))
        for i in ptree.tree[cue].children:
            if ptree.tree[i].relation == 'VOB':
                vob.extend(ptree.sub_tree(i))
        if not sbv and not vob:
            return {}
        if not sbv and vob:
            if ptree.tree[cue].relation == 'COO':
                node = ptree.tree[ptree.tree[cue].parent]
                for i in node.children:
                    if ptree.tree[i].relation == 'SBV':
                        sbv.extend(ptree.sub_tree(i))
                if sbv:
                    tag_left = CandidateRules.filter(ptree, sbv[0], sbv[-1] + 1)
                    tag_right = CandidateRules.filter(ptree, vob[0], vob[-1]+1)
                    if tag_left and tag_right:
                        left, left_pos = words[sbv[0]:sbv[-1]+1], postags[sbv[0]:sbv[-1]+1]
                        right, right_pos = words[vob[0]:vob[-1] + 1], postags[vob[0]:vob[-1] + 1]
                else:
                    pre = -1
                    if words[cue-1] == '，':
                        j = cue-2
                    else:
                        j = cue-1
                    while j >= 0:
                        if words[j] == '，':
                            pre = j
                            break
                        j -= 1
                    tag_left = CandidateRules.filter(ptree, pre+1, cue)
                    tag_right = CandidateRules.filter(ptree, vob[0], vob[-1]+1)
                    if tag_left and tag_right:
                        left, left_pos = words[pre+1:cue], postags[pre+1:cue]
                        right, right_pos = words[vob[0]:vob[-1]+1], postags[vob[0]:vob[-1]+1]

        # if sbv and not vob:
        #     end, j = L, cue+1
        #     while j < L:
        #         if words[j] == '，':
        #             end = j
        #             break
        #         j += 1
        #     tag_right = CandidateRules.filter(ptree, cue+1, end)
        #     tag_left = CandidateRules.filter(ptree, sbv[0], sbv[-1]+1)
        #     if tag_left and tag_right:
        #         left, left_pos = words[sbv[0]:sbv[-1]+1], postags[sbv[0]:sbv[-1]+1]
        #         right, right_pos = words[cue+1:end], postags[cue+1:end]

        if sbv and vob:
            tag_left = CandidateRules.filter(ptree, sbv[0], sbv[-1]+1)
            tag_right = CandidateRules.filter(ptree, vob[0], vob[-1]+1)
            if tag_left and tag_right:
                left, left_pos = words[sbv[0]:sbv[-1]+1], postags[sbv[0]:sbv[-1]+1]
                right, right_pos = words[vob[0]:vob[-1]+1], postags[vob[0]:vob[-1]+1]
        if 1 < len(left) < 31 and 1 < len(right) < 31:
            return {'left': left, 'right': right, 'left_pos': left_pos,
                    'right_pos': right_pos, 'cue': words[cue]}
        return {}

    @staticmethod
    def extract_only_one_comma(ptree, words, postags, comma):
        tag_left = CandidateRules.filter(ptree, 0, comma)
        tag_right = CandidateRules.filter(ptree, comma+1, len(words))
        if tag_left and tag_right:
            left, left_pos = words[:comma], postags[:comma]
            right, right_pos = words[comma + 1:len(words)], postags[comma + 1:len(words)]
            if 3 < len(left) < 31 and 3 < len(right) < 31:
                return {'left': left, 'right': right, 'left_pos': left_pos,
                        'right_pos': right_pos, 'cue': 'comma'}
        return {}

    @staticmethod
    def extract(ptree):
        """
        先两个连词
        再一个连词
        再根据comma
        再动词
        :param ptree: parser tree of a sentence
        :return: a list of many candidate pairs
        """
        words, postags = [node.word for node in ptree.tree], [node.postag for node in ptree.tree]
        candidate_pairs, L = [], len(words)
        # use two conj
        i = 0
        while i < L:
            if words[i] in all_conj_left:
                j = i + 1
                while j < L:
                    if words[j] in all_conj_right:
                        res = CandidateRules.extract_use_two_conj(ptree, words, postags, i, j)
                        if res:
                            candidate_pairs.append(res)
                    j += 1
            i += 1
        # use single conj
        if not candidate_pairs:
            i = 0
            while i < L:
                if words[i] in all_conjunction:
                    res = CandidateRules.extract_use_single_conj(ptree, words, postags, i)
                    if res:
                        candidate_pairs.append(res)
                i += 1

        # only one comma in sentence
        comma = [i for i, w in enumerate(words) if w == '，']
        if len(comma) == 1:
            res = CandidateRules.extract_only_one_comma(ptree, words, postags, comma[0])
            if res:
                candidate_pairs.append(res)

        # use verb
        i = 0
        while i < L:
            if postags[i] == 'v':
                res = CandidateRules.extract_use_verb(ptree, words, postags, i)
                if res:
                    candidate_pairs.append(res)
            i += 1

        return candidate_pairs
