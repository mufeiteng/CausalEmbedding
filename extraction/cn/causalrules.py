# -*- coding: utf-8 -*-
from extraction.cn.cueset import *
import re


class Base(object):
    def __init__(self):
        pass

    @classmethod
    def clean_noise(cls, words, candidates):
        # efface punctuations and words which contains number
        res = [c for c in candidates if (words[c] not in global_cn_punctuation) and (not re.match(r'.*\d', words[c]))]
        return res

    @classmethod
    def judge_num_exist(cls, candidate):
        # （由于指数函数的取值范围为负无穷到正无穷，0是渐近线，因此永远不会到达原点0，无法从原点出发，上述有误）
        # 版、2版、动画版声优均相同，因为版是18的，所以部分声优在版中用了马甲，（）内为声优在版中使用的马甲
        # 如要推知33号元素的位置，因它在18和36之间，所以必在第4周期，由36号往左数，应在族
        for c in candidate:
            if re.match(r'.*\d', c):
                return True
        return False

    @classmethod
    def extract(cls, ptree, result):
        pass
    
    
# rules of extracting positive samples
# 有数字的都不要 括号里的都不要
class CCauseCEffect(Base):
    def __init__(self):
        Base.__init__(self)

    @classmethod
    def extract(cls, ptree, result):
        cue1, cue2 = result['cue1'], result['cue2']
        words, postag = [node.word for node in ptree.tree], [node.postag for node in ptree.tree]
        if words[cue2-1] not in global_cn_comma_words:
            return dict()
        # 因为，所以之间逗号数不能大于2
        if len([w for w in words[cue1:cue2-2] if w in global_cn_comma_words]) > 1:
            return dict()
        cause = [i for i in range(cue1+1, cue2-1)]
        if len(cause) > 20 or len(cause) < 2:
            # 4氢本来不是碱金属，但因为在族，所以归入此表
            # 岳飞生前是无资格穿蟒袍的，因后封鄂王，所以身着蟒袍
            return {}
        i, comma = len(words)-1, []
        while i > cue2+1:
            if words[i] in global_cn_comma_words:
                comma.append(i)
            i -= 1
        comma.reverse()
        if len(comma) == 0:
            effect = [i for i in range(cue2+1, len(words))]
        elif len(comma) == 1:
            effect = [i for i in range(cue2+1, comma[0])] if len(words)-cue2 > 25 else [i for i in range(cue2+1, len(words))]
        else:
            effect = [i for i in range(cue2 + 1, comma[0])] if comma[1]-cue2 > 25 else [i for i in range(cue2+1, comma[1])]
        if 2 < len(cause) < 26 and 2 < len(effect) < 26:
            pos_cause, pos_effect = [postag[c] for c in cause], [postag[e] for e in effect]
            cause, effect = [words[c] for c in cause], [words[e] for e in effect]
            return {'left': cause, 'right': effect, 'cue': '_'.join([words[cue1], words[cue2]]),
                    'left_pos': pos_cause, 'right_pos': pos_effect}
        return {}


class CEffectCCause(Base):
    def __init__(self):
        Base.__init__(self)

    @classmethod
    def is_negative_cue(cls, words, index):
        # '缘于', '因为', '由于', '在于', '基于'
        # 并非因为，不在于，不源于，并非是，并不是
        flag1 = words[index-1] in global_cn_neg_modifier
        flag2 = words[index-1] == '是' and words[index-2] in global_cn_neg_modifier
        return flag1 or flag2

    # 之所以..., 不是..., 而是...
    @classmethod
    def find_positive_cue(cls, words, index):
        """
        1.  在这些文章中，鲁迅多次提及杨荫榆，对她的所作所为给予冷嘲热讽，就如后来人们所知道的，杨荫榆之所以能够出名，
            不是因为她早年大胆的抗婚之举，也不是因为她是中国近现代历史上第一位女大学校长，而是因为女师大风潮更准确地说，
            是因为鲁迅对她在女师大的所作所为进行的讥讽嘲骂
        2.  之所以非要请威远镖局，表面上是因为他们的镖师武艺高强，实际上是他们与这条道上的响马多有交情，常以重金打点，
            才得允许他们护的部分镖顺利过境，自然他们索要的报酬是不低的
        3.  妓女是一种毫无尊严的职业，而且这种职业之所以能够长盛不衰，并不是因为有人喜欢做，
            而是因为有人喜欢捧场----有人 喜欢 捧场----能够 长盛不衰
        """
        i = len(words)-1
        while i >= index:
            if words[i] in cn_reverse_cause_cue_set and not cls.is_negative_cue(words, i):
                return i
            i -= 1
        return -1

    @classmethod
    def get_cause(cls, words, start):
        comma = [i for i in range(start, len(words)) if words[i] in global_cn_comma_words]
        if len(comma) >= 2:
            end = comma[0] if comma[1]-start > 25 else comma[1]
        elif len(comma) == 1:
            end = comma[0] if len(words)-start > 25 else len(words)
        else:
            end = len(words)
        return [i for i in range(start + 1, end)]

    @classmethod
    def extract(cls, ptree, result):
        # 教师格言∶河流之所以能到达目的地，是因为它知道如何避开障碍
        # 网友∶这对父母之所以抢了拍摄者的相机存储卡，是因为存储卡中有他们两岁幼女小便的影像
        cue1, cue2 = result['cue1'], result['cue2']
        words, postag = [node.word for node in ptree.tree], [node.postag for node in ptree.tree]
        # 之所以， 因为 之间逗号不能大于2
        comma = [i for i in range(cue1, cue2) if words[i] in global_cn_comma_words]
        if len(comma) > 2 or not comma:
            return {}
        effect = [i for i in range(cue1+1, comma[-1])]
        if cls.is_negative_cue(words, cue2):
            new_cue2 = cls.find_positive_cue(words, cue2)
            if new_cue2 == -1:
                return dict()
            cause = cls.get_cause(words, new_cue2)
        else:
            cause = cls.get_cause(words, cue2)
        if 1 < len(cause) < 26 and 1 < len(effect) < 26:
            pos_cause, pos_effect = [postag[c] for c in cause], [postag[e] for e in effect]
            cause, effect = [words[c] for c in cause], [words[e] for e in effect]
            return {'left': cause, 'right': effect, 'cue': '_'.join([words[cue1], words[cue2]]),
                    'left_pos': pos_cause, 'right_pos': pos_effect}
        return {}


class CauseCEffect(Base):
    def __init__(self):
        Base.__init__(self)

    @classmethod
    def extract(cls, ptree, result):
        cue, words, postag = result['cue'], [node.word for node in ptree.tree], [node.postag for node in ptree.tree]
        if words[cue-1] not in global_cn_comma_words:
            return {}
        comma = [i for i in range(0, cue-1) if words[i] in global_cn_comma_words]
        if not comma:
            cause = [i for i in range(cue-1)]
        elif len(comma) == 1:
            cause = [i for i in range(comma[0] + 1, cue - 1)] if cue > 25 else [i for i in range(cue-1)]
        else:
            return {}
        # 金汕说，“奥运会和任何比赛都不一样，它的影响力覆盖全世界，所以，中国观众在赛场上的表现，会直接影响其他国家人民对中国人的看法。”
        comma = [i for i in range(cue+2, len(words)) if words[i] in global_cn_comma_words]
        if not comma:
            effect = [i for i in range(cue+1, len(words))]
        elif len(comma) == 1:
            effect = [i for i in range(cue + 1, comma[0])] if len(words)-cue > 25 else [i for i in range(cue+1, len(words))]
        else:
            return {}
        if 1 < len(cause) < 26 and 1 < len(effect) < 26:
            pos_cause, pos_effect = [postag[c] for c in cause], [postag[e] for e in effect]
            cause, effect = [words[c] for c in cause], [words[e] for e in effect]
            return {'left': cause, 'right': effect, 'cue': words[cue],
                    'left_pos': pos_cause, 'right_pos': pos_effect}
        return {}


class CauseVEffect(Base):

    def __init__(self):
        Base.__init__(self)

    # 富人不仅享用了廉价汽油，而且造成的污染还得穷人一起埋单
    # 月日上午，征战韩国站超级赛的中国队回到北京，有关林丹在日进行的男单决赛中因质疑裁判判罚而引发的事件顿时成为记者争相了解的热点。
    # 去除指代情况 给/对/对于...带来的 由此引发的
    @classmethod
    def purity(cls, ptree, words, postags, cause, effect, cue):
        try:
            assert max(cause) < len(words), max(effect) < len(words)
            tag_cause = cls.filter(ptree, postags, cause[0], cause[-1] + 1)
            tag_effect = cls.filter(ptree, postags, effect[0], effect[-1] + 1)
            if not (tag_cause and tag_effect):
                return {}
            len_cause = 25 > len(cause) > 0
            len_effect = 25 > len(effect) > 0
            if len_cause and len_effect:
                cause_words, cause_pos = [words[c] for c in cause], [postags[c] for c in cause]
                effect_words, effect_pos = [words[e] for e in effect], [postags[e] for e in effect]
                return {'left': cause_words, 'right': effect_words,
                        'left_pos': cause_pos, 'right_pos': effect_pos, 'cue': words[cue]}
        except Exception:
            return {}

    @classmethod
    def filter(cls, ptree, postags, start, end):
        tag = False
        for i in range(start, end):
            if postags[i] == 'v':
                for j in ptree.tree[i].children:
                    if postags[j] in {'a', 'v', 'n', 'd', 'i'}:
                        tag = True
                        break
            if postags[i] in {'n', 'i'}:
                for j in ptree.tree[i].children:
                    if postags[j] in {'a', 'n', 'i', 'r'}:
                        tag = True
                        break
            if postags[i] == 'a':
                for j in ptree.tree[i].children:
                    if postags[j] in {'a', 'd'}:
                        tag = True
                        break
            # 少于两个词
            if end - start < 3:
                if postags[i] in {'n', 'v', 'i', 'a'}:
                    tag = True
        return tag

    @classmethod
    def extract(cls, ptree, result):
        # 导致(COO)的抽取
        try:
            cue, words, postags = result['cue'], [node.word for node in ptree.tree], [node.postag for node in ptree.tree]
            if words[cue] in {'使', '使得'}:
                if words[cue-1] in global_cn_comma_words and cue < len(words)-2:
                    start, j = 0, cue-2
                    while j >= 0:
                        if words[j] in global_cn_comma_words:
                            start = j + 1
                            break
                        j -= 1
                    # 之前的第一个逗号
                    cause = [i for i in range(start, cue - 1)]
                    for i in range(start, cue-1):
                        if words[i] in global_cn_conj_words:
                            cause = [j for j in range(i+1, cue-1)]
                            break
                    effect, tag_dbl, tag_vob, node = [], False, False, ptree.tree[cue]
                    for i in node.children:
                        if ptree.tree[i].relation == 'DBL':
                            effect.extend(ptree.subtree(i))
                            tag_dbl = True
                        if ptree.tree[i].relation == 'VOB':
                            effect.extend(ptree.subtree(i))
                            tag_vob = True
                    if not tag_dbl or not tag_vob:
                        return {}
                    return cls.purity(ptree, words, postags, cause, effect, cue)
                return {}
            if cue < len(words)-2 and words[cue+1] == '的':
                if words[cue] in {'使', '使得'}:
                    return {}
                cause, start = [], 0
                for i in ptree.tree[cue].children:
                    cause.extend(ptree.subtree(i))
                if len(cause) == 0:
                    return {}
                if words[cause[-1]] == '的':
                    cause = cause[:-1]
                if not cause:
                    return {}
                cause.sort()
                # 前面有 因，因为，由于
                j = cue-2 if words[cue - 1] in global_cn_comma_words else cue-1
                while j >= 0:
                    if words[j] in global_cn_comma_words:
                        start = j + 1
                        break
                    j -= 1
                for i in range(start, cue):
                    if words[i] in global_cn_conj_words:
                        cause = [j for j in range(i+1, cue)]
                        break
                if not cause or len(set(words[cause[0]:cue]) & global_cn_refers_words) > 0:
                    return {}
                # 去除 所，而 及修饰关键词的副词
                end = len(cause)-1
                if words[cause[end]] in {'所', '而', '可', '能', '会', '可能'}:
                    cause = cause[:end]
                if not cause or len(global_cn_noise_words & set(cause)) > 0:
                    return {}
                node = ptree.tree[cue]
                while node.relation == 'ATT' and node.parent != -1:
                    node = ptree.tree[node.parent]
                if node.index == cue:
                    return {}
                end, tag = node.index, len(words)
                for i in range(node.index + 1, len(words)):
                    if words[i] in global_cn_comma_words:
                        tag = i
                        break
                # 特殊情况： 用途用于各种功能性心律失常、室上性及室性异位期外收缩、心房纤维颤动和麻醉引起的心律不齐等
                # 自然灾害所造成的食物严重缺乏使某动物种群大量饥饿致死
                if node.relation == 'SBV':
                    if node.parent <= tag-1:
                        if postags[node.parent] in {'n', 'a'}:
                            end = node.parent
                        if postags[node.parent] == 'v' and len(words[node.parent]) == 2:
                            if node.parent == tag-1 or node.parent == len(words)-1 or node.parent < min(tag, len(words)):
                                end = node.parent
                effect = [i for i in range(cue+2, end+1)]
                for child in ptree.tree[node.index].children:
                    if node.index < child < tag:
                        effect.extend(ptree.subtree(child))
                effect = list(set(effect))
                effect.sort()
                return cls.purity(ptree, words, postags, cause, effect, cue)

            # 普通的　‘导致’
            node, res = ptree.tree[cue], []
            for i in node.children:
                if ptree.tree[i].relation == 'ADV':
                    res.extend(ptree.subtree(i))
            adv_words = set([words[r] for r in res])
            if len(adv_words & (global_cn_neg_words | global_cn_refers_words)) > 0:
                return {}
            cause, effect, tag_cause, tag_effect = [], [], False, False
            for i in node.children:
                if ptree.tree[i].relation == 'DBL':
                    effect.extend(ptree.subtree(i))
                if ptree.tree[i].relation == 'VOB':
                    effect.extend(ptree.subtree(i))
                    tag_effect = True
                if ptree.tree[i].relation == 'SBV':
                    cause.extend(ptree.subtree(i))
                    tag_cause = True
            if not tag_cause or not tag_effect:
                return {}
            # 对我的人身名誉造成严重伤害
            if len(global_cn_noise_words & set(cause)) > 0:
                return {}
            if len(set(words[cause[0]:cue]) & global_cn_refers_words) > 0:
                return {}
            i = cue-1
            while i >= 0:
                if words[i] in global_cn_conj_words:
                    cause = [r for r in range(i+1, cue)]
                    break
                i -= 1
            cause.sort()
            return cls.purity(ptree, words, postags, cause, effect, cue)
        except Exception:
            return {}

# 与水性好的沐童是死对头，两人在日后的生活中，麻烦不断，使他们的矛盾升级----麻烦(a) 断(v)----矛盾(a) 升级(v)
# 连绵的棘林偶尔会间有小片的棕榈林、盐土乾草原和由火或砍伐造成的稀树草原
# 用途用于各种功能性心律失常、室上性及室性异位期外收缩、心房纤维颤动和麻醉引起的心律不齐等
# 煎蛋炸鸡容易引发妇科恶疾----煎蛋(v) 炸鸡(n) 容易(a)----妇科恶疾(n)
# 元凶2∶过度劳累引发的腿抽筋
# 近年试作红日、雪林等，具现代气息，使人耳目一新----具(v) 气息(n)----人(n) 耳目一新(i)
# 血腥可造成的物理伤害增加15百分号持续15秒消耗35 //修饰关键词的副词 舍弃

# 自然灾害所造成的食物严重缺乏使某动物种群大量饥饿致死
# 1000年前，“播种者”在火星上培育的生物由于彗星的撞击导致毁灭----播种者(n) 火星(n) 培育(v) 生物(n)----毁灭(v)
# 怀孕期间胎儿父亲因他人侵权行为造成死亡的，婴儿出生后享有请求赔偿的权利王德钦诉杨德胜、泸州市汽车二队交通事故损害赔偿纠纷案----胎儿(n) 父亲(n)----死亡(v)
# 笔记本的丢失同样也可能引起数据丢失 //两个VOB


class PositivesRules(object):
    def __init__(self):
        return

    @classmethod
    def extract(cls, ptree):
        pos_list = []
        if not ptree.isCausal:
            return pos_list
        words, postags = [node.word for node in ptree.tree], [node.postag for node in ptree.tree]
        L = len(words)
        # Cause_V_Effect模式：<...cue(v1)...> eg: ……导致……。
        if len(ptree.bow & cn_causal_verb_set) > 0:
            i = 1
            while i < L:
                if postags[i] == 'v' and (words[i] in cn_causal_verb_set):
                    result = {'type': 'cause_v_effect', 'cue': i}
                    res = CauseVEffect.extract(ptree, result)
                    if res:
                        pos_list.append(res)
                i += 1
        return pos_list

        combo_flag = False
        # C_Cause_C_Effect模式：<cue(c4)...,cue(c5)...> eg: 因为……，所以……。
        if len(ptree.bow & cause_cue_set) > 0 and len(ptree.bow & effect_cue_set) > 0:
            i = 0
            while i < L:
                if words[i] in cause_cue_set:
                    j = i + 1
                    while j < L:
                        if words[j] in effect_cue_set:
                            if (words[i] + words[j]) in combo_cue_set:
                                combo_flag = True
                                result = {'cue1': i, 'cue2': j, 'type': 'c_cause_c_effect'}
                                res = CCauseCEffect.extract(ptree, result)
                                if res:
                                    pos_list.append(res)
                        j += 1
                i += 1

        if not combo_flag:
            # Cause_C_Effect模式：<...,cue(c2)...> eg: ……，因此……。
            if len(ptree.bow & single_effect_cue_set) > 0:
                i = 1
                while i < L:
                    if words[i] in single_effect_cue_set:
                        result = {'type': 'cause_c_effect', 'cue': i}
                        res = CauseCEffect.extract(ptree, result)
                        if res:
                            pos_list.append(res)
                    i += 1
        # C_Effect_C_Cause模式：<cue(c6)...,cue(c7)...> eg: 之所以……，因为……。
        if len(ptree.bow & reverse_cause_cue_set) > 0 and len(ptree.bow & reverse_effect_cue_set) > 0:
            i = 0
            while i < L:
                if words[i] in reverse_effect_cue_set:
                    j = i + 1
                    while j < L:
                        if words[j] in reverse_cause_cue_set:
                            if (words[i]+words[j]) in reverse_combo_cue_set:
                                result = {'cue1': i, 'cue2': j, 'type': 'c_effect_c_cause'}
                                res = CEffectCCause.extract(ptree, result)
                                if res:
                                    pos_list.append(res)
                                return pos_list
                        j += 1
                i += 1

        return pos_list
