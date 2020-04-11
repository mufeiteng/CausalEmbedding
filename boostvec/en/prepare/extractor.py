from spacy import displacy
from tools import sharp_causal_verb_set
import re

"""
VB: verb, base form
VBD: verb, past tense 过去时态
VBG: verb, present participle or gerund 现在分词或动名词
VBN: verb, past participle 过去分词
VBP: verb, present tense, not 3rd person singular 现在时态，非3单
VBZ: verb, present tense,3rd person singular 现在时态，3单

JJ: adjective or numeral, ordinal 形容词或序数词
JJR: adjective, comparative 形容词比较级
JJS: adjective, superlative 形容词最高级

MD: modal auxiliary 情态助动词
NN: noun, common, singular or mass 单数或质量
NNS: noun, common, plural 复数
NNP: noun, proper, singular 单数
NNPS: noun, proper, plural

RB: adverb 副词
RBR: adverb, comparative 副词比较级
RBS: adverb, superlative 副词最高级

dobj : direct object直接宾语
iobj : indirect object，非直接宾语，也就是所以的间接宾语；
nsubj : nominal subject，名词主语
nsubjpass: passive nominal subject，被动的名词主语
aux: auxiliary，非主要动词和助词，如BE,HAVE SHOULD/COULD等到
auxpass: passive auxiliary 被动词
"""


class EnExtractor(object):
    global_en_comma = {',', ';'}
    stop_verbs = {'am', 'is', 'are', 'was', 'were', 'being', 'been', 'do', 'did', 'does'} | sharp_causal_verb_set
    prep_words = {'with', 'into', 'on', 'to', 'in', 'of', 'from', 'for', 'about'}

    @staticmethod
    def visualize(doc):
        displacy.serve(doc, style='dep', options={'distance': 120})

    @staticmethod
    def show(doc):
        s = "{0}/{1} <--{2}-- {3}/{4}"
        for token in doc:
            print(s.format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))

    @staticmethod
    def purge_sent(s):
        s = re.sub(r"\(.*\)", '', s)
        s = re.sub(r"\[.*\]", '', s)
        s = re.sub("[\n<>\"|/?(){}@!%&*_+=“”．，；＇]", '', s)
        s = re.sub(' +', ' ', s)
        return s

    @staticmethod
    def judge_para_valid(s):
        if s.startswith('<doc') or s.startswith('/doc>') or s.startswith('curid=') or len(s) < 40:
            return False
        return True

    def regex_split_en_para(self, paragraph):
        _res = list()
        for s_str in paragraph.split('.'):
            if '?' in s_str:
                _res.extend(s_str.split('?'))
            elif '!' in s_str:
                _res.extend(s_str.split('!'))
            else:
                _res.append(s_str)
        return [self.purge_sent(s) for s in _res]

    def split_en_para(self, paragraph):
        _res = self.splitter.tokenize(paragraph)
        return [self.purge_sent(s) for s in _res]

    @staticmethod
    def get_response(arg1_words, arg2_words, arg1_tags, arg2_tags, cue):
        def strip(ws):
            return [w.strip('\n') for w in ws]

        _res = '----'.join(
            (' '.join(strip(arg1_words)), ' '.join(strip(arg2_words)),
             ' '.join(strip(arg1_tags)), ' '.join(strip(arg2_tags)), cue)
        )
        return _res

    def __init__(self, sent_splitter_model, stem_model, ptree):
        self.splitter = sent_splitter_model
        self.ptree = ptree
        self.stemmer = stem_model

    def get_sbv(self, node):
        for idx in node.childs:
            if self.ptree.tree[idx].dep.startswith('nsubj'):
                return True, self.ptree.subtree(idx)
        return False, None

    def judge_arg_valid(self, items):
        for idx in items:
            w = self.ptree.tree[idx]
            if w.tag in {'NN', 'NNS'} or w.tag.startswith('VB') or w.tag.startswith('JJ') or w.tag.startswith('RB'):
                return True
        return False

    def judge_passive(self, node):
        if node.tag != 'VBN':
            return False
        for idx in node.childs:
            if self.ptree.tree[idx].dep == 'auxpass':
                return True
        return False

    def judge_active(self, node):
        tag1 = node.dep == 'advcl' and node.tag == 'VBG'
        tag2 = self.ptree.tree[node.id-1].text in self.global_en_comma
        tag3 = node.id > node.head
        return tag1 and tag2 and tag3

    def get_main_clause(self, node):
        # i = node.id-2
        # while i >= 0:
        #     if self.ptree.tree[i].text in self.global_en_comma:
        #         break
        #     i -= 1
        # return list(range(i+1, node.id-1))
        _res = []
        childs = self.ptree.tree[node.head].childs
        for idx in childs:
            if idx < node.id:
                _res.extend(self.ptree.subtree(idx))
        if len(_res) > 0:
            _res.append(node.head)
            _res.sort()
            return _res
        return []

    def get_comp(self, node):
        for idx in node.childs:
            if idx > node.id and self.ptree.tree[idx].dep.endswith('comp'):
                tokens = self.ptree.subtree(idx)
                end = len(tokens)
                for i, token in enumerate(tokens):
                    if self.ptree.tree[tokens[i]].text in self.global_en_comma:
                        end = i
                        break
                return tokens[:end]
        return []

    def get_prep_obj(self, node):
        if node.text in self.prep_words and node.dep in {'dative', 'prep'} and node.tag == 'IN':
            for child in node.childs:
                if child > node.id and self.ptree.tree[child].dep == 'pobj':
                    tokens = self.ptree.subtree(child)
                    end = len(tokens)
                    for i, token in enumerate(tokens):
                        if self.ptree.tree[tokens[i]].text in self.global_en_comma:
                            end = i
                            break
                    return True, tokens[:end], node.text
        return False, [], None

    def convert(self, sbv, root, obj, pattern):
        if len(sbv) == 0 or len(obj) == 0:
            return None
        vp = pattern.split('_')
        if self.judge_passive(root) and len(vp) > 1:
            pattern = vp[0]
        arg1, arg2 = [], []
        for idx in sbv:
            c = self.ptree.tree[idx]
            if c.text in {' ', ''} or c.tag in {' ', ''}:
                continue
            arg1.append((c.text, c.tag))
        for idx in obj:
            e = self.ptree.tree[idx]
            if e.text in {' ', ''} or e.tag in {' ', ''}:
                continue
            arg2.append((e.text, e.tag))
        if not (0 < len(arg1) < 25 and 0 < len(arg2) < 25):
            return None
        arg1_words, arg1_tags = zip(*arg1)
        arg2_words, arg2_tags = zip(*arg2)
        if self.judge_passive(root):
            _res = self.get_response(arg2_words, arg1_words, arg2_tags, arg1_tags, pattern)
        else:
            _res = self.get_response(arg1_words, arg2_words, arg1_tags, arg2_tags, pattern)
        return _res

    def get_dobj(self, node):
        tag, obj = False, []
        for idx in node.childs:
            if self.ptree.tree[idx].dep == 'dobj':
                obj.extend(self.ptree.subtree(idx))
                return True, obj
        return False, None

    def stem(self, w, postag='v'):
        return self.stemmer.lemmatize(w, postag)

    def extract_from_doc(self, doc):
        _response = []
        self.ptree.create(doc)
        for node in self.ptree.tree:
            if not node.tag.startswith('VB') or node.text in self.stop_verbs:
                continue
            tag1, sbv = self.get_sbv(node)
            if tag1:
                if not (len(sbv) > 0 and self.judge_arg_valid(sbv)):
                    continue
                tag2, obj = self.get_dobj(node)
                if tag2:
                    pobj_tag = False
                    for idx in node.childs:
                        if idx <= node.id:
                            continue
                        prep_tag, prep_obj, prep_text = self.get_prep_obj(self.ptree.tree[idx])
                        if prep_tag:
                            if self.judge_arg_valid(prep_obj):
                                item1 = self.convert(sbv, node, obj, self.stem(node.text))
                                if item1 is not None:
                                    _response.append(item1)
                                item2 = self.convert(sbv, node, prep_obj, self.stem(node.text) + '_' + prep_text)
                                if item2 is not None:
                                    _response.append(item2)
                                pobj_tag = True
                                break
                    if not pobj_tag:
                        comp_tag = False
                        for idx in node.childs:
                            if idx > node.id and self.ptree.tree[idx].dep.endswith('comp'):
                                obj.extend(self.ptree.subtree(idx))
                                if self.judge_arg_valid(obj):
                                    item = self.convert(sbv, node, obj, self.stem(node.text))
                                    if item is not None:
                                        _response.append(item)
                                    comp_tag = True
                                    break
                        if not comp_tag:
                            if self.judge_arg_valid(obj):
                                item = self.convert(sbv, node, obj, self.stem(node.text))
                                if item is not None:
                                    _response.append(item)
                else:
                    prep_tag, prep_obj, prep_text = self.get_prep_obj(self.ptree.tree[node.id + 1])
                    if prep_tag and self.judge_arg_valid(prep_obj):
                        item = self.convert(sbv, node, prep_obj, self.stem(node.text)+'_'+prep_text)
                        if item is not None:
                            _response.append(item)
                    if not prep_tag:
                        ccomp = self.get_comp(node)
                        if len(ccomp) > 1:
                            item = self.convert(sbv, node, ccomp, self.stem(node.text))
                            if item is not None:
                                _response.append(item)
            else:
                if not self.judge_active(node):
                    continue
                main_clause = self.get_main_clause(node)
                if not (len(main_clause) > 3 and self.judge_arg_valid(main_clause)):
                    continue
                prep_tag, prep_obj, prep_text = self.get_prep_obj(self.ptree.tree[node.id + 1])
                if prep_tag and len(prep_obj) > 0 and self.judge_arg_valid(prep_obj):
                    assert prep_text is not None
                    item = self.convert(main_clause, node, prep_obj, self.stem(node.text)+'_'+prep_text)
                    if item is not None:
                        _response.append(item)
                if not prep_tag:
                    ccomp = self.get_comp(node)
                    if len(ccomp) > 1:
                        item = self.convert(main_clause, node, ccomp, self.stem(node.text))
                        if item is not None:
                            _response.append(item)
        return _response
