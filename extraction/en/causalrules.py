import re
import nltk


def join(w1, w2): return '{}_{}'.format(w1, w2)


def judge_causal(s):
    item1 = "cause|causes|caused|create|created|creates|spark|sparked|sparks"
    item2 = "provoke|provokes|provoked|ignite|ignites|ignited|prompt|prompts|prompted"
    item3 = "lead|leads|led|result|results|resulted|trigger|triggers|triggered"
    item4 = "contribute|contributes|contributed|bring|brings|brought|stem|stems|stemmed"
    p = '|'.join([item1, item2, item3, item4])
    p = "(.+) (?:{}) (.+)".format(p)
    if re.match(p, s):
        return True
    return False


class Base(object):
    global_en_comma = {',', ';'}
    en_causal_vt = {'cause', 'create', 'spark', 'trigger', 'provoke', 'ignite', 'prompt'}
    prep_cue = {'to', 'in'}

    @staticmethod
    def judge_passive(ptree, node):
        if node.pos != 'VBN':
            return False
        for idx in node.childs:
            if ptree.tree[idx].dep == 'auxpass':
                return True
        return False

    @staticmethod
    def judge_active(ptree, node):
        if node.pos in {'VB', 'VBP', 'VBZ', 'VBD'}:
            return True
        if node.pos == 'VBN':
            for idx in node.childs:
                if ptree.tree[idx].dep == 'auxpass':
                    return False
            for idx in node.childs:
                if ptree.tree[idx].dep == 'aux':
                    return True
        return False

    @staticmethod
    def get_sbv(ptree, node):
        tag, sbv = False, []
        for idx in node.childs:
            if ptree.tree[idx].dep.startswith('nsubj') or ptree.tree[idx].dep.startswith('csubj'):
                tag = True
                sbv.extend(ptree.subtree(idx))
        return tag, sbv

    @staticmethod
    def get_dobj(ptree, node):
        tag, obj = False, []
        for idx in node.childs:
            if ptree.tree[idx].dep == 'dobj':
                obj.extend(ptree.subtree(idx))
                return True, obj
        return False, None

    def get_agent_obj(self, ptree, node):
        for child in node.childs:
            if child > node.idx and ptree.tree[child].text == 'by':
                agent = ptree.tree[child]
                for idx in agent.childs:
                    if idx > agent.idx and ptree.tree[idx].dep in {'pobj', 'pcomp'}:
                        tokens = ptree.subtree(idx)
                        end = len(tokens)
                        for i, token in enumerate(tokens):
                            if ptree.tree[tokens[i]].text in self.global_en_comma:
                                end = i
                                break
                        return True, tokens[:end], 'by'
        return False, [], None

    def get_prep_obj(self, ptree, node, prep_set):
        for child in node.childs:
            cue = ptree.tree[child]
            if child > node.idx and cue.dep in {'dative', 'prep'} and cue.pos == 'IN' and cue.text in prep_set:
                pobj_tokens, count = [], 0
                for idx in cue.childs:
                    if idx > cue.idx and ptree.tree[idx].dep in {'pobj', 'pcomp'}:
                        tokens = ptree.subtree(idx)
                        end = len(tokens)
                        for i, token in enumerate(tokens):
                            if ptree.tree[tokens[i]].text in self.global_en_comma:
                                end = i
                                break
                        if count < 2:
                            pobj_tokens.extend(tokens[:end])
                            count += 1
                if len(pobj_tokens) > 0:
                    return True, pobj_tokens, cue
        return False, [], None

    @staticmethod
    def judge_arg_valid(ptree, items):
        for idx in items:
            w = ptree.tree[idx]
            if w.pos in {'NN', 'NNS'} or w.pos.startswith('VB') or w.pos.startswith('JJ') or w.pos.startswith('RB'):
                return True
        return False


class CausalVerbRules(Base):
    def __init__(self, stem_model):
        self.stemmer = stem_model

    def stem(self, w, postag='v'):
        return self.stemmer.lemmatize(w, postag)

    def extract(self, ptree, sent):
        # active voice of vt, perfect tense for active voice of vt
        for node in ptree.tree:
            stem = self.stem(node.text)
            if stem in self.en_causal_vt and stem != 'create' and self.judge_active(ptree, node):
                sbv_tag, subj = self.get_sbv(ptree, node)
                if not (sbv_tag and self.judge_arg_valid(ptree, subj)):
                    return None
                dobj_tag, dobj = self.get_dobj(ptree, node)
                if not (dobj_tag and self.judge_arg_valid(ptree, dobj)):
                    return None
                return [ptree.tree[i] for i in subj], [ptree.tree[i] for i in dobj], node.text

        # passive voice of vt, or perfect tense for passive voice of vt
        for node in ptree.tree:
            if self.stem(node.text) in self.en_causal_vt and self.judge_passive(ptree, node):
                sbv_tag, subjpass = self.get_sbv(ptree, node)
                if not (sbv_tag and self.judge_arg_valid(ptree, subjpass)):
                    return None
                agent_tag, agent_obj, prep_text = self.get_agent_obj(ptree, node)
                if not (agent_tag and self.judge_arg_valid(ptree, agent_obj)):
                    return None
                return [ptree.tree[i] for i in agent_obj], [ptree.tree[i] for i in subjpass], join(node.text, prep_text)

        # stem_from, result_from (regular tense)
        for node in ptree.tree:
            if self.stem(node.text) in {'stem', 'result'} and self.judge_active(ptree, node):
                sbv_tag, subj = self.get_sbv(ptree, node)
                if not (sbv_tag and self.judge_arg_valid(ptree, subj)):
                    return None
                prep_tag, prep_obj, prep_text = self.get_prep_obj(ptree, node, prep_set={'from'})
                if not (prep_tag and self.judge_arg_valid(ptree, prep_obj)):
                    return None
                return [ptree.tree[i] for i in prep_obj], [ptree.tree[i] for i in subj], join(node.text, prep_text)

        # lead_to, result_in, contribute_to, bring_about (regular tense)
        for node in ptree.tree:
            if self.stem(node.text) in {'lead', 'result', 'contribute'} and self.judge_active(ptree=ptree, node=node):
                sbv_tag, subj = self.get_sbv(ptree, node)
                if not (sbv_tag and self.judge_arg_valid(ptree, subj)):
                    return None
                prep_tag, prep_obj, prep_text = self.get_prep_obj(ptree, node, prep_set=self.prep_cue)
                if not (prep_tag and self.judge_arg_valid(ptree, prep_obj)):
                    return None
                # finding its complement is necessary to assure the precision of extractions.
                return [ptree.tree[i] for i in subj], [ptree.tree[i] for i in prep_obj], join(node.text, prep_text)
        for node in ptree.tree:
            if self.stem(node.text) == 'bring' and self.judge_active(ptree, node):
                sbv_tag, subj = self.get_sbv(ptree, node)
                if not (sbv_tag and self.judge_arg_valid(ptree, subj)):
                    return None
                prep_tag, prep_obj, prep_text = self.get_prep_obj(ptree, node, prep_set={'about'})
                if not (prep_tag and self.judge_arg_valid(ptree, prep_obj)):
                    return None
                return [ptree.tree[i] for i in subj], [ptree.tree[i] for i in prep_obj], join(node.text, prep_text)

        # special rules, to be continue


class CausalConjRules(object):
    max_len, min_len, diff_thres = 30, 4, 15
    global_en_comma = {',', ';'}

    @classmethod
    def tokenizer(cls, arg):
        return nltk.pos_tag(nltk.word_tokenize(arg))

    @classmethod
    def judge_arg_valid(cls, tokens):
        for (word, pos) in tokens:
            # if pos.startswith('VB') or pos.startswith('JJ') or pos.startswith('RB') or pos in {'NN', 'NNS'}:
            if pos.startswith('VB'):
                return True
        return False

    @classmethod
    def split_args(cls, arg):
        response, temp = [], []
        for item in arg:
            w, pos = item
            temp.append(item)
            if pos in cls.global_en_comma:
                response.append(temp)
                temp = []
        if len(temp) > 0:
            response.append(temp)
        return response

    @classmethod
    def truncate_arg(cls, arg, target_len, reverse=False):
        tokens = cls.tokenizer(arg)
        if reverse:
            tokens.reverse()
        phrases = cls.split_args(tokens)
        response = []
        for span in phrases:
            if len(response) + len(span) <= target_len:
                response.extend(span)
            else:
                break
        if reverse:
            response.reverse()
        return response

    pattern_set1 = {
        '(?P<cause>.+)(?P<cue>, leading to) (?P<effect>.+)',
        '(?P<cause>.+)(?P<cue>, resulting in) (?P<effect>.+)',
        '(?P<cause>.+)(?P<cue>, contributing in) (?P<effect>.+)',
        '(?P<cause>.+)(?P<cue>, bringing about) (?P<effect>.+)',
        '(?P<effect>.+)(?P<cue>, resulting from) (?P<cause>.+)',
        '(?P<effect>.+)(?P<cue>, spring from) (?P<cause>.+)',
        '(?P<effect>.+)(?P<cue>, owing to) (?P<cause>.+)'
    }
    @classmethod
    def extract_from_pattern_set1(cls, sent):
        response = []
        for p in cls.pattern_set1:
            g = re.match(p, sent)
            if not g:
                continue
            arg1, arg2, cue = g.group('cause'), g.group('effect'), g.group('cue')
            arg1_tokens, arg2_tokens = cls.truncate_arg(arg1, cls.max_len, reverse=True), cls.truncate_arg(arg2, cls.max_len, reverse=False)
            if len(arg1_tokens) < cls.min_len or len(arg2_tokens) < cls.min_len:
                continue
            if abs(len(arg1_tokens) - len(arg2_tokens)) > cls.diff_thres:
                continue
            # if not judge_arg_valid(arg1_tokens) or not judge_arg_valid(arg2_tokens):
            #     continue
            response.append([arg1_tokens, arg2_tokens, cue, sent])
        return response

    pattern_set2 = {'Consequently,', 'As a result,', 'Accordingly,', 'Therefore,', 'Thus,', 'Hence,', 'As a consequence,'}
    @classmethod
    def extract_from_pattern_set2(cls, sent, prev):
        response = []
        for p in cls.pattern_set2:
            if not sent.startswith(p):
                continue
            arg1, cue, arg2 = prev, p, sent[len(p) + 1:]
            if arg2.startswith('of'):
                continue
            arg1_tokens, arg2_tokens = cls.truncate_arg(arg1, cls.max_len, reverse=True), cls.truncate_arg(arg2, cls.max_len, reverse=False)
            if len(arg1_tokens) < cls.min_len or len(arg2_tokens) < cls.min_len:
                continue
            if abs(len(arg1_tokens) - len(arg2_tokens)) > cls.diff_thres:
                continue
            # if not judge_arg_valid(arg1_tokens) or not judge_arg_valid(arg2_tokens):
            #     continue
            response.append([arg1_tokens, arg2_tokens, cue, prev+sent])
        return response

    pattern_set3 = {
        '(?P<cause>.+)(?P<cue>, hence) (?P<effect>.+)',
        '(?P<cause>.+)(?P<cue>, and hence) (?P<effect>.+)',
        '(?P<cause>.+)(?P<cue>, thus) (?P<effect>.+)',
        '(?P<cause>.+)(?P<cue>, and thus) (?P<effect>.+)',
        '(?P<cause>.+)(?P<cue>, therefore) (?P<effect>.+)',
        '(?P<cause>.+)(?P<cue>, and therefore) (?P<effect>.+)'
    }
    @classmethod
    def extract_from_pattern_set3(cls, sent):
        response = []
        for p in cls.pattern_set3:
            g = re.match(p, sent)
            if not g:
                continue
            arg1, cue, arg2 = g.group('cause'), g.group('cue'), g.group('effect')
            arg1_tokens, arg2_tokens = cls.truncate_arg(arg1, cls.max_len, reverse=True), cls.truncate_arg(arg2, cls.max_len, reverse=False)
            if len(arg1_tokens) < cls.min_len or len(arg2_tokens) < cls.min_len:
                continue
            if abs(len(arg1_tokens) - len(arg2_tokens)) > cls.diff_thres:
                continue
            if not cls.judge_arg_valid(arg1_tokens) or not cls.judge_arg_valid(arg2_tokens):
                continue
            response.append([arg1_tokens, arg2_tokens, cue, sent])
        return response

    pattern_set4 = {
        '(?P<cause>.+)(?P<cue>, as (?:a|the) result) (?P<effect>.+)',
        '(?P<cause>.+)(?P<cue>, and as (?:a|the) result) (?P<effect>.+)',
        '(?P<cause>.+)(?P<cue>, as (?:a|the) consequence) (?P<effect>.+)',
        '(?P<cause>.+)(?P<cue>, and as (?:a|the) consequence) (?P<effect>.+)',
        '(?P<cause>.+)(?P<cue>, so that) (?P<effect>.+)'
    }
    @classmethod
    def extract_from_pattern_set4(cls, sent):
        response = []
        for p in cls.pattern_set4:
            g = re.match(p, sent)
            if not g:
                continue
            arg1, cue, arg2 = g.group('cause'), g.group('cue'), g.group('effect')
            if arg2.startswith('of'):
                continue
            arg1_tokens, arg2_tokens = cls.truncate_arg(arg1, cls.max_len, reverse=True), cls.truncate_arg(arg2, cls.max_len, reverse=False)
            if len(arg1_tokens) < cls.min_len or len(arg2_tokens) < cls.min_len:
                continue
            if abs(len(arg1_tokens) - len(arg2_tokens)) > cls.diff_thres:
                continue
            # if not judge_arg_valid(arg1_tokens) or not judge_arg_valid(arg2_tokens):
            #     continue
            response.append([arg1_tokens, arg2_tokens, cue, sent])
        return response

    pattern_set5 = {
        '(?P<effect>.+)(?P<cue>, due to) (?P<cause>.+)',
        '(?P<effect>.+)(?P<cue>, because of) (?P<cause>.+)',
        '(?P<effect>.+)(?P<cue>, owing to) (?P<cause>.+)',
        '(?P<effect>.+)(?P<cue>, as (?:a|the) result of) (?P<cause>.+)',
        '(?P<effect>.+)(?P<cue>, as (?:a|the) consequence of) (?P<cause>.+)',
        '(?P<effect>.+)(?P<cue>, inasmuch as) (?P<cause>.+)'
    }
    @classmethod
    def extract_from_pattern_set5(cls, sent):
        def judge_valid(tokens):
            for w, _ in tokens:
                if w in cls.global_en_comma:
                    return False
            return True

        response = []
        for p in cls.pattern_set5:
            g = re.match(p, sent)
            if not g:
                continue
            arg1, cue, arg2 = g.group('cause'), g.group('cue'), g.group('effect')
            arg1_tokens, arg2_tokens = cls.truncate_arg(arg1, cls.max_len, reverse=True), cls.truncate_arg(arg2, cls.max_len, reverse=False)
            if not judge_valid(arg1_tokens):
                continue
            if len(arg1_tokens) < cls.min_len or len(arg2_tokens) < cls.min_len:
                continue
            if abs(len(arg1_tokens) - len(arg2_tokens)) > cls.diff_thres:
                continue
            # if not judge_arg_valid(arg1_tokens) or not judge_arg_valid(arg2_tokens):
            #     continue
            response.append([arg1_tokens, arg2_tokens, cue, sent])
        return response

    pattern_set6 = {
        '(?P<cue>(?:Because|Because of)) (?P<cause>.+?), (?P<effect>.+?)',
        '(?P<cue>Inasmuch as) (?P<cause>.+?), (?P<effect>.+)',
        '(?P<cue>Due to) (?P<cause>.+?), (?P<effect>.+)',
        '(?P<cue>Owing to) (?P<cause>.+?), (?P<effect>.+)',
        '(?P<cue>As (?:a|the) result of) (?P<cause>.+?), (?P<effect>.+)',
        '(?P<cue>As (?:a|the) consequence of) (?P<cause>.+?), (?P<effect>.+)',
    }
    @classmethod
    def extract_from_pattern_set6(cls, sent):
        response = []
        for p in cls.pattern_set6:
            g = re.match(p, sent)
            if not g:
                continue
            tokens = cls.tokenizer(sent)
            if len([1 for w, _ in tokens if w in cls.global_en_comma]) != 1:
                continue
            arg1, cue, arg2 = g.group('cause'), g.group('cue'), g.group('effect')
            arg1_tokens, arg2_tokens = cls.truncate_arg(arg1, cls.max_len, reverse=True), cls.truncate_arg(arg2, cls.max_len, reverse=False)
            if len(arg1_tokens) < cls.min_len or len(arg2_tokens) < cls.min_len:
                continue
            if abs(len(arg1_tokens) - len(arg2_tokens)) > cls.diff_thres:
                continue
            # if not judge_arg_valid(arg1_tokens) or not judge_arg_valid(arg2_tokens):
            #     continue
            response.append([arg1_tokens, arg2_tokens, cue, sent])
        return response

    @classmethod
    def extract(cls, prev_sent, current_sent):
        response = []
        response.extend(cls.extract_from_pattern_set1(current_sent))
        response.extend(cls.extract_from_pattern_set2(current_sent, prev_sent))
        response.extend(cls.extract_from_pattern_set3(current_sent))
        response.extend(cls.extract_from_pattern_set4(current_sent))
        response.extend(cls.extract_from_pattern_set5(current_sent))
        response.extend(cls.extract_from_pattern_set6(current_sent))
        return response
