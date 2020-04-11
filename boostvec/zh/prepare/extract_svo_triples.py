from tools import *
from extraction.cn.parser import CnParseTree, LTPModel
from extraction.cn.causalrules import Base
from extraction.cn.cueset import *
import sys


def get_pattern():
    def judge(char):
        if len(char) == 1 and char not in {'让', '令', '使'}:
            return True
        return False
    fin = codecs.open(params['pattern_path'], 'r', 'utf-8')
    pa_counter, thres = set(), params['pa_threshold']
    lines = fin.readlines()
    for line in lines:
        res = line.strip().split(' ')
        k, v = res[0], int(res[1])
        if not judge(k):
            if k not in cn_causal_verb_set and thres[0] <= v <= thres[1]:
                pa_counter.add(k)
    return pa_counter


class ExtractRules(Base):
    def __init__(self):
        Base.__init__(self)

    @staticmethod
    def purity(ptree, words, postags, cause, effect, cue):
        try:
            assert max(effect) < len(words), max(cause) < len(words)
            tag_left = ExtractRules.filter(ptree, postags, cause[0], cause[-1] + 1)
            tag_right = ExtractRules.filter(ptree, postags, effect[0], effect[-1] + 1)
            # len_cause = 21 > len(set([c for c in cause if words[c] not in punctuation])) > 2
            # len_effect = 21 > len(set([e for e in effect if words[e] not in punctuation])) > 2
            if not (tag_left and tag_right):
                return None
            if 1 < len(cause) < 23 and 1 < len(effect) < 23:
                return [cause, effect, words[cue]]
        except Exception:
            return None

    @staticmethod
    def filter(ptree, postags, start, end):
        tag, useful = False, {'v', 'n', 'i', 'a', 'd'}
        for i in range(start, end):
            if postags[i] in useful:
                return True
        return False

        # for i in range(start, end):
        #     if postags[i] in useful:
        #         tag = True
        #         break
        #     if postags[i] == 'v':
        #         for j in ptree.tree[i].children:
        #             if postags[j] in {'a', 'v', 'n', 'd', 'i'}:
        #                 tag = True
        #                 break
        #     if postags[i] in {'n', 'i'}:
        #         for j in ptree.tree[i].children:
        #             if postags[j] in {'a', 'n', 'i', 'r'}:
        #                 tag = True
        #                 break
        # return tag

    @classmethod
    def extract(cls, ptree, result):
        try:
            cue, words, postags = result['cue'], [node.word for node in ptree.tree], [node.postag for node in ptree.tree]
            node, res = ptree.tree[cue], []
            for i in node.children:
                if ptree.tree[i].relation == 'ADV':
                    res.extend(ptree.subtree(i))
            adv_words = set([words[r] for r in res])
            if len(adv_words & (global_cn_neg_words | global_cn_refers_words)) > 0:
                return None
            cause, effect, tag_left, tag_right = [], [], False, False
            for i in node.children:
                if ptree.tree[i].relation in {'DBL', 'IOB'}:
                    effect.extend(ptree.subtree(i))
                if ptree.tree[i].relation == 'VOB':
                    effect.extend(ptree.subtree(i))
                    tag_right = True
                if ptree.tree[i].relation == 'SBV':
                    cause.extend(ptree.subtree(i))
                    tag_left = True
            if not tag_right or not tag_left:
                return None
            if len(global_cn_noise_words & set(cause)) > 0:
                return None
            # i = cue - 1
            # while i >= 0:
            #     if words[i] in global_conj_words:
            #         cause = [r for r in range(i + 1, cue)]
            #         break
            #     i -= 1
            cause.sort()
            return ExtractRules.purity(ptree, words, postags, cause, effect, cue)
        except Exception:
            return None
        
        
class Handler(object):
    def __init__(self, ptree, patterns):
        self.ptree, self.patterns = ptree, patterns

    def process(self, filename):
        total, count = 0, 0
        fin = codecs.open(os.path.join(params['input_path'], filename), 'r', 'utf-8')
        fout = codecs.open(os.path.join(params['output_path'], filename), 'w', 'utf-8')
        line = fin.readline()
        while line:
            try:
                sents = self.ptree.split_sentence(line, True)
                for sent in sents:
                    try:
                        self.ptree.create(sent)
                        words = [node.word for node in self.ptree.tree]
                        postags = [node.postag for node in self.ptree.tree]
                        L, samples, response, i = len(words), [], [], 1
                        while i < L:
                            try:
                                if postags[i] == 'v' and (words[i] in self.patterns):
                                    result = {'type': 'cause_v_effect', 'cue': i}
                                    res = ExtractRules.extract(self.ptree, result)
                                    if res is not None:
                                        cause, effect, cue = res
                                        cause_words, cause_postag = ' '.join([words[c] for c in cause]), ' '.join([postags[c] for c in cause])
                                        effect_words, effect_postag = ' '.join([words[e] for e in effect]), ' '.join([postags[e] for e in effect])
                                        extraction = '----'.join([cause_words, effect_words, cause_postag, effect_postag, cue])
                                        fout.write('----'.join([sent, extraction]) + '\n')
                            except Exception:
                                pass
                            i += 1
                    except Exception:
                        pass
            except Exception:
                pass
            line = fin.readline()
        
        
def process_file(seq, l, file_list):
    ptree = CnParseTree(ltp_model)
    while not file_list.empty():
        l.acquire()
        name = file_list.get()
        l.release()
        print('Process {} is processing {} now.'.format(seq, name))
        handler = Handler(ptree=ptree, patterns=pa_set)
        handler.process(name)
        

if __name__ == '__main__':
    params = {
        'pattern_path': os.path.join(project_source_path, 'boostvec/zh/prepare/allpattern.count'),
        'pa_threshold': [2000, 500000],
        'input_path': os.path.join(project_source_path, 'sg_data/raw'),
        'output_path': os.path.join(project_source_path, 'boostvec/zh/prepare/pa2pp'),
        'num_threads': 16,
    }
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            key = arg.split("=")[0][2:]
            val = arg.split("=")[1]
            params[key] = val
    
    ltp_model = LTPModel()
    pa_set = get_pattern()
    print('length of pattern list is {}.\n'.format(len(pa_set)))

    items = os.listdir(params['input_path'])
    input_file_list = multiprocessing.Queue()
    for item in items:
        if os.path.isfile(os.path.join(params['input_path'], item)):
            input_file_list.put(item)
    lock = multiprocessing.Lock()
    thread_list = []
    for i in range(params['num_threads']):
        sthread = multiprocessing.Process(target=process_file, args=(str(i+1), lock, input_file_list))
        thread_list.append(sthread)
    for th in thread_list:
        th.start()
    for th in thread_list:
        th.join()