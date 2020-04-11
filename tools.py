# -*- coding: utf-8 -*-
import codecs
import numpy as np
import os
import multiprocessing
from time import time
from functools import wraps

#
# stopwords = {
#     '', 'lol', 'latterly', 'besides', 'unfortunately', 'ones', 'corresponding', 'anybody', 'comes',
#     'just', 'little', 'viz', 'outside', 'wants', 't', '?', "hasn't", 'way', "'d", 'somewhat',
#     'off', 'by', 'oh', 'upon', 'last', 'described', 'never', 'later', 'edu', 'became', 'follows',
#     're', 'yourself', 'ought', 'relatively', 'twice', 'since', 'four', 'five', 'third', 'itself',
#     'inward', 'everywhere', 'respectively', 'secondly', 'afterwards', 's', 'given', 'an', 'after',
#     'cant', 'trying', 'changes', 'ok', "they'll", 'normally', 'better', 'we', 'hereafter',
#     'other', 'allows', 'alone', "you'll", 'merely', 'self', 'wherein', 'with', 'others',
#     'unlikely', 'aren', "he'd", 'wasn', 'certain', 'like', 'except', 'latter', 'say', 'qv',
#     'causes', 'near', 'hasn', 'doing', 'eight', 'i', 'want', 'how', 'won', 'who', ',', 'right',
#     'himself', 'when', 'hadn', "it's", "let's", 'into', 'did', 'those', 'sensible', 'didn',
#     'between', 'weren', 'above', 'consequently', "how's", 'gotten', 'so', 'thence', 'nine',
#     'anyone', 'if', 'either', 'u', 'have', 'what', 'almost', 'know', 'enough', 'here',
#     "shouldn't", 'containing', 'without', 'be', 'particular', 'theres', 'able', 'more',
#     'mainly', "'m", 'many', 'hello', 'selves', 'yes', "that's", 'make', 'or', 'vs', 'will',
#     'good', 'entirely', 'non', 'the', 'awfully', 'everyone', 'also', 'nothing', 'possible',
#     'no', 'name', 'hither', 'brief', 'yet', "it'll", "when's", 'sorry', 'together', 'why',
#     'thereafter', 'herself', "didn't", 'onto', 'its', 'available', 'though', 'for', 'which',
#     'm', 'around', 'knows', "where's", 'different', 'especially', 'gets', 'quite', 'thoroughly',
#     'consider', 'very', "ain't", 'her', 'goes', 'seven', 'haha', 'et', "she's", 'than',
#     "doesn't", 'anything', 'greetings', 'd', 'll', 'first', "isn't", 'best', 'try', 'noone',
#     'regardless', 'various', 'themselves', 'whereupon', 'wherever', 'said', 'however', 'none',
#     "there's", 'are', 'likely', 'they', 'you', 'tries', 'contain', 'thank', 'okay', 'indeed',
#     'therein', 'seriously', 'was', 'there', 'anyway', 'new', 'amongst', 'associated', 'already',
#     "i'll", "we'd", 'kept', 'thereupon', 'through', 'sometimes', 'where', "he'll", 'being',
#     'perhaps', "needn't", "they'd", 'it', 'theirs', 'tell', 'eg', 'nor', 'she', 'hence', 'sub',
#     'see', 'truly', "here's", 'insofar', 'indicate', 'o', 'usually', 'followed', 'yours', 'done',
#     'in', 'my', 'ourselves', 'using', 'looks', "that'll", 'per', "who's", 'on', 'get', 'unless',
#     'clearly', 'sup', 'somebody', 'allow', 'beside', 'meanwhile', 'whom', 'then', 'had',
#     'herein', 'any', 'wouldn', 'asking', 'from', 'ever', 'seem', 'whenever', 'saying', 'someone',
#     'reasonably', 'definitely', 'always', 'despite', 'while', 'hereupon', 'your', 'please', 'un',
#     'fifth', 'uses', 'placed', "it'd", 'day', 'across', 'accordingly', 'because', 'considering',
#     'needs', 'shouldn', 'own', 'ask', "i've", 'lately', 'whence', 'still', 'actually', "mustn't",
#     'everybody', "you're", 'thorough', 'should', 'overall', 'hopefully', 'yourselves',
#     'indicated', 'anyhow', 'ain', 'must', 'a', 'certainly', 'hereby', 'thru', 'elsewhere',
#     'shall', 'example', "won't", 'whether', 'anyways', 'seems', 'anywhere', 'downwards', 'namely',
#     'such', 'nearly', 'mostly', 'wish', 'further', 'appreciate', 'gone', "i'm", 'came',
#     'contains', 'today', 'until', "can't", "a's", 'specifying', 'not', 'don', "t's", 'obviously',
#     'become', 'do', 'somehow', 'thanx', 'formerly', 'people', 'howbeit', 'six', 'que', 'soon',
#     'before', 'were', "wouldn't", 'love', "she'll", 'value', 'that', 'been', 'wonder', 'thereby',
#     'each', 'mustn', 'along', "she'd", 'y', 'happens', 'ltd', 'back', 'nevertheless', 'does',
#     'moreover', 'inc', 'appear', 'apart', 'whatever', 'several', 'whereas', 'come', 'furthermore',
#     'taken', 'doesn', 'use', 'sometime', 'to', 'neither', "aren't", 'else', 'under', "c's",
#     'appropriate', 'beyond', 'behind', 'haven', 'up', "they've", "hadn't", 'less', 'out',
#     'regarding', "haven't", 'too', 'rather', "we'll", 'once', 'lest', 'whereafter', 'hi',
#     'toward', 'really', 've', "'ll", 'needn', 'via', 'within', 'every', 'willing', 'whose', 'as',
#     'us', "you've", 'down', 'necessary', 'tried', 'can', 'over', 'is', "i'd", 'but', "what's",
#     "mightn't", 'thus', 'course', 'much', 'his', "why's", 'shan', 'getting', 'went', 'three',
#     'aside', 'these', 'otherwise', "wasn't", 'same', 'ours \tourselves', 'least', 'inasmuch',
#     'forth', 'help', 'seeming', 'mean', 'most', 'whoever', 'isn', 'th', 'next', 'among', '.',
#     'ignored', 'believe', "weren't", 'tends', "they're", 'nd', 'of', "he's", 'plus', 'look',
#     "we're", 'some', 'couldn', 'our', 'he', 'mightn', 'another', 'known', 'at', 'keeps', 'hers',
#     'having', 'probably', 'specified', 'against', 'him', 'me', 'towards', 'gives', 'com', 'inner',
#     '2', "c'mon", 'two', 'well', 'thats', "n't", 'time', 'seen', 'everything', 'ma', 'ex',
#     'immediate', 'particularly', 'even', 'could', 'zero', 'them', 'would', 'again', 'got',
#     "should've", 'something', 'take', 'looking', 'exactly', 'hardly', 'saw', 'although', 'im',
#     'maybe', 'following', 'has', 'may', 'indicates', 'let', 'think', 'sure', 'provides', 'going',
#     'during', 'ours', "couldn't", 'often', 'whole', 'concerning', 'might', "don't", 'currently',
#     'now', 'this', 'both', "shan't", '!', 'specify', 'cannot', 'one', 'unto', 'serious', 'ie',
#     'old', 'took', 'beforehand', 'away', 'used', "you'd", 'according', 'second', 'need', 'seeing',
#     'id', 'below', 'few', 'nowhere', 'only', "'s", 'presumably', 'etc', 'all', 'thanks', 'their',
#     'says', 'cause', 'about', 'co', 'becomes', 'regards', 'sent', 'go', 'former', 'rd',
#     'therefore', 'whereby', 'am', 'useful', 'throughout', 'welcome', 'myself', 'becoming',
#     'somewhere', 'and', 'keep', 'seemed', 'far', 'instead', 'whither', 'nobody', 'novel', "'t",
#     'liked', "we've"
# }
#
# stopverbs = {
#     'get', 'become', 'pay', 'function', 'exist', 'return', "'ve", 'king', 'boys', 'be', 'join',
#     'load', 'rank', '–', 'top', 'let', 'look', 'date', 'teach', 'total', 'cede', 'plan', 'alfred',
#     'run', 'use', 'study', 'charles', 'call', 'feature', 'back', 'root', 'need', 'fight', 'film',
#     'turn', 'come', 'canada', 'trade', 'want', 'john', 'talk', 'remember', 'see', 'ten', 'angeles',
#     'commence', 'make', 'think', 'swim', 'london', 'send', 'fee', 'man', 'andrew', 'state', 'aid',
#     'add', 'mount', 'sing', 'continue', 'end', 'edit', 'home', 'william', 'leave', 'south', "'m",
#     'score', 'open', 'like', 'neighbour', 'find', 'power', 'hop', 'commission', 'fly', 'test',
#     'oversee', 'mean', 'string', 'i', 'europe', 'produce', 'marry', 'include', 'tend', 'quote',
#     'islands', 'ask', "'s", 'note', 'stand', 'sound', 'flow', 'span', 'air', 'value', 'set', 'voice',
#     'rename', 'have', 'color', 'approach', '’', 'neighbor', 'worldwide', 'go', 'lake', 'read', 'give',
#     'start', 'enter', 'england', 'point', 'love', 'own', 'last', 'range', 'welcome', 'speak', 'station',
#     'eat', 'work', 'watch', 'rat', 'china', 'say', 'term', 'farm', 'headquarter', 'listen', 'borrow',
#     'meet', 'appear', 'dance', 'project', 'tell', 'seem', 'dub', 'sleep', 'map', 'center', 'nickname',
#     'live', 'land', 'travel', 'explain', 'pass', 'double', 'experiment', 'number', 'coin', 'sport',
#     'list', 'hand', 'happen', 'follow', 'offer', 'target', "'re", 'play', 'reach', 'name', 'show',
#     'face', 'place', 's', 'buy', 'step', 'know', 'view', 'relate', 'house', 'write', '—', 'reference',
#     'visit', 'ring', 'document', 'walk', 'god', 'agree', 'put', 'age', 'stop', 'do', 'sit', 'begin',
#     'take', 'aim'
# }

global_useful_postag = {'v', 'd', 'a', 'n', 'i'}

sharp_causal_verb_set = {
    'lead', 'leads', 'led', 'leading', 'cause', 'caused', 'causing', 'causes',
    'create', 'creates', 'created', 'creating', 'result', 'results', 'resulted', 'resulting'
}

project_source_path = os.path.join(os.path.expanduser('~'), 'Documents/sources/')
# project_source_path = os.path.join(os.path.expanduser('~'), 'data_dir/feiteng/sources')


ltp_path = os.path.join(project_source_path, 'ltp_data/')
stanford_nlp_path = '/usr/local/share/stanford_parser/'
allennlp_parser_dir = os.path.join(project_source_path, 'allennlp/')


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        res = func(*args, **kwargs)
        end_time = time()
        print('function {} uses {} s.'.format(func.__name__, end_time-start_time))
        return res
    return wrapper


def logfunc(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print('{} starts, waiting...'.format(func.__name__))
        res = func(*args, **kwargs)
        print('{} ends!'.format(func.__name__))
        return res
    return wrapper


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def join(c, e, val='_'): return val.join((c, e))


def shuffle_data(data, data_size, seed):
    data = np.array(data)
    shuffle_indices = np.random.RandomState(seed=seed).permutation(np.arange(data_size))
    return data[shuffle_indices]


def generate_batches(data, batch_size, shuffle=True, seed=None):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the graph at each epoch
    if shuffle:
        shuffled_data = shuffle_data(data, data_size, seed)
        #  shuffle_indices = np.random.permutation(np.arange(data_size))
        #  shuffled_data = graph[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


class WordCounter(dict):
    def __init__(self):
        dict.__init__(self)

    def __augment__(self, w):
        if w not in self:
            self[w] = 1
        else:
            self[w] += 1

    def add(self, item):
        assert isinstance(item, (list, str))
        if isinstance(item, list):
            for w in item:
                self.__augment__(w)
        else:
            self.__augment__(item)


class CausalVector(object):
    def __init__(self, cause_vec_path, effect_vec_path, norm=False):
        self.causeVec, self.effectVec = WordVec(cause_vec_path, norm), WordVec(effect_vec_path, norm)
        assert self.causeVec.embedding_size == self.effectVec.embedding_size
        self.dim = self.causeVec.embedding_size

    def __contains__(self, item):
        assert len(item) == 2
        return item[0] in self.causeVec and item[1] in self.effectVec

    def __getitem__(self, item):
        assert len(item) == 2
        return np.dot(self.causeVec[item[0]], self.effectVec[item[1]])

    def cause_contain(self, w):
        return w in self.causeVec

    def effect_contain(self, w):
        return w in self.effectVec

    def get_cause(self, w):
        return self.causeVec[w]

    def get_effect(self, w):
        return self.effectVec[w]


class WordVec(object):
    def __contains__(self, item):
        return item in self.indices

    def __getitem__(self, item):
        if item not in self.indices:
            raise KeyError
        return self.embeddings[self.indices[item]]

    def _normalize_vec_(self, vec):
        l2 = np.sqrt(sum(np.square(vec)))
        if l2 != 0.0:
            vec /= l2
            return True, vec
        return False, None

    def __init__(self, embedding_path, filter_vocabs=None, normalized=False):
        print('load embedding: {} ...'.format(embedding_path))
        self.words, self.embeddings = [], []
        fin = codecs.open(embedding_path, 'r', 'utf-8')
        line = fin.readline()
        res = line.strip().split(' ')
        vocab_size, self.embedding_size = list(map(int, res))
        line = fin.readline()
        while line:
            r = line.strip().split(' ')
            if len(r) == self.embedding_size + 1:
                w, vec = r[0], [float(r[i]) for i in range(1, len(r))]
                if filter_vocabs is not None:
                    if w in filter_vocabs:
                        if normalized:
                            tag, vec = self._normalize_vec_(vec)
                            if tag:
                                self.embeddings.append(vec)
                                self.words.append(w)
                        else:
                            self.embeddings.append(vec)
                            self.words.append(w)
                else:
                    if normalized:
                        tag, vec = self._normalize_vec_(vec)
                        if tag:
                            self.embeddings.append(vec)
                            self.words.append(w)
                    else:
                        self.embeddings.append(vec)
                        self.words.append(w)
            line = fin.readline()

        self.vocab_size = len(self.words)
        self.embeddings = np.array(self.embeddings, dtype=np.float32)
        self.indices = {w: i for i, w in enumerate(self.words)}
        print('finished loading embeddings {}.'.format(embedding_path))

    @staticmethod
    def __get_score__(v1, v2, norm):
        score = v1.dot(v2)
        return score/(np.sqrt(np.sum(np.square(v1)))*np.sqrt(np.sum(np.square(v2)))) if norm else score

    # def __per_process__(self, queue, target, start, end, norm):
    #     d = {}
    #     for i in range(start, end):
    #         score = self.__get_score__(target, self.embeddings[i], norm)
    #         if score < 1.0:
    #             d[self.words[i]] = score
    #     res = sorted(d.items(), key=lambda item: item[1], reverse=True)
    #     queue.put(res[0])
    #
    # def __find_sim_by_multi_process__(self, v, num_threads, norm):
    #     try:
    #         assert len(v) == self.embedding_size
    #         thread_list, queue = [], multiprocessing.Queue()
    #         num_vec_per_thread = int((self.vocab_size - 1) / num_threads) + 1
    #         for i in range(num_threads):
    #             s, e = i*num_vec_per_thread, min((i+1)*num_vec_per_thread, self.vocab_size)
    #             thread = multiprocessing.Process(target=self.__per_process__, args=(queue, v, s, e, norm))
    #             thread.start()
    #             thread_list.append(thread)
    #         for th in thread_list:
    #             th.join()
    #
    #         result = []
    #         while not queue.empty():
    #             result.append(queue.get())
    #         res = sorted(result, key=lambda item: item[1], reverse=True)
    #         return res[0][0]
    #     except AssertionError:
    #         print('embedding size should be {}.!'.format(self.vocab_size))
    #         exit(0)
    #
    # def __find_sim_by_single_process__(self, v, norm):
    #     d = {}
    #     for w in self.indices:
    #         score = self.__get_score__(v, self.embeddings[self.indices[w]], norm)
    #         if score < 1.0:
    #             d[w] = score
    #     # d = {w: self.__get_score__(v, self.embeddings[self.indices[w]], norm) for w in self.indices}
    #     res = sorted(d.items(), key=lambda item: item[1], reverse=True)
    #     return res[0][0]
    #
    # def find_sim(self, x, num_threads, norm):
    #     if isinstance(x, str):
    #         try:
    #             if num_threads == 1:
    #                 return self.__find_sim_by_single_process__(self.embeddings[self.indices[x]], norm)
    #             else:
    #                 return self.__find_sim_by_multi_process__(self.embeddings[self.indices[x]], num_threads, norm)
    #         except KeyError:
    #             raise KeyError('key: {} not in vocab!.'.format(x))
    #     else:
    #         try:
    #             assert isinstance(x, np.ndarray)
    #             if num_threads == 1:
    #                 return self.__find_sim_by_single_process__(x, norm)
    #             else:
    #                 return self.__find_sim_by_multi_process__(x, num_threads, norm)
    #         except AssertionError:
    #             raise KeyError('please give input which has type of str or np.ndarray!'.format(x))
    #
