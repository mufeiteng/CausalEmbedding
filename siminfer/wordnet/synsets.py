from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from tools import *
import codecs
from collections import defaultdict
import json


def is_valid(pos):
    if pos.startswith('NN') or pos.startswith('RB'):
        return True
    if pos.startswith('VB') or pos.startswith('JJ'):
        return True
    return False


def func(text):
    items = text.split(' ')
    tokens, postags = [], []
    for item in items:
        try:
            text, postag = item.split('_')
            if not is_valid(postag):
                continue
            tokens.append(text)
            postags.append(postag)
        except:
            pass
    return tokens, postags


def load_weak_pattern(pa_prob_path, prob_thres):
    _score_dict = dict()
    fin = codecs.open(pa_prob_path, 'r', 'utf-8')
    lines = fin.readlines()
    for line in lines:
        _res = line.strip().split(' ')
        # assert len(_res) == 2
        # k, v = _res[0], float(_res[-1])
        k, v = ' '.join(_res[:-1]), float(_res[-1])
        if v > prob_thres and k.split('_')[0] not in {'cause', 'result', 'lead', 'create'}:
            _score_dict[k] = v
    _valid_dict = dict()
    for k in _score_dict:
        if k.endswith('_rev'):
            c2e, e2c = k[:-4], k
        else:
            c2e, e2c = k, k + '_rev'
        if c2e in _score_dict and e2c in _score_dict:
            used = c2e if _score_dict[c2e] > _score_dict[e2c] else e2c
            _valid_dict[used] = _score_dict[used]
        else:
            _valid_dict[k] = _score_dict[k]
    print('before filter: {}, after filter {}. followings are filtered:'.format(len(_score_dict), len(_valid_dict)))
    print(set(_score_dict.keys()) - set(_valid_dict.keys()))
    return _valid_dict


def filter_arg(words, postags):
    assert len(words) == len(postags)
    _valid_words = []
    for i in range(len(words)):
        pos = postags[i]
        if pos in {'NN', 'NNS'} or pos.startswith('VB') or pos.startswith('JJ') or pos.startswith('RB'):
            _valid_words.append((words[i], postags[i]))
    words, postags = list(zip(*_valid_words))
    return words, postags

def load_weak_phrase_pairs(phrase_pairs_path, pa_prob_dict):
    items = []
    fin = open(phrase_pairs_path, 'rb')
    for line in fin:
        try:
            line = line.decode('utf-8')
            res = line.strip().split('----')
            if len(res) != 6:
                continue
            pattern = res[-1]
            if pattern not in pa_prob_dict:
                continue
            arg1_words, arg2_words = res[1].lower().split(' '), res[2].lower().split(' ')
            arg1_pos, arg2_pos = res[3].split(' '), res[4].split(' ')
            if len(arg1_words) != len(arg1_pos) or len(arg2_words) != len(arg2_pos):
                continue
            arg1_words, arg1_pos = filter_arg(arg1_words, arg1_pos)
            arg2_words, arg2_pos = filter_arg(arg2_words, arg2_pos)
            if len(arg1_words) and len(arg2_words):
                items.append((arg1_words, arg2_words, arg1_pos, arg2_pos))
        except Exception as e:
            pass
    return items


parameters = {
    'pa_prob_path': os.path.join(project_source_path, 'causalembedding/siminfer/en/prob_of_patterns_rev_2.txt'),
    'strong_pp_path': os.path.join(project_source_path, 'causalembedding/siminfer/en/final_sharp_data_with_postag.txt'),
    'strong_w_count': 1,  # 强pattern的min-count
    'weak_w_count': 5,  # 弱pattern的min-count
    'pa_thres': 0.59,  # pattern的概率
    'weak_p_diff': False,
    'weak_p_offset': 0.2,
}
oov = '<pad>'
in_path = '/home/aszzy/Documents/sharp_data/causalTuples'
path = os.path.join(project_source_path, 'causalembedding/siminfer/en/')
# sharp_data = os.path.join(path, 'final_sharp_data_with_postag.txt')
# fout = codecs.open(sharp_data, 'w', 'utf-8')
# names = os.listdir(in_path)
# for name in names:
#     if not name.endswith('argsC'):
#         continue
#     fin = codecs.open(os.path.join(in_path, name), 'r', 'utf-8')
#     for line in fin:
#         causal = line.strip().split('\t-->\t')
#         causes = causal[0].split(',')
#         effects = causal[1].split(',')
#         for cause in causes:
#             cause_words, cause_postags = func(cause)
#             if not cause_words:
#                 continue
#             for effect in effects:
#                 effect_words, effect_postags = func(effect)
#                 if not effect_words:
#                     continue
#                 s = '----'.join([
#                     ' '.join(cause_words), ' '.join(effect_words),
#                     ' '.join(cause_postags), ' '.join(effect_postags)])
#                 fout.write('{}\n'.format(s))


weak_causal_pp_path = os.path.join(project_source_path, 'causalembedding/siminfer/en/weakcausalpp.txt')
pa_prob_dict = load_weak_pattern(
    parameters['pa_prob_path'], parameters['pa_thres']
)
print(len(pa_prob_dict))

weak_pp = load_weak_phrase_pairs(
    weak_causal_pp_path, pa_prob_dict
)
print(len(weak_pp))

sharp_data = os.path.join(path, 'final_sharp_data_with_postag.txt')
fin = codecs.open(sharp_data, 'r', 'utf-8')
for line in fin:
    res = line.strip().split('----')
    cause_words, effect_words = res[0].split(' '), res[1].split(' ')
    cause_postags, effect_postags = res[2].split(' '), res[3].split(' ')
    weak_pp.append((cause_words, effect_words, cause_postags, effect_postags))

print(len(weak_pp))

stemmer = WordNetLemmatizer()
def get_pos(pos):
    if pos.startswith('NN'):
        return 'n'
    # if pos.startswith('JJ'):
    #     return 'a'
    if pos.startswith('VB'):
        return 'v'
    return pos


words_sets = defaultdict(set)

for cause_words, effect_words, cause_postags, effect_postags in weak_pp:
    for i in range(len(cause_words)):
        pos = get_pos(cause_postags[i])
        if pos in {'n', 'v'}:
            lemm = stemmer.lemmatize(cause_words[i], pos='n')
            if lemm is None:
                continue
            words_sets[lemm].add(pos)
            words_sets[cause_words[i]].add(pos)
    for i in range(len(effect_words)):
        pos = get_pos(effect_postags[i])
        if pos in {'n', 'v'}:
            lemm = stemmer.lemmatize(effect_words[i], pos='n')
            if lemm is None:
                continue
            words_sets[lemm].add(pos)
            words_sets[effect_words[i]].add(pos)

print('the number of word is {}.'.format(len(words_sets)))


def get_synsets(word, d):
    res = dict()
    for pos in d[word]:
        for synset in wn.synsets(word, pos=pos):
            words = synset.lemma_names()
            for w in words:
                if w == word:
                    continue
                scores = []
                for s in wn.synsets(w):
                    score = synset.path_similarity(s)
                    if score is not None:
                        scores.append(score)
                if not scores:
                    continue
                # res[w] = sum(scores)/len(scores)
                res[w] = 1.0
    return res


synsets_resources = dict()
for word in words_sets:
    val = get_synsets(word, words_sets)
    if len(val) > 0:
        synsets_resources[word] = val

print('en: {}.'.format(len(synsets_resources)))
fout = codecs.open(os.path.join(path, 'wp_sim_use_wordnet_without_weight.txt'), 'w', 'utf-8')
for word in synsets_resources:
    fout.write('{}----{}\n'.format(word, json.dumps(synsets_resources[word])))
