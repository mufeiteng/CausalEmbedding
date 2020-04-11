from tools import *
import sys
import json
from tqdm import tqdm


def sigmoid(x): return 1.0/(1.0+np.exp(-x))


def filter_words(words, postags):
    new_words, new_postags = [], []
    for i in range(len(words)):
        if postags[i] in {'v', 'a', 'n', 'i'}:
            new_words.append(words[i])
            new_postags.append(postags[i])
    return new_words, new_postags


def get_pos_wp(path):
    global causalVectors
    response, d = [], dict()
    fin = codecs.open(path, 'r', 'utf-8')
    lines = fin.readlines()
    for line in tqdm(lines):
        res = line.strip().split('----')
        if res[-1] in {'使得', '使'}:
            continue
        arg1_words, arg2_words = res[1].split(' '), res[2].split(' ')
        arg1_postag, arg2_postag = res[3].split(' '), res[4].split(' ')
        assert len(arg1_words) == len(arg1_postag)
        assert len(arg2_words) == len(arg2_postag)
        if len(arg1_words) > 4 or len(arg2_words) > 5:
            continue
        arg1, _ = filter_words(arg1_words, arg1_postag)
        arg2, _ = filter_words(arg2_words, arg2_postag)
        arg1 = [w for w in arg1 if causalVectors.cause_contain(w)]
        arg2 = [w for w in arg2 if causalVectors.effect_contain(w)]
        if len(arg1) == 0 or len(arg2) == 0:
            continue
        max_score, wp = 0, None
        for w1 in arg1:
            for w2 in arg2:
                if w1 == w2:
                    continue
                score = sigmoid(causalVectors[(w1, w2)])
                if score < 0.3:
                    continue
                if max_score < score:
                    max_score, wp = score, (w1, w2)
        if wp is not None:
            k = join(wp[0], wp[1])
            if k not in d:
                d[k] = {'count': 1, 'score': max_score}
            else:
                d[k]['count'] += 1
    for k in d:
        response.append((k, d[k]['score'], d[k]['count']))
    return response


if __name__ == '__main__':
    params = {
        'cause_vec_path': os.path.join(project_source_path, 'boostvec/zh/bk_max_cause_31.txt'),
        'effect_vec_path': os.path.join(project_source_path, 'boostvec/zh/bk_max_effect_31.txt'),
        'threshold': {
            'wp': 5,
            'pa': 1
        },
        'pos_pp_path': os.path.join(project_source_path, 'boostvec/zh/bk_verb_positives.txt'),
        'pos_wp_path': os.path.join(project_source_path, 'boostvec/zh/bk_sorted_pos_wp.txt'),
        'num_threads': 16,
    }
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            key = arg.split("=")[0][2:]
            val = arg.split("=")[1]
            params[key] = val
    causalVectors = CausalVector(params['cause_vec_path'], params['effect_vec_path'])
    pairs = get_pos_wp(params['pos_pp_path'])
    fout = codecs.open(params['pos_wp_path'], 'w', 'utf-8')
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    for (k, score, count) in sorted_pairs:
        fout.write('{} {} {}\n'.format(k, score, count))

