from tools import *
import sys
import json
from tqdm import tqdm
from extraction.zh.cueset import global_cn_punctuation


def sigmoid(x): return 1.0/(1.0+np.exp(-x))


def filter_words(words, postags):
    new_words, new_postags = [], []
    for i in range(len(words)):
        if postags[i] in global_useful_postag and words[i] not in global_cn_punctuation:
            new_words.append(words[i])
            new_postags.append(postags[i])
    return new_words, new_postags


def get_pos_wp():
    response, d = [], dict()
    fin = codecs.open(params['pos_pp_path'], 'r', 'utf-8')
    lines = fin.readlines()
    for line in tqdm(lines):
        res = line.strip().split('----')
        arg1_words, arg2_words = res[0].split(' '), res[1].split(' ')
        arg1 = [w for w in arg1_words if causalVectors.cause_contain(w)]
        arg2 = [w for w in arg2_words if causalVectors.effect_contain(w)]
        if len(arg1) == 0 or len(arg2) == 0:
            continue
        max_score, wp = 0, None
        for w1 in arg1:
            for w2 in arg2:
                score = sigmoid(causalVectors[(w1, w2)])
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
        'cause_vec_path': os.path.join(project_source_path, 'boostvec/en/fl_max_cause_11.txt'),
        'effect_vec_path': os.path.join(project_source_path, 'boostvec/en/fl_max_effect_11.txt'),
        'threshold': {
            'wp': 5,
            'pa': 1
        },
        'pos_pp_path': os.path.join(project_source_path, 'boostvec/en/sharp_data.txt'),
        'pos_wp_path': os.path.join(project_source_path, 'boostvec/en/en_sharp_sorted_pos_wp.txt'),
    }
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            key = arg.split("=")[0][2:]
            val = arg.split("=")[1]
            params[key] = val
    causalVectors = CausalVector(params['cause_vec_path'], params['effect_vec_path'])
    pairs = get_pos_wp()
    fout = codecs.open(params['pos_wp_path'], 'w', 'utf-8')
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    for (k, score, count) in sorted_pairs:
        fout.write('{} {} {}\n'.format(k, score, count))

