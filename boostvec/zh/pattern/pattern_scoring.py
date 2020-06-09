from tools import *
import json
import numpy as np
import gc
import sys
from tqdm import tqdm


def sigmoid(x): return 1.0/(1.0+np.exp(-x))


def filter_words(words, postags):
    new_words, new_postags = [], []
    for i in range(len(words)):
        if postags[i] in global_useful_postag:
            new_words.append(words[i])
            new_postags.append(postags[i])
    return new_words, new_postags


def load_pattern(pattern_path, pa_thres):
    _keys, _scores = [], []
    lines = codecs.open(pattern_path, 'r', 'utf-8').readlines()
    for line in lines:
        k, v = line.strip().split(' ')
        # if float(v) > pa_thres:
        _keys.append(k)
        _scores.append(float(v))
    _indices = {w: i for i, w in enumerate(_keys)}
    return _keys, _scores, _indices


def multi_core_counter_loader(filename, start, end, wp_thres):
    global pa_keys, pa_scores, pa_indices
    final_dict = dict()
    with open(filename, 'rb') as f:
        f.seek(start)
        if start != 0:
            f.readline()
        line = f.readline().decode('utf-8')
        while line:
            w1, s = line.strip().split('----')
            try:
                w2_dict = json.loads(s)
                temp_w1_dict = dict()
                for w2 in w2_dict:
                    c = w2_dict[w2]['wp']
                    if not (wp_thres[0] <= c <= wp_thres[1]):
                        continue
                    _pa_dict = w2_dict[w2]['pa']
                    # wp_score, total = 0.0, 0
                    # for pa in _pa_dict:
                    #     if pa in pa_indices:
                    #         total += 1
                    #         wp_score += pa_scores[pa_indices[pa]]
                    # if total > 0:
                    #     temp_w1_dict[w2] = sigmoid(wp_score / float(total))
                    wp_score, total = 0.0, 0
                    for pa in _pa_dict:
                        if pa in pa_indices:
                            wp_score += _pa_dict[pa] * pa_scores[pa_indices[pa]]
                            total += _pa_dict[pa]
                    if total > 0:
                        temp_w1_dict[w2] = sigmoid(wp_score / float(total))
                if len(temp_w1_dict) > 0:
                    final_dict[w1] = temp_w1_dict
            except:
                pass
            if f.tell() >= end:
                break
            line = f.readline().decode('utf-8')
    return final_dict


def load_wp_counter(counter_path, workers, wp_thres):
    pool = multiprocessing.Pool()
    filesize = os.path.getsize(counter_path)
    results = []
    for i in range(workers):
        s, e = (filesize * i) // workers, (filesize * (i + 1)) // workers
        results.append(pool.apply_async(
            multi_core_counter_loader, (counter_path, s, e, wp_thres)
        ))
    pool.close()
    pool.join()
    global_dict = dict()
    for result in results:
        temp_dict = result.get()
        global_dict.update(temp_dict)
    gc.collect()
    total = 0
    for w1 in global_dict:
        total += len(global_dict[w1])
    print('total number of word pair is {}.'.format(total))
    return global_dict


def multi_core_prob_calculator(filename, start, end):
    global pa_keys, pa_scores, pa_indices, wp_scores_dict
    prob_dict = dict()
    # exist_wp = dict()
    with open(filename, 'rb') as f:
        f.seek(start)
        if start != 0:
            f.readline()
        line = f.readline().decode('utf-8')
        while line:
            _res = line.strip().split('----')
            if len(_res) == 6 and _res[-1] in pa_indices:
                _arg1_words, _arg2_words = _res[1].split(' '), _res[2].split(' ')
                _arg1_pos, _arg2_pos, _pattern = _res[3].split(' '), _res[4].split(' '), _res[5]
                if len(_arg1_words) == len(_arg1_pos) and len(_arg2_words) == len(_arg2_pos):
                    # 正向
                    max_prob = 0.0
                    for w1 in _arg1_words:
                        if w1 not in wp_scores_dict:
                            continue
                        for w2 in _arg2_words:
                            if w2 not in wp_scores_dict[w1]:
                                continue
                            wp_prob = wp_scores_dict[w1][w2]
                            if max_prob < wp_prob:
                                max_prob = wp_prob
                    if _pattern not in prob_dict:
                        prob_dict[_pattern] = [max_prob, 1]
                    else:
                        prob_dict[_pattern][0] += max_prob
                        prob_dict[_pattern][1] += 1
                    # 反向
                    rev_pattern = _pattern+'_rev'
                    max_prob = 0
                    for w1 in _arg2_words:
                        if w1 not in wp_scores_dict:
                            continue
                        for w2 in _arg1_words:
                            if w2 not in wp_scores_dict[w1]:
                                continue
                            wp_prob = wp_scores_dict[w1][w2]
                            if max_prob < wp_prob:
                                max_prob = wp_prob
                    if rev_pattern not in prob_dict:
                        prob_dict[rev_pattern] = [max_prob, 1]
                    else:
                        prob_dict[rev_pattern][0] += max_prob
                        prob_dict[rev_pattern][1] += 1
            if f.tell() >= end:
                break
            line = f.readline().decode('utf-8')
    return prob_dict


def calculate_pa_prob(pp_dir, workers=14):
    # global pa_list, pa_indices, all_wp_sparse_counter
    global_prob_items = dict()
    files = {pp_dir}
    for file in tqdm(files):
        pool = multiprocessing.Pool()
        pp_path = file
        filesize = os.path.getsize(pp_path)
        results = []
        for i in range(workers):
            s, e = (filesize * i) // workers, (filesize * (i + 1)) // workers
            results.append(pool.apply_async(
                multi_core_prob_calculator, (pp_path, s, e)
            ))
        pool.close()
        pool.join()
        for result in results:
            d = result.get()
            for key in d:
                if key not in global_prob_items:
                    global_prob_items[key] = d[key]
                else:
                    global_prob_items[key][0] += d[key][0]
                    global_prob_items[key][1] += d[key][1]
            del d
            gc.collect()
    _response = dict()
    for key in global_prob_items:
        # l, s = len(global_prob_items[key]), sum(global_prob_items[key])
        value, count = global_prob_items[key][0], global_prob_items[key][1]
        _response[key] = value/float(count)
    return _response


def write_pa_prob_dict(d, output_path):
    fout = codecs.open(output_path, 'w', 'utf-8')
    _res = sorted(d.items(), key=lambda x: x[1], reverse=True)
    for (k, v) in _res:
        fout.write('{} {}\n'.format(k, v))


if __name__ == '__main__':
    path = os.path.join(project_source_path, 'boostvec/')
    parameters = {
        'counter_path': os.path.join(path, 'zh/bk_wp_dual_counter.txt'),
        'weak_pp_path': os.path.join(path, 'zh/bk_allverb_negatives.txt'),
        'pa_weight_path': os.path.join(path, 'zh/models/pattern/bk_pattern_weights_rev_2.txt'),
        'pa_prob_output_path': os.path.join(path, 'zh/models/pattern/bk_pa_prob_rev_2.txt'),
        'threshold': {
            'wp': [2, 100],
            'pa': 1,
            'pa_weight': 0.0,
        },
        'num_threads': 16,
    }
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            parameters[arg.split("=")[0][2:]] = arg.split("=")[1]
    print('=========\nprepare data.....')
    pa_keys, pa_scores, pa_indices = load_pattern(parameters['pa_weight_path'], parameters['threshold']['pa_weight'])
    pa_length = len(pa_keys)
    print('pattern set length: {}.'.format(pa_length))
    print('load counter...')
    wp_scores_dict = load_wp_counter(parameters['counter_path'], parameters['num_threads'], parameters['threshold']['wp'])
    print('calculate pattern prob...')
    pa_prob_dict = calculate_pa_prob(parameters['weak_pp_path'], parameters['num_threads'])
    write_pa_prob_dict(pa_prob_dict, parameters['pa_prob_output_path'])

