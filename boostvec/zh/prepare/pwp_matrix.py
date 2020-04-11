from tools import *
import sys
import json
import gc
from tqdm import tqdm


def write_pattern_counter(items, output_path):
    fout = codecs.open(output_path, 'w', 'utf-8')
    for (k, count) in items:
        fout.write('{}_rev {}\n'.format(k, count))
        fout.write('{} {}\n'.format(k, count))


def multicore_paset_geter(filename, startP, endP):
    _counter = WordCounter()
    with open(filename, 'rb') as f:
        f.seek(startP)
        if startP != 0:
            f.readline()
        line = f.readline().decode('utf-8')
        while line:
            results = line.strip().split('----')
            if len(results) == 6:
                _counter.add(results[-1])
            if f.tell() >= endP:
                break
            line = f.readline().decode('utf-8')
    return _counter


def get_pattern_set():
    global_pa_set = dict()
    v_pp_count, workers = params['threshold']['v_pp_count'], params['num_threads']
    path = params['extract_pp_dir']
    files = {path}
    print('get pattern set.\nprocess...')
    for name in tqdm(files):
        filename = name
        pool = multiprocessing.Pool()
        filesize = os.path.getsize(filename)
        results = []
        for i in range(workers):
            chunk_start, chunk_end = (filesize * i) // workers, (filesize * (i + 1)) // workers
            results.append(pool.apply_async(
                multicore_paset_geter, (filename, chunk_start, chunk_end)
            ))
        pool.close()
        pool.join()

        for result in results:
            d = result.get()
            for k in d:
                if k in global_pa_set:
                    global_pa_set[k] += d[k]
                else:
                    global_pa_set[k] = d[k]
    gc.collect()

    def valid_tag(k, v, thres):
        return thres[0] < v < thres[1] and k not in {'am', 'is', 'are', 'was', 'were', 'be', 'being', 'been', 'do', 'did', 'does', 'have', 'has'}

    sorted_pp_count = sorted(global_pa_set.items(), key=lambda x: x[1], reverse=True)
    write_pattern_counter(sorted_pp_count, params['pa_set_output_path'])
    print('还剩{}个pp.'.format(sum([global_pa_set[k] for k in global_pa_set if valid_tag(k, global_pa_set[k], v_pp_count)])))
    return {k for k in global_pa_set if valid_tag(k, global_pa_set[k], v_pp_count)}


def judge_content_word(p):
    if p in {'n', 'v', 'a', 'd', 'i'}:
        return True
    return False


def filter_words(words, postags):
    new_words, new_postags = [], []
    for i in range(len(words)):
        if judge_content_word(postags[i]):
            new_words.append(words[i])
            new_postags.append(postags[i])
    return new_words, new_postags


def calculate_counter():
    global_sparse_dict = dict()
    """
    -rw-r--r-- 1 aszzy aszzy 596M 1月  25 10:14 101_0_3_v3.txt
    -rw-r--r-- 1 aszzy aszzy 591M 1月  25 10:15 116_135_175_v3.txt
    -rw-rw-r-- 1 aszzy aszzy 1.6G 1月  25 10:11 121_225_330_v3.txt
    -rw-rw-r-- 1 aszzy aszzy 1.6G 1月  25 18:32 126_3_135_v3.txt
    -rw-rw-r-- 1 aszzy aszzy 728M 1月  25 10:15 61_175_225_v3.txt

    """
    used = {'121_225_330_v3.txt', '126_3_135_v3.txt', '116_135_175_v3.txt'}
    path = params['extract_pp_dir']
    files = {path}
    # valid = [file for file in files if file in used]  # if file in used

    print('calculate counter.\nprocess..')
    for file in tqdm(files):
        with open(file, 'rb') as f:
            line = f.readline().decode('utf-8')
            while line:
                results = line.strip().split('----')
                pattern = results[-1]
                if len(results) == 6 and pattern in global_pattern_sets:
                    left_words, left_postags = results[1].split(' '), results[3].split(' ')
                    right_words, right_postags = results[2].split(' '), results[4].split(' ')
                    if len(left_words) == len(left_postags) and len(right_words) == len(right_postags):
                        left_words, left_postags = filter_words(left_words, left_postags)
                        right_words, right_postags = filter_words(right_words, right_postags)
                        if len(left_words) > 0 and len(right_words) > 0:
                            # 正向算一遍
                            for c in left_words:
                                if c not in global_sparse_dict:
                                    global_sparse_dict[c] = dict()
                                for e in right_words:
                                    if e not in global_sparse_dict[c]:
                                        global_sparse_dict[c][e] = {'wp': 1, 'pa': dict()}
                                    else:
                                        global_sparse_dict[c][e]['wp'] += 1
                                    if pattern in global_sparse_dict[c][e]['pa']:
                                        global_sparse_dict[c][e]['pa'][pattern] += 1
                                    else:
                                        global_sparse_dict[c][e]['pa'][pattern] = 1
                            # 反向算一遍
                            pa = pattern + '_rev'
                            for e in right_words:
                                if e not in global_sparse_dict:
                                    global_sparse_dict[e] = dict()
                                for c in left_words:
                                    if c not in global_sparse_dict[e]:
                                        global_sparse_dict[e][c] = {'wp': 1, 'pa': dict()}
                                    else:
                                        global_sparse_dict[e][c]['wp'] += 1
                                    if pa in global_sparse_dict[e][c]['pa']:
                                        global_sparse_dict[e][c]['pa'][pa] += 1
                                    else:
                                        global_sparse_dict[e][c]['pa'][pa] = 1
                line = f.readline().decode('utf-8')
    return global_sparse_dict


def save_counter(d):
    print('save...')
    fout = codecs.open(params['counter_output_path'], 'w', 'utf-8')
    for w1 in d:
        s = json.dumps(d[w1])
        fout.write('{}----{}\n'.format(w1, s))


if __name__ == '__main__':
    params = {
        # 'cause_vec_path': os.path.join(project_source_path, 'boostvec/en/fl_max_cause_11.txt'),
        # 'effect_vec_path': os.path.join(project_source_path, 'boostvec/en/fl_max_effect_11.txt'),
        'threshold': {
            'wp': 1,  # 词对出现的次数
            'pa': 1,  # 词对被动词连接的次数
            'v_pp_count': [350, 500000]
        },
        'num_threads': 15,
        'extract_pp_dir': os.path.join(project_source_path, 'boostvec/zh/bk_allverb_negatives.txt'),
        'pa_set_output_path': os.path.join(project_source_path, 'boostvec/zh/bk_dual_pattern_set.txt'),
        'counter_output_path': os.path.join(project_source_path, 'boostvec/zh/bk_wp_dual_counter.txt'),
    }
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            key = arg.split("=")[0][2:]
            val = arg.split("=")[1]
            params[key] = val

    global_pattern_sets = get_pattern_set()
    print('number of patterns is {}.'.format(len(global_pattern_sets)))

    final_counter = calculate_counter()
    save_counter(final_counter)
    print('finished!')


