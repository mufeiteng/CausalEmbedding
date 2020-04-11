from tools import *
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import auc
import gc
import sys
from time import time
import tensorflow as tf


def get_pr_points(sorted_score, relevant_label):
    """
    :param sorted_score: item in it is like: (score, label)
    :param relevant_label:
    :return:
    """
    numPair = len(sorted_score)
    assert numPair > 0
    numRelevant = sum([1 for s in sorted_score if s[1] == relevant_label])
    curvePoints = []
    scores = sorted(list(set([s[0] for s in sorted_score])), reverse=True)
    groupedByScore = [(s, [item for item in sorted_score if item[0] == s]) for s in scores]
    for i in range(len(groupedByScore)):
        score, items = groupedByScore[i]
        numRelevantInGroup = sum([1 for item in items if item[1] == relevant_label])
        if numRelevantInGroup > 0:
            sliceGroup = groupedByScore[:i + 1]
            items_slice = [x for y in sliceGroup for x in y[1]]
            numRelevantInSlice = sum([1 for s in items_slice if s[1] == relevant_label])
            sliceRecall = numRelevantInSlice / float(numRelevant)
            slicePrecision = numRelevantInSlice / float(len(items_slice))
            curvePoints.append((sliceRecall, slicePrecision))
    return curvePoints


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def multi_core_load_counter(filename, startP, endP, wp_thres, pa_thres):
    global causalVectors, global_pa_set
    with open(filename, 'rb') as f:
        occur_map = dict()
        f.seek(startP)
        if startP != 0:
            f.readline()
        line = f.readline().decode('utf-8')
        while line:
            current_val = dict()
            results = line.strip().split('----')
            try:
                assert len(results) == 2
                w1, val = results[0], json.loads(results[1])
                # if causalVectors.cause_contain(w1):
                for w2 in val:
                    # if not causalVectors.effect_contain(w2):
                    #     continue
                    d = val[w2]
                    if not (wp_thres[0] <= d['wp'] <= wp_thres[1]):  # 词对在语料中通过动词共现的次数
                        continue
                    temp_dict, pa_dict = {'wp': d['wp']}, d['pa']
                    new_pa = {pa: pa_dict[pa] for pa in pa_dict if pa_dict[pa] >= pa_thres and pa in global_pa_set}
                    if len(new_pa) == 0:
                        continue
                    temp_dict['pa'] = new_pa
                    current_val[w2] = temp_dict
                if len(current_val) > 0:
                    occur_map[w1] = current_val
            except Exception:
                pass
            if f.tell() >= endP:
                break
            line = f.readline().decode('utf-8')
    return occur_map


def load_counter(params):
    filename, thres = params['neg_counter'], params['threshold']
    wp_thres, pa_thres = thres['wp'], thres['pa']
    # workers = params['num_threads']
    # pool = multiprocessing.Pool()
    # filesize = os.path.getsize(filename)
    # results = []
    # for i in range(workers):
    #     s, e = (filesize * i) // workers, (filesize * (i + 1)) // workers
    #     results.append(pool.apply_async(
    #         multi_core_load_counter, (filename, s, e, wp_thres, pa_thres)
    #     ))
    # pool.close()
    # pool.join()
    # global_occur_map = dict()
    # for result in results:
    #     d = result.get()
    #     global_occur_map.update(d)
    #     del d
    # gc.collect()
    # global causalVectors
    global global_pa_set
    with open(filename, 'rb') as f:
        occur_map = dict()
        line = f.readline().decode('utf-8')
        while line:
            current_val = dict()
            results = line.strip().split('----')
            try:
                assert len(results) == 2
                w1, val = results[0], json.loads(results[1])
                # if causalVectors.cause_contain(w1):
                for w2 in val:
                    # if not causalVectors.effect_contain(w2):
                    #     continue
                    d = val[w2]
                    if not (wp_thres[0] <= d['wp'] <= wp_thres[1]):  # 词对在语料中通过动词共现的次数
                        continue
                    temp_dict, pa_dict = {'wp': d['wp']}, d['pa']
                    new_pa = {pa: pa_dict[pa] for pa in pa_dict if pa_dict[pa] >= pa_thres and pa in global_pa_set}
                    if len(new_pa) == 0:
                        continue
                    temp_dict['pa'] = new_pa
                    current_val[w2] = temp_dict
                if len(current_val) > 0:
                    occur_map[w1] = current_val
            except Exception:
                pass
            line = f.readline().decode('utf-8')
    return occur_map


def get_neg_samples(thres):
    global all_wp_sparse_counter, w1_indices, w2_indices, valid_pos_dict, valid_pa_set
    response, count = [], 0
    for _w1 in all_wp_sparse_counter:
        for _w2 in all_wp_sparse_counter[_w1]:
            if _w1 in valid_pos_dict and _w2 in valid_pos_dict[_w1]:
                continue
            _pa_dict = all_wp_sparse_counter[_w1][_w2]['pa']
            if len(set(_pa_dict.keys()) & valid_pa_set) == 0:
                continue
            # score = all_wp_sparse_counter[w1][w2]['score']
            # if low <= sigmoid(score) <= high:
            # print("Negative", w1, w2)
            # cnt = all_wp_sparse_counter[w1][w2]['wp']
            # if 5 <= cnt < 500:
            # if 3 <= cnt <= 8:
            response.append([w1_indices[_w1], w2_indices[_w2], 0])
            count += 1
    print('number of neg word pair is {}.'.format(count))
    return shuffle_data(np.array(response)), count


def get_valid_wp_list_and_pattern(wp_path, wp_score_thres, wp_count_thres, v_wp_count, pa_dict_len):
    _selected_pos_dict = dict()
    # 加载数据
    low, high = wp_count_thres
    fin = codecs.open(wp_path, 'r', 'utf-8')
    lines = fin.readlines()
    for line in lines:
        try:
            wp, score, count = line.strip().split(' ')
            _w1, _w2 = wp.split('_')
            if _w1 in {'原因', '缘由'} or _w2 in {'结果', '后果'}:
                continue
            if len(_w1) < 2 or len(_w2) < 2:
                continue
            if low <= float(count) <= high and float(score) >= wp_score_thres and _w1 != _w2:
                if _w1 in all_wp_sparse_counter and _w2 in all_wp_sparse_counter[_w1]:
                    if _w1 not in _selected_pos_dict:
                        _selected_pos_dict[_w1] = {_w2}
                    else:
                        _selected_pos_dict[_w1].add(_w2)
        except Exception:
            pass
    # 过滤(<a,b><b,a>)和不连接正样本的pattern
    _pa_counter_dict = dict()
    filtered_wp_list = []
    for _w1 in _selected_pos_dict:
        for _w2 in _selected_pos_dict[_w1]:
            if _w2 in _selected_pos_dict and _w1 in _selected_pos_dict[_w2]:
                continue
            filtered_wp_list.append((_w1, _w2))
            pa_dict = all_wp_sparse_counter[_w1][_w2]['pa']
            for pa in pa_dict:
                c = pa_dict[pa]
                if pa not in _pa_counter_dict:
                    _pa_counter_dict[pa] = c
                else:
                    _pa_counter_dict[pa] += c
    _valid_pa_set = set()
    for k in _pa_counter_dict:
        if _pa_counter_dict[k] > v_wp_count:
            _valid_pa_set.add(k)
    final_wp_list = []
    final_pos_dict = dict()
    for _w1, _w2 in filtered_wp_list:
        pa_dict = all_wp_sparse_counter[_w1][_w2]['pa']
        if len(set(pa_dict.keys()) & _valid_pa_set) > 0:
            if pa_dict_len[0] < len(pa_dict) < pa_dict_len[1]:
                final_wp_list.append([w1_indices[_w1], w2_indices[_w2], 1])
                if _w1 not in final_pos_dict:
                    final_pos_dict[_w1] = {_w2}
                else:
                    final_pos_dict[_w1].add(_w2)
    return final_wp_list, final_pos_dict, _valid_pa_set


def load_test_data(test_path):
    global causalVectors, all_wp_sparse_counter
    _test_samples = []
    count = 0
    lines = codecs.open(test_path, 'r', 'utf-8').readlines()
    for line in lines:
        res = line.strip().split('##')
        phrase_pair = res[0].split('----')
        arg1, arg2 = phrase_pair[1].split(' '), phrase_pair[2].split(' ')
        annotate = res[1].split(' ')
        if annotate[0] in valid_pos_dict and annotate[1] in valid_pos_dict[annotate[0]]:
            count += 1
            # continue
        if annotate[0] in all_wp_sparse_counter:
            if annotate[1] in all_wp_sparse_counter[annotate[0]]:
                _test_samples.append((arg1, arg2, annotate))
            else:
                # print("effect not found:", annotate)
                pass
        else:
            # print("cause not found:", annotate)
            pass

    acc, mrr, total = 0, [], 0
    print('numbers of initial test set {}.'.format(len(_test_samples)))
    print('numbers of initial test set in positives {}.'.format(count))

    for (arg1, arg2, (target_c, target_e)) in _test_samples:
        if (target_c, target_e) not in causalVectors:
            continue
        total += 1
        scores = {join(c, e): causalVectors[(c, e)] for c in arg1 for e in arg2 if (c, e) in causalVectors}
        res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        target = join(target_c, target_e)
        if res[0][0] == target:
            acc += 1
        for i, [k, _] in enumerate(res):
            if k == target:
                mrr.append(1.0 / (i + 1))
    assert len(mrr) == total
    try:
        initial_acc, initial_mrr = acc / float(total), sum(mrr) / float(len(mrr))
    except ZeroDivisionError:
        initial_acc, initial_mrr = 0.0, 0.0
    print('criteria of initial set is acc {}, mrr {}, validate {}.\n'.format(
        initial_acc, initial_mrr, total)
    )
    return _test_samples


def complete(_batch, workers=8):
    results = []
    global all_wp_sparse_counter, w1_list, w2_list, pa_indices
    for (_w1, _w2, _label) in _batch:
        _frequency = np.zeros(shape=[pa_len])
        pa_dict = all_wp_sparse_counter[w1_list[_w1]][w2_list[_w2]]['pa']
        for pa in pa_dict:
            if pa in valid_pa_set:
                _frequency[pa_indices[pa]] = pa_dict[pa]
            # _frequency[pa_indices[pa]] = 1. if pa_dict[pa] > 0 else 0.
        if np.sum(_frequency) > 0:
            results.append((_w1, _w2, _frequency, _label))
    # assert len(results) == len(_batch)
    return results


def sampling_negatives(data, number):
    _res, L = [], len(data)
    for i in range(number):
        k = np.random.randint(0, L)
        _res.append(data[k])
    return _res


def shuffle_data(data):
    data_size = len(data)
    shuffle_indices = np.random.permutation(np.arange(data_size))
    return data[shuffle_indices]


def get_global_pa_set():
    def judge(char):
        if len(char) == 1 and char not in {'让', '令', '使', '致'}:
            return True
        return False
    _set = set()
    thres = parameters['threshold']['v_count_thres']
    path = parameters['pattern_set_path']
    lines = codecs.open(path, 'r', 'utf-8')
    for line in lines:
        pattern, count = line.strip().split(' ')
        if thres[0] < float(count) < thres[1] and not judge(pattern.split('_')[0]):
            _set.add(pattern)
    return _set


class Trainer(object):
    @staticmethod
    def generate_batches(data, batch_size, shuffle=True):
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

    def __init__(self, params):
        self.graph = tf.Graph()
        with self.graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            self.sess = tf.Session(config=session_conf)
            with tf.variable_scope('calculator_scope'):
                self.tune_pa_embed = tf.get_variable(
                    name='tune_pa_embed', trainable=True, shape=[1, pa_len],
                    # initializer=tf.zeros_initializer()
                    initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01, seed=params['seed'], dtype=tf.float32)
                )
                self.global_steps = tf.get_variable(name='global_steps', initializer=0, trainable=False)
            self.w1 = tf.placeholder(dtype=tf.int32, shape=[None, ])
            self.w2 = tf.placeholder(dtype=tf.int32, shape=[None, ])
            self.pattern = tf.placeholder(dtype=tf.float32, shape=[None, pa_len])
            self.label = tf.placeholder(dtype=tf.float32, shape=[None, ])
            self.alpha = tf.placeholder(dtype=tf.float32)
            self.gamma = tf.placeholder(dtype=tf.float32)

            # pa_logits = tf.squeeze(tf.matmul(self.pattern, self.tune_pa_embed)) / tf.reduce_sum(self.pattern, axis=1)
            # pos_pa_logits = tf.reduce_max(self.tune_pa_embed * self.pattern, axis=1) #* tf.squeeze(tf.matmul(self.pattern, self.tune_pa_embed)) / tf.reduce_sum(self.pattern, axis=1)

            pa_logits = tf.reduce_sum(self.tune_pa_embed * self.pattern, axis=1) / tf.reduce_sum(self.pattern, axis=1)
            # pa_logits = tf.reduce_sum(self.tune_pa_embed * self.pattern, axis=1) / tf.reduce_sum(self.pattern, axis=1)
            pa_prob = tf.clip_by_value(tf.nn.sigmoid(pa_logits), 1e-8, 1.0 - 1e-8)
            pos_pa_prob = pa_prob * self.label
            neg_pa_prob = pa_prob * (1 - self.label)
            _, top_k_pos_indice = tf.nn.top_k(pos_pa_prob, k=tf.to_int32(0.7 * tf.reduce_sum(self.label)))
            _, top_k_neg_indice = tf.nn.top_k((1 - pa_prob) * (1 - self.label), k=tf.to_int32(tf.reduce_sum(1 - self.label)))

            top_k_pos_pa_prob = tf.gather(pos_pa_prob, top_k_pos_indice)
            top_k_neg_pa_prob = tf.gather(neg_pa_prob, top_k_neg_indice)

            pos_loss = tf.reduce_sum(- tf.log(top_k_pos_pa_prob))
            neg_loss = tf.reduce_sum(- tf.log(1 - top_k_neg_pa_prob))

            reg_loss = 1e-3 * tf.reduce_sum(self.tune_pa_embed * self.tune_pa_embed)
            self.loss = pos_loss + neg_loss + reg_loss
            optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])
            gradients = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(
                gradients, global_step=self.global_steps
            )
            self.sess.run(tf.global_variables_initializer())

    def get_feed_dict(self, _w1, _w2, _pa, _label):
        _dict = {
            self.w1: _w1,
            self.w2: _w2,
            self.pattern: _pa,
            self.label: _label,
            self.alpha: parameters['alpha'],
            self.gamma: parameters['gamma'],
        }
        return _dict

    def save(self, saved_path, prefix, epoch, extract_pp_dir):
        with self.sess.as_default():
            pattern_weight = np.squeeze(self.tune_pa_embed.eval())
            # write pattern weight
            pattern_output = codecs.open(os.path.join(saved_path, '{}_{}.txt'.format(prefix, epoch)), 'w', 'utf-8')
            pattern_items = zip(pa_list, pattern_weight)
            res = sorted(pattern_items, key=lambda x: x[1], reverse=True)
            for item in res:
                pattern_output.write('{} {}\n'.format(item[0], item[1]))

            # # pa_list, pattern_weight, pa_indices
            # pa_prob_dict = calculate_pa_prob(extract_pp_dir, parameters['num_threads'], pattern_weight)
            # sorted_pa_prob_dict = sorted(pa_prob_dict.items(), key=lambda x: x[1], reverse=True)
            # pa_prob_output = codecs.open(os.path.join(saved_path, 'en_pattern_prob_{}.txt'.format(epoch)), 'w', 'utf-8')
            # for (pa, prob) in sorted_pa_prob_dict:
            #     pa_prob_output.write('{} {}\n'.format(pa, prob))

    def eval(self):
        global all_wp_sparse_counter, pa_len, pa_indices
        with self.sess.as_default(), self.graph.as_default():
            pattern_embed = np.squeeze(self.tune_pa_embed.eval())
            pa_total, pa_acc, pa_mrr = 0, 0, []
            for (arg1, arg2, (target_c, target_e)) in test_samples:
                if not (target_c in all_wp_sparse_counter and target_e in all_wp_sparse_counter[target_c]):
                    continue
                scores, target = dict(), join(target_c, target_e)
                for c in arg1:
                    for e in arg2:
                        if not (c in all_wp_sparse_counter and e in all_wp_sparse_counter[c]):
                            continue
                        pa_dict = all_wp_sparse_counter[c][e]['pa']
                        if len(set(pa_dict.keys()) & valid_pa_set) == 0:
                            continue
                        bitmap = np.zeros([pa_len], dtype=np.float32)
                        for pa in pa_dict:
                            if pa in valid_pa_set:
                                # bitmap[pa_indices[pa]] = pa_dict[pa]
                                bitmap[pa_indices[pa]] = 1. if pa_dict[pa] > 0 else 0.
                        assert np.sum(bitmap) != 0
                        scores[join(c, e)] = np.sum(bitmap * pattern_embed)
                res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                if len(res) == 0:
                    continue
                if res[0][0] == target:
                    pa_acc += 1
                for indices, [key, _] in enumerate(res):
                    if key == target:
                        pa_mrr.append(1.0 / (indices + 1))
                pa_total += 1
            print('criteria calculated by PATTERN WEIGHTS:')
            print('total {}, acc {}, mrr {}.'.format(pa_total, pa_acc / float(pa_total), sum(pa_mrr) / float(len(pa_mrr))))

    def run(self, params):
        print('\n\n======initial criteria:')
        self.eval()
        print('======')
        for current_epoch in range(params['num_epochs']):
            ave_loss, count = 0.0, 0
            start_time = time()
            _batches = self.generate_batches(neg_wp, params['batch_size'])
            for _batch in _batches:
                sampled_batch = sampling_negatives(pos_wp_list, len(_batch))
                concat_batch = np.concatenate([_batch, sampled_batch], axis=0)
                response = complete(concat_batch, workers=8)  # params['num_threads']
                _w1, _w2, _pa, _label = zip(*response)
                feed_dict = self.get_feed_dict(_w1, _w2, _pa, _label)
                _, _loss, steps = self.sess.run([self.train_op, self.loss, self.global_steps], feed_dict)
                count += 1
                ave_loss += _loss
            print('Average loss at epoch {} is {}!'.format(current_epoch+1, ave_loss))
            self.eval()
            self.save(
                saved_path=params['saved_path'], epoch=current_epoch+1,
                prefix=parameters['saved_pattern_prefix'], extract_pp_dir=parameters['extracted_pp_dir']
            )
            end_time = time()
            print('epoch: {} uses {} mins.\n'.format(current_epoch + 1, float(end_time - start_time) / 60))
            # if current_epoch > 0 and current_epoch % 2 == 0:


if __name__ == '__main__':
    parameters = {
        'cause_vec_path': os.path.join(project_source_path, 'cesim/sg_max_cause_41.txt'),
        'effect_vec_path': os.path.join(project_source_path, 'cesim/sg_max_effect_41.txt'),
        'extracted_pp_dir': os.path.join(project_source_path, 'boostvec/zh/tutorial/sg_allverb_negatives.txt'),
        'pattern_set_path': os.path.join(project_source_path, 'boostvec/zh/tutorial/sg_dual_pattern_set.txt'),
        'neg_counter': os.path.join(project_source_path, 'boostvec/zh/tutorial/sg_wp_dual_counter.txt'),
        'pos_wp_path': os.path.join(project_source_path, 'boostvec/zh/tutorial/sg_sorted_pos_wp.txt'),
        'test_path': os.path.join(project_source_path, 'cross/bk_eva.txt'),
        'saved_path': os.path.join(project_source_path, 'boostvec/zh/tutorial/models/pattern/'),
        'saved_pattern_prefix': 'bk_pattern_weights_rev',
        'threshold': {
            'wp': [4, 50],
            'pos_wp_score': 0.5663,
            'pos_wp_count': [3, 120],
            'neg_wp_score': [-5.0, 0.5],
            'pa': 1,
            'v_count_thres': [1000, 500000],
            'v_wp_count': 1,
        },
        'num_sample': 10,
        'num_threads': 15,
        'seed': 5,
        'num_epochs': 5,
        'batch_size': 4096,
        'show': 200,
        'lr': 1e-3,
        'alpha': 0.8,
        'gamma': 2.0,
    }
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            parameters[arg.split("=")[0][2:]] = arg.split("=")[1]
    print('=========\nprepare data.....')

    # 加载因果向量库
    causalVectors = CausalVector(parameters['cause_vec_path'], parameters['effect_vec_path'])
    global_pa_set = get_global_pa_set()
    print('global pa set len is {}.'.format(len(global_pa_set)))
    # 加载负样本
    print('load counter...')
    all_wp_sparse_counter = load_counter(parameters)
    # 获取词表
    _w1_set, _w2_set = set(), set()
    for w1 in all_wp_sparse_counter:
        _w1_set.add(w1)
        for w2 in all_wp_sparse_counter[w1]:
            _w2_set.add(w2)
    w1_list, w2_list = list(_w1_set), list(_w2_set)
    w1_indices = {w: i for i, w in enumerate(w1_list)}
    w2_indices = {w: i for i, w in enumerate(w2_list)}
    print('finished!')
    # # 加载正样本
    pos_wp_list, valid_pos_dict, valid_pa_set = get_valid_wp_list_and_pattern(
        wp_path=parameters['pos_wp_path'], wp_score_thres=parameters['threshold']['pos_wp_score'],
        wp_count_thres=parameters['threshold']['pos_wp_count'], v_wp_count=parameters['threshold']['v_wp_count'],
        pa_dict_len=[1, 10]
    )
    print('number of pos wp is {}.'.format(len(pos_wp_list)))
    pa_list = list(valid_pa_set)
    w1_len, w2_len, pa_len = len(w1_list), len(w2_list), len(pa_list)
    pa_indices = {w: i for i, w in enumerate(pa_list)}
    print('vocab length: w1 {}, w2 {}, pa {}.'.format(w1_len, w2_len, pa_len))
    # 加载测试数据
    test_samples = load_test_data(parameters['test_path'])

    # 从负样本中去除正样本
    neg_wp, neg_number = get_neg_samples(parameters['threshold']['neg_wp_score'])

    print('finished!\n=========\n\nstart training...')
    Trainer(parameters).run(parameters)
