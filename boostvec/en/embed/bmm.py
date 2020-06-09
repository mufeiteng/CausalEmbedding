import sys
from tools import *
import tensorflow as tf
from time import time
from tqdm import tqdm
import gc
from sklearn.metrics import auc


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


def filter_arg(words, postags):
    assert len(words) == len(postags)
    _valid_words = []
    for i in range(len(words)):
        pos = postags[i]
        if pos in {'NN', 'NNS'} or pos.startswith('VB') or pos.startswith('JJ') or pos.startswith('RB'):
            _valid_words.append(words[i])
    return _valid_words


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
            c2e, e2c = k, k+'_rev'
        if c2e in _score_dict and e2c in _score_dict:
            used = c2e if _score_dict[c2e] > _score_dict[e2c] else e2c
            _valid_dict[used] = _score_dict[used]
        else:
            _valid_dict[k] = _score_dict[k]
    print('before filter: {}, after filter {}. followings are filtered:'.format(len(_score_dict), len(_valid_dict)))
    print(set(_score_dict.keys())-set(_valid_dict.keys()))
    return _valid_dict


def load_strong_pp(pp_path, min_count):
    _left_counter, _right_counter = WordCounter(), WordCounter()
    fin = codecs.open(pp_path, 'r', 'utf-8')
    line = fin.readline().strip()
    _items = []
    while line:
        _res = line.split('----')
        if len(_res) == 2:
            _arg1_words, _arg2_words = _res[0].split(' '), _res[1].split(' ')
            _left_counter.add(_arg1_words)
            _right_counter.add(_arg2_words)
            _items.append((_arg1_words, _arg2_words))
        line = fin.readline().strip()
    _left_set = {k for k in _left_counter if _left_counter[k] >= min_count}
    _right_set = {k for k in _right_counter if _right_counter[k] >= min_count}
    _response = []
    for (arg1, arg2) in _items:
        _left = [w for w in arg1 if w in _left_set]
        _right = [w for w in arg2 if w in _right_set]
        if len(_left) > 0 and len(_right) > 0:
            _response.append((_left, _right, 1.0, 1.0))
    return _response, _left_set, _right_set


def convert2indices(left_rev, right_rev, pp_items, s):
    _response = []
    for (arg1, arg2, weight, label) in pp_items:
        left = [left_rev[w] for w in arg1]
        right = [right_rev[w] for w in arg2]
        if len(left) < s and len(right) < s:
            _response.append((left, right, weight, label))
    return _response


def complete(batch, left_rev, right_rev, s):
    _left_w, _right_w, _len1, _len2, _prob, _label = [], [], [], [], [], []
    for (left, right, prob, label) in batch:
        l1, l2 = len(left), len(right)
        _len1.append(l1)
        _len2.append(l2)
        # _prob.append(prob)
        _label.append(label)
        _left_w.append(left + [left_rev['pad']] * (s - l1))
        _right_w.append(right + [right_rev['pad']] * (s - l2))
    # return _left_w, _right_w, _len1, _len2, _prob, _label, len(_len1)
    return _left_w, _right_w, _len1, _len2, _label


def sample_negative(data, amount):
    L, _response = len(data), []
    for i in range(amount):
        k = np.random.randint(0, L)
        _left = data[k][0]
        j = np.random.randint(0, L)
        _right = data[j][1]
        _response.append((_left, _right, 0.0, 0.0))
    return _response


def get_relevant_pp(filename, start, end):
    global pa_prob_dict
    _items = []
    with open(filename, 'rb') as f:
        f.seek(start)
        if start != 0:
            f.readline()
        line = f.readline().decode('utf-8')
        while line:
            _res = line.strip().split('----')
            if len(_res) == 6:
                _pa = _res[-1]
                _arg1_words, _arg2_words = _res[1].split(' '), _res[2].split(' ')
                _arg1_pos, _arg2_pos = _res[3].split(' '), _res[4].split(' ')
                if len(_arg1_words) == len(_arg1_pos) and len(_arg2_words) == len(_arg2_pos):
                    valid_arg1 = filter_arg(_arg1_words, _arg1_pos)
                    valid_arg2 = filter_arg(_arg2_words, _arg2_pos)
                    if len(valid_arg1) > 0 and len(valid_arg2) > 0:
                        if _pa in pa_prob_dict:
                            _items.append((valid_arg1, valid_arg2, _pa))
                        rev_pa = _pa + '_rev'
                        if rev_pa in pa_prob_dict:
                            _items.append((valid_arg2, valid_arg1, rev_pa))
            if f.tell() >= end:
                break
            line = f.readline().decode('utf-8')
    return _items


def load_weak_pp(pp_dir, min_count, workers):
    _filtered_pp = []
    # files = os.listdir(pp_dir)
    files = {pp_dir}
    for file in tqdm(files):
        pool = multiprocessing.Pool()
        # pp_path = os.path.join(pp_dir, file)
        pp_path = file
        filesize = os.path.getsize(pp_path)
        results = []
        for i in range(workers):
            s, e = (filesize * i) // workers, (filesize * (i + 1)) // workers
            results.append(pool.apply_async(
                get_relevant_pp, (pp_path, s, e)
            ))
        pool.close()
        pool.join()
        for result in results:
            _part = result.get()
            _filtered_pp.extend(_part)
            del _part
            gc.collect()
    print('{} weak items is obtained.'.format(len(_filtered_pp)))
    _left_counter, _right_counter = WordCounter(), WordCounter()
    for (arg1, arg2, pa) in _filtered_pp:
        _left_counter.add(arg1)
        _right_counter.add(arg2)
    _left_set = {k for k in _left_counter if _left_counter[k] >= min_count}
    _right_set = {k for k in _right_counter if _right_counter[k] >= min_count}
    _response = []
    for (arg1, arg2, pa) in _filtered_pp:
        _left = [w for w in arg1 if w in _left_set]
        _right = [w for w in arg2 if w in _right_set]
        if len(_left) > 0 and len(_right) > 0:
            _response.append((_left, _right, pa_prob_dict[pa], 1.0))
    return _response, _left_set, _right_set


def load_test_data(annotate_test_path, en_wp_path, left_rev, right_rev):
    _annotate_test_samples = []
    _sharp_word_pair = []
    lines = codecs.open(annotate_test_path, 'r', 'utf-8').readlines()
    for line in lines:
        res = line.strip().split('##')
        phrase_pair = res[0].split('----')
        arg1, arg2 = phrase_pair[0].split(' '), phrase_pair[1].split(' ')
        target_c, target_e = res[1].split(' ')
        if target_c not in left_rev or target_e not in right_rev:
            continue
        _annotate_test_samples.append((arg1, arg2, (target_c, target_e)))
    print('total {} in test file, {} are valid.'.format(len(lines), len(_annotate_test_samples)))
    lines = codecs.open(en_wp_path, 'r', 'utf-8').readlines()
    for line in lines:
        res = line.strip().split(' ')
        w1, w2, label = res
        _sharp_word_pair.append((w1, w2, float(label)))
    print('total number of wp test set is {}.'.format(len(_sharp_word_pair)))
    _res, count = [], 0
    for (w1, w2, label) in _sharp_word_pair:
        if (w1, w2) not in causalVectors:
            _sim = -1.0
            count += 1
        else:
            _sim = causalVectors[(w1, w2)]
        _res.append((_sim, label))
    print('{} items not exist in previous causal embeddings.'.format(count))
    sorted_wp_sim = sorted(_res, key=lambda x: x[0], reverse=True)
    res = get_pr_points(sorted_wp_sim, 1.0)
    recall, precision = zip(*res)
    auc_val = auc(recall, precision)
    print('total number of valid wps which exist in counter is {}, auc value under pr curve is {}.'.format(len(_sharp_word_pair)-count, auc_val))
    return _annotate_test_samples, _sharp_word_pair
        

class Trainer(object):
    @staticmethod
    def norm_embed(x):
        return x/np.sqrt(np.sum(np.square(x)))

    def eval(self, epoch):
        with self.sess.as_default():
            left_embed, right_embed = self.left_embed_dict.eval(), self.right_embed_dict.eval()
        acc, mrr, total = 0, [], len(annotate_test_set)
        print('numbers of initial test set {}.'.format(total))
        for (arg1, arg2, (target_c, target_e)) in annotate_test_set:
            if target_c not in left_indices or target_e not in right_indices:
                continue
            scores = dict()
            for c in arg1:
                if c not in left_indices:
                    continue
                for e in arg2:
                    if e not in right_indices:
                        continue
                    scores[join(c, e)] = np.dot(
                        left_embed[left_indices[c]], right_embed[right_indices[e]]
                    )
            res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            target = join(target_c, target_e)
            if res[0][0] == target:
                acc += 1
            for i, [k, _] in enumerate(res):
                if k == target:
                    mrr.append(1.0 / (i + 1))
        assert len(mrr) == total
        _acc, _mrr = acc / float(total), sum(mrr) / float(len(mrr))
        print('criteria in epoch {}: total {}, acc {}, mrr {}.'.format(epoch+1, total, _acc, _mrr))

        _res, count = [], 0
        for (w1, w2, label) in sharp_wp_set:
            if w1 not in left_indices or w2 not in right_indices:
                _sim = -1.0
                count += 1
            else:
                _sim = np.dot(
                    self.norm_embed(left_embed[left_indices[w1]]), self.norm_embed(right_embed[right_indices[w2]])
                )
            _res.append((_sim, label))
        print('{} items not exist in previous causal embeddings.'.format(count))
        sorted_wp_sim = sorted(_res, key=lambda x: x[0], reverse=True)
        res = get_pr_points(sorted_wp_sim, 1.0)
        recall, precision = zip(*res)
        auc_val = auc(recall, precision)
        print('total number of valid wps which exist in counter is {}, auc value under pr curve is {}.'.format(len(sharp_wp_set)-count, auc_val))
        return _acc, _mrr

    def __init__(self, params):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            self.sess = tf.Session(config=session_conf)
            with tf.variable_scope('initialize_variable'):
                self.left_embed_dict = tf.get_variable(
                    name='left_embed_dict', shape=[left_size, parameters['dim']], dtype=tf.float32, trainable=True,
                    initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=parameters['seed'], dtype=tf.float32),
                )
                self.right_embed_dict = tf.get_variable(
                    name='right_embed_dict', shape=[right_size, parameters['dim']], dtype=tf.float32, trainable=True,
                    initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=parameters['seed'], dtype=tf.float32),
                )
                self.global_steps = tf.get_variable(name='global_steps', initializer=0, trainable=False)

            self.input_left = tf.placeholder(tf.int32, [None, max_len])
            self.input_right = tf.placeholder(tf.int32, [None, max_len])
            self.left_len = tf.placeholder(tf.float32, [None, ])
            self.right_len = tf.placeholder(tf.float32, [None, ])
            self.targets = tf.placeholder(tf.float32, [None, ])
            # self.pattern_prob = tf.placeholder(tf.float32, [None, ])
            # self.alpha = tf.placeholder(tf.float32, name='alpha')
            # self.gamma = tf.placeholder(tf.float32, name='gamma')
            # self.bs = tf.placeholder(tf.int32, name='bs')

            self.input_left_embed = tf.nn.embedding_lookup(self.left_embed_dict, self.input_left)
            self.input_right_embed = tf.nn.embedding_lookup(self.right_embed_dict, self.input_right)
            left_mask = tf.sequence_mask(self.left_len, max_len, dtype=tf.float32)
            right_mask = tf.sequence_mask(self.right_len, max_len, dtype=tf.float32)
            mask_matrix = tf.matmul(tf.expand_dims(left_mask, 2), tf.expand_dims(right_mask, 1))
            logits = tf.matmul(self.input_left_embed, tf.transpose(self.input_right_embed, perm=[0, 2, 1]))
            _probs = tf.clip_by_value(tf.sigmoid(logits), 1e-5, 1.0 - 1e-5)
            max_prob = tf.reduce_max(_probs*mask_matrix, axis=[1, 2])
            # pos_fl = -self.alpha * tf.pow(1 - max_prob, self.gamma) * tf.log(max_prob) * pos_target
            # _3d_neg_fl = (self.alpha - 1) * tf.pow(_probs, self.gamma) * tf.log(1 - _probs)
            # neg_fl = tf.reduce_sum(_3d_neg_fl*mask_matrix, axis=[1, 2])*(1.0-self.targets)
            # pos_fl = -tf.log(max_prob)*(2.*self.pattern_prob-0.6)*self.targets
            pos_fl = -tf.log(max_prob)*self.targets

            neg_fl = tf.reduce_sum(-tf.log(1.0 - _probs) * mask_matrix, axis=[1, 2]) * (1.0 - self.targets)
            self.loss = tf.reduce_sum([pos_fl, neg_fl])
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=parameters['lr']).minimize(
                self.loss, global_step=self.global_steps)
            self.sess.run(tf.global_variables_initializer())

    def write_embedding(self, cause_output_path, effect_output_path, step):
        tail = '_{}.txt'.format(step)
        cause_path, effect_path = cause_output_path+tail, effect_output_path+tail
        with codecs.open(cause_path, 'w', 'utf-8') as fcause, codecs.open(effect_path, 'w', 'utf-8') as feffect:
            with self.sess.as_default():
                cause_embeddings, effect_embeddings = self.left_embed_dict.eval(), self.right_embed_dict.eval()
                fcause.write('{} {}\n'.format(left_size, parameters['dim']))
                for i in range(left_size):
                    fcause.write('{} {}\n'.format(left_vocab[i], ' '.join(list(map(str, cause_embeddings[i])))))

                feffect.write('{} {}\n'.format(right_size, parameters['dim']))
                for i in range(right_size):
                    feffect.write('{} {}\n'.format(right_vocab[i], ' '.join(list(map(str, effect_embeddings[i])))))
        print('word embedding are stored in {} and {} respectively!'.format(cause_path, effect_path))

    def run(self, params):
        print('model: Max started!\n')
        with self.sess.as_default():
            for current_epoch in range(parameters['num_epochs']):
                print('current epoch: {} started.'.format(current_epoch + 1))
                ave_loss, count = 0.0, 0
                start_time = time()
                train_batches = generate_batches(samples, parameters['batch_size'])
                for pos_batch in train_batches:
                    neg_batch = sample_negative(samples, len(pos_batch)*parameters['num_sample'])
                    # whole_batch = np.concatenate([pos_batch, np.array(neg_batch)], 0)
                    whole_batch = list(pos_batch)+neg_batch
                    # left_w, right_w, len1, len2, pa_prob, label, bs = complete(whole_batch, left_indices, right_indices, max_len)
                    left_w, right_w, len1, len2, label = complete(whole_batch, left_indices, right_indices, max_len)

                    feed_dict = {
                        self.input_left: left_w,
                        self.input_right: right_w,
                        self.targets: label,
                        # self.pattern_prob: pa_prob,
                        self.left_len: len1,
                        self.right_len: len2,
                        # self.alpha: parameters['alpha'],
                        # self.gamma: parameters['gamma'],
                        # self.bs: bs
                    }
                    _, _, _loss = self.sess.run([self.train_op, self.global_steps, self.loss], feed_dict=feed_dict)
                    count += 1
                    if count % 500 == 0:
                        print(count)

                    ave_loss += _loss
                ave_loss /= count
                print('Average loss at epoch {} is {}!'.format(current_epoch + 1, ave_loss))
                self.eval(current_epoch)
                # if current_epoch % 2 == 0 and current_epoch != 0:
                #     self.write_embedding(parameters['cause_path'], parameters['effect_path'], str(current_epoch + 1))
                end_time = time()
                print('epoch: {} uses {} minutes.\n'.format(current_epoch + 1, float(end_time - start_time) / 60))


if __name__ == '__main__':
    path = os.path.join(project_source_path, 'boostvec/')
    parameters = {
        'target': '强弱pp混合,10倍负采样,gpu内算mask',
        'cause_vec_path': os.path.join(path, 'en/nn/fl_max_cause_11.txt'),
        'effect_vec_path': os.path.join(path, 'en/nn/fl_max_effect_11.txt'),
        'pa_prob_path': os.path.join(path, 'en/nn/models/pattern/prob_of_patterns_rev_2.txt'),
        'strong_pp_path': os.path.join(path, 'en/nn/sharp_data.txt'),
        'weak_pp_dir': os.path.join(path, 'en/nn/EnwikiStrongExtractions'),
        'annotated_test_path': os.path.join(path, 'en/nn/sharp_annotate.txt'),
        'en_wp_testset': os.path.join(path, 'en/nn/en_wp_testset.txt'),
        'cause_path': os.path.join(path, 'en/nn/models/embed/boosted_max_cause'),
        'effect_path': os.path.join(path, 'en/nn/models/embed/boosted_max_effect'),

        'strong_w_count': 1,  # 强pattern的min-count
        'weak_w_count': 5,  # 弱pattern的min-count
        'pa_thres': 0.5,  # pattern的概率

        'dim': 100,
        'max_len': 25,
        'num_sample': 10,
        'num_threads': 16,
        'seed': 5,
        'num_epochs': 30,
        'batch_size': 256,
        'show': 100,
        'lr': 5e-3,
        'alpha': 0.8,
        'gamma': 2.0,
    }
    print(parameters['pa_prob_path'])
    print('boosting max-matching start!\n')

    pa_prob_dict = load_weak_pattern(
        parameters['pa_prob_path'], parameters['pa_thres']
    )

    weak_pp, weak_left_set, weak_right_set = load_weak_pp(
        pp_dir=parameters['weak_pp_dir'], min_count=parameters['weak_w_count'],
        workers=parameters['num_threads']
    )
    strong_pp, strong_left_set, strong_right_set = load_strong_pp(
        parameters['strong_pp_path'], parameters['strong_w_count']
    )
    causalVectors = CausalVector(parameters['cause_vec_path'], parameters['effect_vec_path'], norm=True)

    left_v = strong_left_set | weak_left_set
    right_v = strong_right_set | weak_right_set

    # left_v = strong_left_set
    # right_v = strong_right_set
    oov = ['pad']
    left_vocab, right_vocab = oov + list(left_v), oov + list(right_v)
    left_indices = {w: i for i, w in enumerate(left_vocab)}
    right_indices = {w: i for i, w in enumerate(right_vocab)}
    left_size, right_size = len(left_vocab), len(right_vocab)
    print('vocab len, left {}, right {}.'.format(len(left_v) + 1, len(right_v) + 1))
    samples = convert2indices(left_indices, right_indices, strong_pp + weak_pp, parameters['max_len'])
    # samples = convert2indices(left_indices, right_indices, strong_pp, parameters['max_len'])

    max_len = parameters['max_len']
    print('number of pp is {}.'.format(len(samples)))
    # del strong_pp, weak_pp
    annotate_test_set, sharp_wp_set = load_test_data(parameters['annotated_test_path'], parameters['en_wp_testset'], left_indices, right_indices)

    Trainer(parameters).run(parameters)

