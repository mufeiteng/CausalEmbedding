import numpy as np
import tensorflow as tf
import math
import codecs
from causalvec.en.dataloader import *
from sklearn.metrics import auc


def norm_embed(x):
    return x / np.sqrt(np.sum(np.square(x)))


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


class BaseModel(object):

    def __init__(self, embedding_size, batch_size, num_epochs, num_samples, learning_rate, data_loader):
        assert isinstance(data_loader, Data)
        self.dataloader = data_loader
        self.max_len = data_loader.max_len
        self.num_samples, self.num_epochs = num_samples, num_epochs
        self.embedding_size, self.batch_size = embedding_size, batch_size
        # used for showing similarity of cause->effect and effect->cause
        self.cause_word_id, self.effect_word_id = None, None
        self.cause_normed_embed, self.c2e_similar = None, None
        self.effect_normed_embed, self.e2c_similar = None, None
    
        self.input_left, self.input_right = None, None
        self.left_len, self.right_len, self.targets = None, None, None
        self.input_left_embed, self.input_right_embed = None, None
        # embedding of cause and effect vocabs
        self.cause_embed_dict, self.effect_embed_dict = None, None
    
        self.train_op, self.loss, self.global_steps = None, None, None
        self.learning_rate, self.average_loss = learning_rate, 0.0
        self.graph, self.sess = tf.Graph(), None

    @staticmethod
    def shuffle_pos_neg(pos_data, neg_data):
        data = np.array(pos_data+neg_data)
        shuffle_indices = np.random.permutation(np.arange(len(data)))
        return data[shuffle_indices]

    def init_embedding(self, left_size, right_size):
        self.cause_embed_dict = tf.Variable(tf.truncated_normal(
            [left_size, self.embedding_size], stddev=0.01 / math.sqrt(self.embedding_size), dtype=tf.float32)
        )
    
        self.effect_embed_dict = tf.Variable(tf.truncated_normal(
            [right_size, self.embedding_size], stddev=0.01 / math.sqrt(self.embedding_size), dtype=tf.float32)
        )

    def calculate_similar(self):
        self.cause_word_id = tf.constant(self.dataloader.c2e_visualize, dtype=tf.int32)
        cause_norm = tf.sqrt(tf.reduce_sum(tf.square(self.cause_embed_dict), 1, keep_dims=True))
        self.cause_normed_embed = self.cause_embed_dict / cause_norm
        c_test_embed = tf.nn.embedding_lookup(self.cause_normed_embed, self.cause_word_id)
        self.effect_word_id = tf.constant(self.dataloader.e2c_visualize, dtype=tf.int32)
        effect_norm = tf.sqrt(tf.reduce_sum(tf.square(self.effect_embed_dict), 1, keep_dims=True))
        self.effect_normed_embed = self.effect_embed_dict / effect_norm
        e_test_embed = tf.nn.embedding_lookup(self.effect_normed_embed, self.effect_word_id)
        self.c2e_similar = tf.matmul(c_test_embed, tf.transpose(self.effect_normed_embed))
        self.e2c_similar = tf.matmul(e_test_embed, tf.transpose(self.cause_normed_embed))

    def show_similar(self):
        top_k = 15
        sim = self.c2e_similar.eval()
        for i in range(len(self.dataloader.c2e_visualize)):
            valid_word = self.dataloader.vocab_left[self.dataloader.c2e_visualize[i]]
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest effect words to %s:' % valid_word
            for k in range(top_k):
                close_word = self.dataloader.vocab_right[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
        print('\n\n')
        sim = self.e2c_similar.eval()
        for i in range(len(self.dataloader.e2c_visualize)):
            valid_word = self.dataloader.vocab_right[self.dataloader.e2c_visualize[i]]
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest cause words to %s:' % valid_word
            for k in range(top_k):
                close_word = self.dataloader.vocab_left[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def generate_batches(data, batch_size, shuffle=True):
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

    def write_embedding(self, cause_output_path, effect_output_path, step):
        assert isinstance(self.dataloader, Data)
        tail = '_{}.txt'.format(step)
        cause_path, effect_path = cause_output_path + tail, effect_output_path + tail
        with codecs.open(cause_path, 'w', 'utf-8') as fcause, codecs.open(effect_path, 'w', 'utf-8') as feffect:
            with self.sess.as_default():
                cause_embeddings, effect_embeddings = self.cause_embed_dict.eval(), self.effect_embed_dict.eval()
                fcause.write('{} {}\n'.format(self.dataloader.vocab_left_size, self.embedding_size))
                for i in range(self.dataloader.vocab_left_size):
                    fcause.write('{} {}\n'.format(self.dataloader.vocab_left[i], ' '.join(list(map(str, cause_embeddings[i])))))
            
                feffect.write('{} {}\n'.format(self.dataloader.vocab_right_size, self.embedding_size))
                for i in range(self.dataloader.vocab_right_size):
                    feffect.write('{} {}\n'.format(self.dataloader.vocab_right[i], ' '.join(list(map(str, effect_embeddings[i])))))
        print('word embedding are stored in {} and {} respectively!'.format(cause_path, effect_path))

    @staticmethod
    def mask_softmax(match_matrix, mask_matrix):
        """
        :param match_matrix: (batch, max_len, max_len)
        :param mask_matrix: (batch, max_len, max_len)
        :return:
        """
        match_matrix_masked = match_matrix * mask_matrix

        match_matrix_shifted_1 = mask_matrix * tf.exp(
            match_matrix_masked - tf.reduce_max(match_matrix_masked, axis=1, keep_dims=True))
        match_matrix_shifted_2 = mask_matrix * tf.exp(
            match_matrix_masked - tf.reduce_max(match_matrix_masked, axis=2, keep_dims=True))

        Z1 = tf.reduce_sum(match_matrix_shifted_1, axis=1, keep_dims=True)
        Z2 = tf.reduce_sum(match_matrix_shifted_2, axis=2, keep_dims=True)
        softmax_1 = match_matrix_shifted_1 / (Z1 + 1e-12)  # weight of left words
        softmax_2 = match_matrix_shifted_2 / (Z2 + 1e-12)  # weight of right words
        return softmax_1, softmax_2

    @staticmethod
    def make_attention(input_left_embed, input_right_embed):
        return tf.matmul(input_left_embed, tf.transpose(input_right_embed, perm=[0, 2, 1]))

    def eval(self, epoch, sharp_wp_set, left_indices, right_indices):
        with self.sess.as_default():
            left_embed, right_embed = self.cause_embed_dict.eval(), self.effect_embed_dict.eval()
    
        _res, count = [], 0
        for (w1, w2, label) in sharp_wp_set:
            if w1 not in left_indices or w2 not in right_indices:
                _sim = -1.0
                count += 1
            else:
                _sim = np.dot(
                    norm_embed(left_embed[left_indices[w1]]), norm_embed(right_embed[right_indices[w2]])
                )
            _res.append((_sim, label))
        print('{} items not exist in previous causal embeddings.'.format(count))
        sorted_wp_sim = sorted(_res, key=lambda x: x[0], reverse=True)
        res = get_pr_points(sorted_wp_sim, 1.0)
        recall, precision = zip(*res)
        auc_val = auc(recall, precision)
        print('total number of valid wps which exist in counter is {}, auc value under pr curve is {}.'.format(len(sharp_wp_set) - count, auc_val))
        return auc_val


