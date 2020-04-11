from causalvec.en.basemodel import *
from causalvec.en.dataloader import *
from tools import *


class MaxMatching(BaseModel):
    def __init__(self, embedding_size, batch_size, num_epochs, num_samples, learning_rate, data_loader):
        BaseModel.__init__(self, embedding_size, batch_size, num_epochs, num_samples, learning_rate, data_loader)
        self.alpha, self.gamma = None, None
    
    def construct_graph(self):
        with self.graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            self.sess = tf.Session(config=session_conf)
            
            self.input_left = tf.placeholder(tf.int32, [None, self.max_len])
            self.input_right = tf.placeholder(tf.int32, [None, self.max_len])
            self.left_len = tf.placeholder(tf.float32, [None, ])
            self.right_len = tf.placeholder(tf.float32, [None, ])
            self.targets = tf.placeholder(tf.float32, [None, ])
            self.alpha = tf.placeholder(tf.float32, name='alpha')
            self.gamma = tf.placeholder(tf.float32, name='gamma')
            
            self.init_embedding(left_size=self.dataloader.vocab_left_size, right_size=self.dataloader.vocab_right_size)
            self.input_left_embed = tf.nn.embedding_lookup(self.cause_embed_dict, self.input_left)
            self.input_right_embed = tf.nn.embedding_lookup(self.effect_embed_dict, self.input_right)
            """
            focal loss:
                for positive samples, loss is -alpha*((1-p)**gamma)*log(p)
                for negative samples, loss is -(1-alpha)*(p**gamma)*log(1-p)

            """
            left_mask = tf.sequence_mask(self.left_len, self.max_len, dtype=tf.float32)
            right_mask = tf.sequence_mask(self.right_len, self.max_len, dtype=tf.float32)
            mask_matrix = tf.matmul(tf.expand_dims(left_mask, 2), tf.expand_dims(right_mask, 1))

            logits = self.make_attention(self.input_left_embed, self.input_right_embed)

            # 按行按列做softmax
            softmax_1, softmax_2 = self.mask_softmax(logits, mask_matrix)
            left_attentive = tf.matmul(tf.transpose(softmax_1, [0, 2, 1]), self.input_left_embed)  # (batch, r, dims)
            right_attentive = tf.matmul(softmax_2, self.input_right_embed)  # (batch, l, dims)
            right_interaction = tf.reduce_sum(left_attentive * self.input_right_embed, axis=2)  # (batch, r)
            left_interaction = tf.reduce_sum(right_attentive * self.input_left_embed, axis=2)  # (batch, l)

            right_probs = tf.clip_by_value(
                tf.reduce_max(tf.sigmoid(right_interaction) * right_mask, 1), 1e-5, 1.0 - 1e-5
            )  # (batch,)
            left_probs = tf.clip_by_value(
                tf.reduce_max(tf.sigmoid(left_interaction) * left_mask, 1), 1e-5, 1.0 - 1e-5
            )  # (batch,)

            left_pos_fl = tf.reduce_sum(
                -self.alpha * tf.pow(1 - left_probs, self.gamma) * tf.log(left_probs) * self.targets
            )
            right_pos_fl = tf.reduce_sum(
                -self.alpha * tf.pow(1 - right_probs, self.gamma) * tf.log(right_probs) * self.targets
            )

            _pro = tf.clip_by_value(tf.sigmoid(logits), 1e-5, 1.0 - 1e-5)
            _3d_focal = (self.alpha - 1) * tf.pow(_pro, self.gamma) * tf.log(1 - _pro) * mask_matrix
            neg_fl = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(_3d_focal, axis=1), axis=1) * (1.0 - self.targets))
            # neg_fl = tf.reduce_sum(
            #     ((self.alpha - 1) * tf.pow(_pro, self.gamma) * tf.log(1 - _pro) * mask_matrix)*(1.0-self.targets)
            # )

            self.loss = tf.reduce_sum([left_pos_fl, right_pos_fl, neg_fl])

            # self.calculate_similar()  # high time-consuming
            self.global_steps = tf.Variable(0, trainable=False)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss, global_step=self.global_steps)
            self.sess.run(tf.global_variables_initializer())
    
    def train_stage(self):
        print('model: Max started!\n')
        with self.sess.as_default():
            assert isinstance(self.dataloader, Data)
            base_auc = 0.5
            for current_epoch in range(self.num_epochs):
                print('current epoch: {} started.'.format(current_epoch + 1))
                ave_loss, count = 0.0, 0
                start_time = time()
                train_batches = self.generate_batches(self.dataloader.train, self.batch_size)
                for pos_batch in train_batches:
                    neg_batch = sample_negative(self.dataloader.train, len(pos_batch) * self.num_samples)
                    whole_batch = list(pos_batch) + neg_batch
                    left_w, right_w, len1, len2, label = complete_data(
                        whole_batch, self.dataloader.vocab_rev_left,
                        self.dataloader.vocab_rev_right, self.dataloader.max_len, '<pad>'
                    )
                    feed_dict = {
                        self.input_left: left_w,
                        self.input_right: right_w,
                        self.targets: label,
                        self.left_len: len1,
                        self.right_len: len2,
                        self.alpha: 0.8,
                        self.gamma: 2.0
                    }
                    _, global_step, _loss = self.sess.run([self.train_op, self.global_steps, self.loss], feed_dict=feed_dict)
                    count += 1
                    if count % 500 == 0:
                        print(count)
                    ave_loss += _loss
                ave_loss /= count
                print('Average loss at epoch {} is {}!'.format(current_epoch + 1, ave_loss))
                auc_val = self.eval(current_epoch)
                if auc_val > base_acc:
                    base_acc = auc_val
                    self.write_embedding(params['cause_path'], params['effect_path'], str(current_epoch + 1))
                end_time = time()
                print('epoch: {} uses {} minutes.\n'.format(current_epoch + 1, float(end_time - start_time) / 60))


if __name__ == '__main__':
    path = os.path.join(project_source_path, 'causalembedding/')
    params = {
        'train_path': os.path.join(path, 'sharp_data.txt'),
        'test_path': os.path.join(path, 'en_wp_testset.txt'),
        'batch_size': 256,
        'num_epochs': 50,
        'embedding_size': 100,
        'learning_rate': 0.005,
        'cause_path': os.path.join(project_source_path, 'embedding/max_cause'),
        'effect_path': os.path.join(project_source_path, 'embedding/max_effect'),
        'min_count': 8,
        'num_samples': 10,
    }
    
    loader = Data()
    loader.prepare_data(params['train_path'], params['test_path'], params['min_count'])
    causalVec = MaxMatching(
        embedding_size=params['embedding_size'], batch_size=params['batch_size'], num_epochs=params['num_epochs'],
        learning_rate=params['learning_rate'], num_samples=params['num_samples'], data_loader=loader
    )
    causalVec.construct_graph()
    causalVec.train_stage()
    print('train stage is over!')