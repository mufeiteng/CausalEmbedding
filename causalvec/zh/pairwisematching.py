from causalvec.zh.basemodel import *
from causalvec.zh.dataloader import *
from tools import *


class MaxMatching(BaseModel):
    def __init__(self, embedding_size, batch_size, num_epochs, num_samples, learning_rate, data_loader):
        BaseModel.__init__(self, embedding_size, batch_size, num_epochs, num_samples, learning_rate, data_loader)
    
    def construct_graph(self):
        with self.graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            self.sess = tf.Session(config=session_conf)
            
            self.input_left = tf.placeholder(tf.int32, [None, self.max_len])
            self.input_right = tf.placeholder(tf.int32, [None, self.max_len])
            self.left_len = tf.placeholder(tf.int32, [None, ])
            self.right_len = tf.placeholder(tf.int32, [None, ])
            self.targets = tf.placeholder(tf.float32, [None, ])

            self.input_left_embed = tf.nn.embedding_lookup(self.cause_embed_dict, self.input_left)
            self.input_right_embed = tf.nn.embedding_lookup(self.effect_embed_dict, self.input_right)
            left_mask = tf.sequence_mask(self.left_len, self.max_len, dtype=tf.float32)
            right_mask = tf.sequence_mask(self.right_len, self.max_len, dtype=tf.float32)
            mask_matrix = tf.matmul(tf.expand_dims(left_mask, 2), tf.expand_dims(right_mask, 1))

            logits = self.make_attention(self.input_left_embed, self.input_right_embed)
            _probs = tf.clip_by_value(tf.sigmoid(logits), 1e-5, 1.0 - 1e-5)

            pos_fl = tf.reduce_sum(-tf.log(_probs) * mask_matrix, axis=[1, 2]) * self.targets
            neg_fl = tf.reduce_sum(-tf.log(1.0 - _probs) * mask_matrix, axis=[1, 2]) * (1.0 - self.targets)
            self.loss = tf.reduce_sum([pos_fl, neg_fl])
            
            # self.calculate_similar()  # high time-consuming
            self.global_steps = tf.Variable(0, trainable=False)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss, global_step=self.global_steps)
            self.sess.run(tf.global_variables_initializer())

    def train_stage(self):
        print('model: Max started!\n')
        with self.sess.as_default():
            assert isinstance(self.dataloader, Data)
            base_acc = 0.5
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
                    }
                    _, global_step, _loss = self.sess.run([self.train_op, self.global_steps, self.loss], feed_dict=feed_dict)
                    count += 1
                    if count % 500 == 0:
                        print(count)
                    ave_loss += _loss
                ave_loss /= count
                print('Average loss at epoch {} is {}!'.format(current_epoch + 1, ave_loss))
                acc, mrr = self.eval(current_epoch)
                if acc > base_acc:
                    base_acc = acc
                    self.write_embedding(params['cause_path'], params['effect_path'], str(current_epoch + 1))
                end_time = time()
                print('epoch: {} uses {} minutes.\n'.format(current_epoch + 1, float(end_time - start_time) / 60))


if __name__ == '__main__':
    path = os.path.join(project_source_path, 'causalembedding/')
    params = {
        'train_path': os.path.join(path, 'sg_positives.txt'),
        'test_path': os.path.join(path, 'bk_eva.txt'),
        'batch_size': 256,
        'num_epochs': 50,
        'embedding_size': 100,
        'learning_rate': 0.005,
        'cause_path': os.path.join(project_source_path, 'embedding/pw_cause'),
        'effect_path': os.path.join(project_source_path, 'embedding/pw_effect'),
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