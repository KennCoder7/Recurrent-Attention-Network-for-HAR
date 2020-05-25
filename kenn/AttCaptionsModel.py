# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf


class AttCaptionsModel(object):
    def __init__(self, word_to_idx, dim_feature=[50, 32], dim_embed=512, dim_hidden=1024, n_time_step=16,
                 prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True, h2out=True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of features.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}  # kenn:iteritems->items
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<start>']
        self._null = word_to_idx['<null>']
        # self._null = 8
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.data = tf.placeholder(tf.float32, [None, 650, 3])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.captions_vector = tf.placeholder(tf.int32, [None, self.T + 1, self.V])
        self.bool_balance = False
        self.h2out = h2out

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    # def _word_embedding(self, inputs, reuse=False):
    #     with tf.variable_scope('word_embedding', reuse=reuse):
    #         w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer, trainable=True)
    #         x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
    #         return x

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            # w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer, trainable=True)
            a = tf.diag([1.0] * self.V, name='diag')
            x = tf.nn.embedding_lookup(a, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    @staticmethod
    def build_attention(local_feature, global_feature, name='Attention', reuse=False):
        cpt_func = 'dot'
        norm_func = 'softmax'
        with tf.variable_scope(name, reuse=reuse):
            n_units = local_feature.get_shape().as_list()[2]
            n_units_g = global_feature.get_shape().as_list()[1]
            if n_units_g != n_units:
                global_feature = tf.layers.dense(
                    inputs=global_feature,
                    units=n_units,
                    # activation=tf.nn.relu,
                    name='{}-Gbf_fc'.format(name),
                )
                print("# {} output shape {}".format(global_feature.name, global_feature.shape))
            with tf.name_scope('{}-Get_cpt_score'.format(name)):
                # local_feature_t = tf.transpose(local_feature, (0, 2, 1))
                print("name", name)
                if cpt_func == 'pc':
                    print("l_vector", local_feature.shape)
                    g_vector = tf.expand_dims(global_feature, axis=1)
                    print("g_vector", g_vector.shape)
                    add_ = tf.add(local_feature, g_vector)
                    print("add_", add_.shape)
                    add_t = tf.transpose(add_, (0, 2, 1))
                    print("add_t", add_t.shape)
                    # len_lcf = local_feature.get_shape().as_list()[1]
                    u_para = tf.Variable(tf.random_normal([1, n_units, 1], mean=0.01, stddev=0.01),
                                         name="{}-U_para".format(name))
                    print("u_para", u_para.shape)
                    dot_ = tf.multiply(add_t, u_para)
                    print("dot_", dot_.shape)
                    score_vector = tf.reduce_sum(dot_, 1)  # (batch,56*56)
                    print("score_vector", score_vector.shape)
                    # compatibility_function = tf.reshape(tf.tensordot(add_, u_para, axes=1),
                    #                                     [-1, len_lcf, 1], name='{}-Cpt_func_pc'.format(name))
                    if norm_func == 'softmax':
                        score = tf.nn.softmax(score_vector)  # (batch,56*56)
                    elif norm_func == 'tanh':
                        score = tf.nn.tanh(score_vector)
                    elif norm_func == 'sigmoid':
                        score = tf.nn.sigmoid(score_vector)
                    else:
                        score = score_vector
                    a_score = tf.expand_dims(score, axis=1)  # (batch,1,56*56)
                    print("a_score", a_score.shape)
                    l_vector_t = tf.transpose(local_feature, (0, 2, 1))
                    print("l_vector_t", l_vector_t.shape)
                    gas = tf.multiply(l_vector_t, a_score)  # (batch,256,56*56)
                    print("gas", gas.shape)
                    ga = tf.reduce_sum(gas, [2])
                    print("ga", ga.shape)
                    return ga, score
                else:
                    # compatibility_function = tf.matmul(local_feature, g, name='{}-Cpt_func_dot'.format(name))
                    l_vector_t = tf.transpose(local_feature, (0, 2, 1))
                    print("l_vector_t", l_vector_t.shape)
                    g_vector = tf.expand_dims(global_feature, axis=2)
                    print("g_vector", g_vector.shape)
                    dot_ = tf.multiply(l_vector_t, g_vector)
                    print("dot_", dot_.shape)
                    score_vector = tf.reduce_sum(dot_, 1)  # (batch,56*56)
                    print("score_vector", score_vector.shape)
                    if norm_func == 'softmax':
                        score = tf.nn.softmax(score_vector)  # (batch,56*56)
                    elif norm_func == 'tanh':
                        score = tf.nn.tanh(score_vector)
                    else:
                        score = score_vector
                    a_score = tf.expand_dims(score, axis=1)  # (batch,1,56*56)
                    gas = tf.multiply(l_vector_t, a_score)  # (batch,256,56*56)
                    ga = tf.reduce_sum(gas, [2])
                    print(ga.shape)
                    return ga, score

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)  # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])  # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')  # (N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')  # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    @staticmethod
    def _batch_norm(x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode == 'train'),
                                            updates_collections=None,
                                            scope=(name + 'batch_norm'))

    def extract_feature_cnn(self):
        data = self.data
        with tf.variable_scope('conv', reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv1d(
                inputs=data,
                filters=16,
                kernel_size=5,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                # trainable=False,
                kernel_initializer=self.weight_initializer,
                bias_initializer=self.const_initializer,
                # use_bias=False
            )
            print('# conv1 shape {}'.format(conv1.shape))
            pool = tf.layers.max_pooling1d(conv1, 2, 2)
            print('# pool1 shape {}'.format(pool.shape))
            conv2 = tf.layers.conv1d(
                inputs=pool,
                filters=32,
                kernel_size=5,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                # trainable=False,
                kernel_initializer=self.weight_initializer,
                bias_initializer=self.const_initializer,
                # use_bias=False
            )
            print('# conv2 shape {}'.format(conv2.shape))
            pool = tf.layers.max_pooling1d(conv2, 2, 2)
            print('# pool2 shape {}'.format(pool.shape))
            conv3 = tf.layers.conv1d(
                inputs=pool,
                filters=64,
                kernel_size=5,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                # trainable=False,
                kernel_initializer=self.weight_initializer,
                bias_initializer=self.const_initializer,
                # use_bias=False
            )
            print('# conv3 shape {}'.format(conv3.shape))
            pool = tf.layers.max_pooling1d(conv3, 2, 2)
            print('# pool3 shape {}'.format(pool.shape))
            conv4 = tf.layers.conv1d(
                inputs=pool,
                filters=128,
                kernel_size=5,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                # trainable=False,
                kernel_initializer=self.weight_initializer,
                bias_initializer=self.const_initializer,
                # use_bias=False
            )
            print('# conv4 shape {}'.format(conv4.shape))
            l_op = conv4
            shape = l_op.get_shape().as_list()
            flat = tf.reshape(l_op, [-1, shape[1] * shape[2]])
            fc1 = tf.layers.dense(
                inputs=flat,
                units=128,
                activation=tf.nn.relu,
                # kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                # bias_initializer=tf.constant_initializer(0)
            )
        return conv1, conv2, conv3, conv4, fc1

    def build_model(self):
        features1, features2, features3, features4, gbf = self.extract_feature_cnn()
        captions = self.captions
        batch_size = tf.shape(features1)[0]

        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        # batch normalize feature vectors
        features1 = self._batch_norm(features1, mode='train', name='conv_features1')
        features2 = self._batch_norm(features2, mode='train', name='conv_features2')
        features3 = self._batch_norm(features3, mode='train', name='conv_features3')
        features4 = self._batch_norm(features4, mode='train', name='conv_features4')

        features = features4
        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=captions_in)
        # x = self.captions_vector[:, :self.T, :]
        print("test_word_embedding", x.shape)
        features_proj = self._project_features(features=features)
        print("test features_proj/features", features_proj.shape, features.shape)

        loss = 0.0
        alpha_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            # context, alpha = self.build_attention(features, h, reuse=(t != 0))
            print("test context/score/h", context.shape, alpha.shape, h.shape)
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))

            with tf.variable_scope('lstm', reuse=(t != 0)):
                # _, (c, h) = lstm_cell(inputs=tf.concat([context], 1), state=[c, h])
                _, (c, h) = lstm_cell(inputs=tf.concat([x[:, t, :], context], 1), state=[c, h])
            print("test x\context shape", x[:, t, :].shape, context.shape)
            logits = self._decode_lstm(x[:, t, :], h, context, dropout=self.dropout, reuse=(t != 0))

            loss += tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=captions_out[:, t], logits=logits) * mask[:, t])

        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))  # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)  # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((1 - alphas_all) ** 2)
            loss += alpha_reg

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=5):
        features = self.extract_feature_cnn()

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            # context, alpha = self.build_attention(features, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t != 0)):
                # _, (c, h) = lstm_cell(inputs=tf.concat([context], 1), state=[c, h])
                _, (c, h) = lstm_cell(inputs=tf.concat([x, context], 1), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=(t != 0))
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))  # (N, T, L)
        if self.selector:
            betas = tf.transpose(tf.squeeze(beta_list), (1, 0))  # (N, T)
        else:
            betas = tf.constant(0)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))  # (N, max_len)
        return alphas, betas, sampled_captions
