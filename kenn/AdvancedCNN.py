import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from utils import shuffle
from utils import extract_batch_size
from utils import cross_val
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
GPU_MEMORY = 1


class AdvancedCNN(object):
    def __init__(self, train_x, train_y, test_x, test_y,
                 seg_len=50, num_channels=3, num_labels=3,
                 num_conv=3, filters=64, k_size=5, conv_strides=1, pool_size=2, pool_strides=2,
                 batch_size=200, learning_rate=0.0001, num_epochs=100,
                 print_val_each_epoch=10, print_test_each_epoch=50, print_cm=False,
                 padding='valid', cnn_type='1d', bool_bn=False, act_func='relu',
                 no_exp= 1):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.seg_len = seg_len
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_batches = train_x.shape[0] // self.batch_size

        self.cnn_type = cnn_type
        if cnn_type == '2d':
            self.X = tf.placeholder(tf.float32, (None, self.seg_len, self.num_channels, 1))
            self.train_x = train_x.reshape([-1, self.seg_len, self.num_channels, 1])
            self.test_x = test_x.reshape([-1, self.seg_len, self.num_channels, 1])
        else:
            self.X = tf.placeholder(tf.float32, (None, self.seg_len, self.num_channels))
        self.Y = tf.placeholder(tf.float32, (None, self.num_labels))
        self.is_training = tf.placeholder(tf.bool)

        self.num_conv = num_conv
        self.filters = filters
        self.k_size = k_size
        self.conv_strides = conv_strides
        self.pool_size = pool_size
        self.pool_strides = pool_strides

        self.print_val_each_epoch = print_val_each_epoch
        self.print_test_each_epoch = print_test_each_epoch
        self.print_cm = print_cm
        self.padding = padding

        self.bool_bn = bool_bn
        # if act_func == 'relu':
        #     self.act_func = tf.nn.relu
        self.act_func = act_func
        self.no_exp = no_exp

    def _batch_norm(self, x):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=self.is_training,
                                            updates_collections=None)

    def build_network(self):
        for i in range(self.num_conv):
            if i == 0:
                conv = tf.layers.conv1d(
                    inputs=self.X,
                    filters=self.filters,
                    kernel_size=self.k_size,
                    strides=self.conv_strides,
                    padding=self.padding,
                    # activation=tf.nn.relu
                )
                print('# conv shape {}'.format(conv.shape))
                if self.bool_bn:
                    conv = self._batch_norm(conv)
                if self.act_func == 'relu':
                    conv = tf.nn.relu(conv)
                pool = tf.layers.max_pooling1d(
                    inputs=conv,
                    pool_size=self.pool_size,
                    strides=self.pool_strides,
                    padding='same'
                )
                print('# pool shape {}'.format(pool.shape))
            else:
                conv = tf.layers.conv1d(
                    inputs=pool,
                    filters=int(self.filters * (i + 1)),
                    kernel_size=self.k_size,
                    strides=self.conv_strides,
                    padding=self.padding,
                    # activation=tf.nn.relu
                )
                print('# conv shape {}'.format(conv.shape))
                if self.bool_bn:
                    conv = self._batch_norm(conv)
                if self.act_func == 'relu':
                    conv = tf.nn.relu(conv)
                pool = tf.layers.max_pooling1d(
                    inputs=conv,
                    pool_size=self.pool_size,
                    strides=self.pool_strides,
                    padding='same'
                )
                print('# pool shape {}'.format(pool.shape))
        l_op = pool
        # print('# bn shape {}'.format(l_op.shape))
        # l_op = pool
        shape = l_op.get_shape().as_list()
        flat = tf.reshape(l_op, [-1, shape[1] * shape[2]])
        fc1 = tf.layers.dense(
            inputs=flat,
            units=100,
            activation=tf.nn.relu
        )
        softmax = tf.layers.dense(
            inputs=fc1,
            units=self.num_labels,
            activation=tf.nn.softmax
        )
        return softmax

    def build_network_2d(self):
        for i in range(self.num_conv):
            if i == 0:
                conv = tf.layers.conv2d(
                    inputs=self.X,
                    filters=self.filters,
                    kernel_size=(self.k_size, 1),
                    strides=(self.conv_strides, 1),
                    padding=self.padding,
                    activation=tf.nn.relu
                )
                print('# conv shape {}'.format(conv.shape))
                if self.bool_bn:
                    conv = self._batch_norm(conv)
                if self.act_func:
                    conv = tf.nn.relu(conv)
                pool = tf.layers.max_pooling2d(
                    inputs=conv,
                    pool_size=(self.pool_size, 1),
                    strides=(self.pool_strides, 1),
                    padding='same'
                )
                print('# pool shape {}'.format(pool.shape))
            else:
                conv = tf.layers.conv2d(
                    inputs=pool,
                    filters=int(self.filters * (i + 1)),
                    kernel_size=(self.k_size, 1),
                    strides=(self.conv_strides, 1),
                    padding=self.padding,
                    activation=tf.nn.relu
                )
                print('# conv shape {}'.format(conv.shape))
                if self.bool_bn:
                    conv = self._batch_norm(conv)
                if self.act_func:
                    conv = tf.nn.relu(conv)
                pool = tf.layers.max_pooling2d(
                    inputs=conv,
                    pool_size=(self.pool_size, 1),
                    strides=(self.pool_strides, 1),
                    padding='same'
                )
                print('# pool shape {}'.format(pool.shape))
        l_op = pool
        shape = l_op.get_shape().as_list()
        flat = tf.reshape(l_op, [-1, shape[1] * shape[2] * shape[3]])
        fc1 = tf.layers.dense(
            inputs=flat,
            units=100,
            activation=tf.nn.relu
        )
        softmax = tf.layers.dense(
            inputs=fc1,
            units=self.num_labels,
            activation=tf.nn.softmax
        )
        return softmax

    def train(self):
        if self.cnn_type == '2d':
            y_ = self.build_network_2d()
        else:
            y_ = self.build_network()
        loss = -tf.reduce_mean(self.Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        correct = tf.equal(tf.argmax(y_, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        train_x, train_y = shuffle(self.train_x, self.train_y)
        train_xc, train_yc, val_xc, val_yc = cross_val(train_x, train_y, self.no_exp)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY
        with tf.Session(config=sess_config) as sess:
            tf.global_variables_initializer().run()
            for epoch in range(self.num_epochs):
                train_xc, train_yc = shuffle(train_xc, train_yc)
                for i in range(self.num_batches):
                    batch_x = extract_batch_size(train_xc, i, self.batch_size)
                    batch_y = extract_batch_size(train_yc, i, self.batch_size)
                    _, c = sess.run([train_op, loss], feed_dict={self.X: batch_x, self.Y: batch_y,
                                                                 self.is_training: True})
                if (epoch + 1) % self.print_val_each_epoch == 0:
                    print("### Epoch: ", epoch + 1, "|Train loss = ", c,
                          "|Val acc = ", sess.run(accuracy, feed_dict={self.X: val_xc, self.Y: val_yc,
                                                                       self.is_training: False}), " ###")
                # if (epoch + 1) % self.print_test_each_epoch == 0:
                #     print("### 1st After Epoch: ", epoch + 1,
                #           " |Test acc = ", sess.run(accuracy,
                #                                     feed_dict={self.X: self.test_x, self.Y: self.test_y,
                #                                                self.is_training: False}), " ###")
                if (epoch + 1) % self.print_test_each_epoch == 0:
                    test_acc = np.empty(0)
                    for i in range(self.test_x.shape[0] // self.batch_size):
                        batch_x_t = extract_batch_size(self.test_x, i, self.batch_size)
                        batch_y_t = extract_batch_size(self.test_y, i, self.batch_size)
                        test_acc = np.append(test_acc,
                                             sess.run(correct,
                                                      feed_dict={self.X: batch_x_t, self.Y: batch_y_t,
                                                                 self.is_training: False}))
                    # print(test_acc.shape)
                    _test_acc = np.average(test_acc)
                    print("### After Epoch: ", epoch + 1,
                          " |Test acc = ", _test_acc, " ###")
                    if self.print_cm:
                        pred_y = sess.run(tf.argmax(y_, 1), feed_dict={self.X: self.test_x, self.is_training: False})
                        cm = confusion_matrix(np.argmax(self.test_y, 1), pred_y, )
                        print(cm)

