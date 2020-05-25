import tensorflow as tf
import matplotlib.pyplot as plt
# import skimage.transform
# import numpy as np
import time
import os
# import cPickle as pickle
# from scipy import ndimage
from utils import *
from evaluate import *
from visual import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
GPU_MEMORY = 1


# from bleu import evaluate
# from ExtractFeatures import cnn_extract_features


class AttCaptionsSolver(object):
    def __init__(self, model, data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17)
                - image_idxs: Indices for mapping caption to image of shape (400000, )
                - word_to_idx: Mapping dictionary from word to index
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_model: String; model path for test
        """

        self.model = model
        self.data = data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_every = kwargs.pop('print_every', 100)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/kenn/model-1')

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.bool_save_model = kwargs.pop('bool_save_model', False)
        self.generated_caption_len = kwargs.pop('generated_caption_len', 4)
        self.bool_val = kwargs.pop('bool_val', True)
        self.bool_selector = kwargs.pop('bool_selector', True)

        self.bool_save_this = False
        self.ini_acc = 0.85

    def train(self):
        n_examples = self.data['data'].shape[0]
        data = self.data['data']
        captions = self.data['captions']
        data_idxs = self.data['data_idxs']  # 多条caps对应单个数据时用来对应的。
        # val_data = self.val_data['data']
        # n_iters_val = int(np.ceil(float(val_data.shape[0]) / self.batch_size))
        captions_s, idxs_s = shuffle(captions, data_idxs)
        train_caps, train_idxs, val_caps, val_idxs = cross_val(captions_s, idxs_s, 1, cross=5)

        # build graphs for training model and sampling captions
        # This scope fixed things!!
        with tf.variable_scope(tf.get_variable_scope()):
            loss = self.model.build_model()
            tf.get_variable_scope().reuse_variables()
            _, _, generated_captions = self.model.build_sampler(max_len=self.generated_caption_len)

        # train op
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # summary op
        # tf.scalar_summary('batch_loss', loss)
        # tf.summary.scalar('batch_loss', loss)
        # for var in tf.trainable_variables():
        #     # tf.histogram_summary(var.op.name, var)
        #     tf.summary.histogram(var.op.name, var)
        # for grad, var in grads_and_vars:
        #     # tf.histogram_summary(var.op.name+'/gradient', grad)
        #     tf.summary.histogram(var.op.name + '/gradient', grad)

        # summary_op = tf.merge_all_summaries()
        summary_op = tf.summary.merge_all()

        print("The number of epoch: %d" % self.n_epochs)
        print("Data size: %d" % n_examples)
        print("Batch size: %d" % self.batch_size)
        # print("Iterations per epoch: %d" % n_iters_per_epoch)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY
        with tf.Session(config=sess_config) as sess:
            tf.global_variables_initializer().run()
            # summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
            # summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                train_caps, train_idxs = shuffle(train_caps, train_idxs)
                n_iters_per_epoch = train_caps.shape[0] // self.batch_size
                for i in range(n_iters_per_epoch):
                    captions_batch = train_caps[i * self.batch_size:(i + 1) * self.batch_size]
                    idxs_batch = train_idxs[i * self.batch_size:(i + 1) * self.batch_size]
                    data_batch = data[idxs_batch]
                    feed_dict = {self.model.data: data_batch, self.model.captions: captions_batch}
                    _, loss_ = sess.run([train_op, loss], feed_dict)
                    curr_loss += loss_

                    # write summary for tensorboard visualization
                    # if i % 10 == 0:
                    #     summary = sess.run(summary_op, feed_dict)
                    #     summary_writer.add_summary(summary, e * n_iters_per_epoch + i)

                    # if (i + 1) % self.print_every == 0:  # 打印出train中epoch x， batch x中第一个数据[0]的预测结果
                    #     print("\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" % (e + 1, i + 1, loss_))
                    #     # ground_truths = captions[data_idxs == idxs_batch[0]]
                    #     # decoded = decode_captions(ground_truths, self.model.idx_to_word)
                    #     # for j, gt in enumerate(decoded):  # 列举出data对应的所有captions
                    #     #     print("Ground truth %d: %s" % (j + 1, gt))
                    #     # gen_caps = sess.run(generated_captions, feed_dict)
                    #     # decoded = decode_captions(gen_caps, self.model.idx_to_word)
                    #     # print("Generated caption: %s" % decoded[0])
                    #
                    #     # kenn evatulate
                    #     # true_ = captions_batch
                    #     # pred_ = gen_caps
                    #     # eva_ = evaluate(true_, pred_)
                    #     # print("test 1:", true_.shape, pred_.shape)        # test 1: (128, 5) (128, 4)
                    #     # print("Train Acc at epoch %d & iteration %d (mini-batch): %.5f\n" % (e + 1, i + 1, eva_))
                if (e + 1) % self.print_every == 0:
                    print("################ Epoch %d ################" % (e + 1))
                    print("Previous epoch loss: ", prev_loss)
                    print("Current epoch loss: ", curr_loss)
                    print("Elapsed time: ", time.time() - start_t)
                    if self.bool_val:
                        val_data = data[val_idxs]
                        val_pred = sess.run(generated_captions,
                                            feed_dict={self.model.data: val_data, self.model.captions: val_caps})
                        val_acc = evaluate(val_caps, val_pred)
                        val_acc_ = evaluate_(val_caps, val_pred)
                        print("Epoch {} validated Acc: \n "
                              "Total:{} & Element-wise:{}".format(e + 1, val_acc, val_acc_))

                        for _ in range(10):
                            n = np.random.randint(val_data.shape[0])
                            val_true_decode = decode_captions(val_caps[n], self.model.idx_to_word)
                            val_pred_decode = decode_captions(val_pred[n], self.model.idx_to_word)
                            print("Val data no.{} Ground-True:{} Pred:{}".format(n, val_true_decode, val_pred_decode))

                    prev_loss = curr_loss
                    curr_loss = 0

                    if val_acc > self.ini_acc:
                        self.ini_acc = val_acc
                        self.bool_save_this = True
                    # save model's parameters
                    if self.bool_save_this and self.bool_save_model and e > 10 or e == self.n_epochs - 1:
                        saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e + 1)
                        print("model-%s saved." % (e + 1))
                        self.bool_save_this = False

    def test(self, attention_visualization=True):
        """
        Args: - data: dictionary with the following keys: - features: Feature vectors of shape (5000, 196,
        512) - file_names: Image file names of shape (5000, ) - captions: Captions of shape (24210, 17) - image_idxs:
        Indices for mapping caption to image of shape (24210, ) - features_to_captions: Mapping feature to captions (
        5000, 4~5) - split: 'train', 'val' or 'test' - attention_visualization: If True, visualize attention weights
        with images for each sampled word. (ipthon notebook) - save_sampled_captions: If True, save sampled captions
        to pkl file for computing BLEU scores.
        """

        # features = data['features']
        data = self.data['data']
        print("data", data.shape)
        captions = self.data['captions']
        print(captions.shape)
        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=self.generated_caption_len)
        # (N, max_len, L), (N, max_len)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY
        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            # features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
            feed_dict = {self.model.data: data}
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)
            de_true = decode_captions(captions, self.model.idx_to_word)
            print("Test Acc:{} ### {}".format(evaluate(captions, sam_cap), evaluate_(captions, sam_cap)))
            if attention_visualization:
                for _ in range(10):
                    n = np.random.randint(data.shape[0])
                    print("No.%d: True Caption: %s, Sampled Caption: %s" % (n, de_true[n], decoded[n]))

                    plt.subplot(5, 2, (1, 2))
                    plot_data(data[n], "No.%d: True: %s, Pred: %s" % (n, de_true[n], decoded[n]))

                    words_ = decoded[n].split(" ")
                    if len(words_) > 4:
                        words = [words_[0], words_[2], words_[4], words_[5]]
                    else:
                        words = words_
                    arr_ = [3, 4, 7, 8]
                    for t in range(len(words)):
                        if t > 6:
                            break
                        plt.subplot(5, 2, arr_[t])
                        if self.bool_selector:
                            plot_data(data[n], '%s(%.2f)' % (words[t], bts[n, t]))
                        else:
                            plot_data(data[n], '%s' % (words[t]))
                        plt.axis('off')
                        plt.subplot(5, 2, arr_[t] + 2)
                        alp_curr = alps[n, t, :]
                        plt.bar(np.arange(len(alp_curr)), alp_curr)
                        # plt.plot(alp_curr)
                        plt.axis('off')
                    plt.show()

            # if save_sampled_captions:
            #     all_sam_cap = np.ndarray((features.shape[0], 20))
            #     num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
            #     for i in range(num_iter):
            #         features_batch = features[i * self.batch_size:(i + 1) * self.batch_size]
            #         feed_dict = {self.model.features: features_batch}
            #         all_sam_cap[i * self.batch_size:(i + 1) * self.batch_size] = sess.run(sampled_captions, feed_dict)
            #     all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
            #     save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" % (split, split))
