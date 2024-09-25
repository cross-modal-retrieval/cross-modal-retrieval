from __future__ import print_function
import os, time, pickle
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from random import shuffle
import random
import sklearn.preprocessing
from sklearn.preprocessing import normalize
import numpy
from models.base_model import BaseModel, BaseModelParams, BaseDataIter
from models.flip_gradient import flip_gradient
from sklearn.metrics.pairwise import cosine_similarity

def Normalize(data):
    l = []
    for i in range(len(data)):
        m = data[i].mean()
        max = data[i].max()
        min = data[i].min()
        re = data[i]-m/(max-min)
        l.append(re)
    return np.array(l)

class DataIter(BaseDataIter):
    def __init__(self, batch_size):
        BaseDataIter.__init__(self, batch_size)
        self.num_train_batch = 0
        self.num_test_batch = 0

        with open('./data/train_v.pkl', 'rb') as f:
            self.train_video_feats = pickle.load(f, encoding='iso-8859-1')

        with open('./data/train_i.pkl', 'rb') as f:
            self.train_img_vecs = pickle.load(f, encoding='iso-8859-1')

        with open('./data/train_label.pkl', 'rb') as f:
            self.train_labels = pickle.load(f, encoding='iso-8859-1')

        with open('./data/test_v.pkl', 'rb') as f:
            self.test_video_feats = pickle.load(f, encoding='iso-8859-1')


        with open('./data/test_i.pkl', 'rb') as f:
            self.test_img_vecs = pickle.load(f, encoding='iso-8859-1')

        with open('./data/test_label.pkl', 'rb') as f:
            self.test_labels = pickle.load(f, encoding='iso-8859-1')


        mean_i = self.train_img_vecs.mean()
        self.train_video_feats = Normalize(self.train_video_feats)
        self.test_video_feats = Normalize(self.test_video_feats)
        self.train_img_vecs = (self.train_img_vecs - mean_i) / (self.train_img_vecs.max() - self.train_img_vecs.min())
        self.test_img_vecs = (self.test_img_vecs - mean_i) / (self.train_img_vecs.max() - self.train_img_vecs.min())
        self.num_train_batch = len(self.train_video_feats) // self.batch_size
        self.num_test_batch = len(self.test_video_feats) // self.batch_size

    def train_data(self):
        for i in range(self.num_train_batch):
            batch_video_feats = self.train_video_feats[i * self.batch_size: (i + 1) * self.batch_size]
            batch_img_vecs = self.train_img_vecs[i * self.batch_size: (i + 1) * self.batch_size]
            batch_labels = self.train_labels[i * self.batch_size: (i + 1) * self.batch_size]
            yield batch_video_feats, batch_img_vecs, batch_labels, i

    def test_data(self):
        for i in range(self.num_test_batch):
            batch_video_feats = self.test_video_feats[i * self.batch_size: (i + 1) * self.batch_size]
            batch_img_vecs = self.test_img_vecs[i * self.batch_size: (i + 1) * self.batch_size]
            batch_labels = self.test_labels[i * self.batch_size: (i + 1) * self.batch_size]
            yield batch_video_feats, batch_img_vecs, batch_labels, i


class ModelParams(BaseModelParams):
    def __init__(self):
        BaseModelParams.__init__(self)

        self.epoch = 200
        self.margin = 0.01
        self.alpha = 1e-2
        self.batch_size = 70
        self.visual_feat_dim = 4096
        self.word_vec_dim = 128

        self.lr_emb = 0.0001
        self.lr_domain = 0.0001
        self.top_k = [10,20,50,100]
        self.semantic_emb_dim = 70
        self.dataset_name = 'thumos14_dataset'
        self.model_name = 'i2v'
        self.model_dir = 'i2v_%d_%d_%d' % (self.visual_feat_dim, self.word_vec_dim, self.semantic_emb_dim)

        self.checkpoint_dir = './checkpoint'
        self.sample_dir = './samples'
        self.dataset_dir = './thumos14'
        self.log_dir = './logs'

    def update(self):
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)


class AdvCrossModalSimple(BaseModel):
    def __init__(self, model_params):
        BaseModel.__init__(self, model_params)
        self.data_iter = DataIter(self.model_params.batch_size)

        self.tar_video = tf.placeholder(tf.float32, [None,64,self.model_params.visual_feat_dim]) # 64
        self.tar_img = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.neg_img = tf.placeholder(tf.int32, [self.model_params.batch_size,])
        self.y = tf.placeholder(tf.int32, [self.model_params.batch_size, 18])  # 10
        self.y_single = tf.placeholder(tf.int32, [self.model_params.batch_size, 1])
        self.l = tf.placeholder(tf.float32, [])
        self.trans = tf.placeholder(tf.float32, [None,64,64])

        self.emb_matrix = self.matrix_embed(self.tar_video)  # batch_size*(100,64)
        self.emb_w = self.label_embed(self.tar_img)
        self.emb_v = self.visual_embed(self.emb_matrix[0])
        for i in range(1, self.model_params.batch_size):
            M = self.visual_embed(self.emb_matrix[i], reuse=True)
            self.emb_v = tf.concat([self.emb_v,M], axis=0)

        # triplet loss
        idata=[]
        for i in range(self.model_params.batch_size):
            em_w = self.emb_w[i]/ tf.norm(self.emb_w[i])
            x = tf.matmul(tf.reshape(em_w,[64,1]),tf.reshape(em_w,[1,64]))
            y = tf.concat([x[k] for k in range(64)],axis=0)
            idata.append(y)

        self.trans, tran2 = self.transpose(self.emb_matrix)
        tran2 = tf.stop_gradient(tran2)
        margin = self.model_params.margin
        self.triplet_loss = self.tripletloss(idata, self.trans /tf.norm(self.trans), margin)
        self.logits_w = self.label_classifier(self.emb_w)
        self.logits_v = self.label_classifier(self.emb_v, reuse=True)
        self.label_loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits_v))
        self.label_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits_w))
        self.trans_loss = tf.reduce_mean(tran2 - self.trans)
        self.emb_loss = 0.01 * self.label_loss1 + self.label_loss2 + 10 * self.triplet_loss + self.model_params.alpha * self.trans_loss
        self.emb_v_class = self.domain_classifier(self.emb_v, self.l)
        self.emb_w_class = self.domain_classifier(self.emb_w, self.l, reuse=True)

        all_emb_v = tf.concat([tf.ones([self.model_params.batch_size, 1]), tf.zeros([self.model_params.batch_size, 1])], 1)
        all_emb_w = tf.concat([tf.zeros([self.model_params.batch_size, 1]), tf.ones([self.model_params.batch_size, 1])], 1)
        self.domain_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_v_class, labels=all_emb_w) + \
                                 tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_w_class, labels=all_emb_v)

        self.domain_class_loss = tf.reduce_mean(self.domain_class_loss)

        self.t_vars = tf.trainable_variables()
        self.vf1_vars = [v for v in self.t_vars if 'vf1_' in v.name]
        self.vf_vars = [v for v in self.t_vars if 'vf_' in v.name]
        self.le_vars = [v for v in self.t_vars if 'le_' in v.name]
        self.dc_vars = [v for v in self.t_vars if 'dc_' in v.name]
        self.lc_vars = [v for v in self.t_vars if 'lc_' in v.name]
        self.rf_vars = [v for v in self.t_vars if 'rf_' in v.name]

    def transpose(self, X):
        net = tf.nn.tanh(slim.fully_connected(X, 200, activation_fn=None, scope='rf_fc_0'))
        net = tf.nn.tanh(slim.fully_connected(net, 100, activation_fn=None, scope='rf_fc_1'))
        T = tf.nn.tanh(slim.fully_connected(net, 64, activation_fn=None, scope='rf_fc_2'))
        T1 = tf.transpose(X, [0, 2, 1])
        T2 = tf.matrix_inverse(tf.matmul(T1, X))
        T3 = tf.matmul(tf.matmul(X, T2),T1)
        return T, T3

    def tripletloss(self, idata, X, margin):
        v_l = []
        for j in range(self.model_params.batch_size):
            data_v_pos = tf.concat([ X[j][k] for k in range(64)], axis=0)
            data_v_neg = tf.concat([ X[self.neg_img[j]][k] for k in range(64)], axis=0)
            emb_w = idata[j]
            ap_v = tf.matmul(tf.reshape(data_v_pos, [1, 4096]),  tf.reshape(emb_w, [4096, 1]))
            an_v = tf.matmul(tf.reshape(data_v_neg, [1, 4096]), tf.reshape(emb_w, [4096, 1]))
            v_l.append(tf.maximum(an_v - ap_v + margin, 0))
        return tf.reduce_sum(v_l)

    def matrix_embed(self, X, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(X,500, scope='vf_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net,400, scope='vf_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, 200, scope='vf_fc_2'))
            net = tf.nn.tanh(slim.fully_connected(net, 64, scope='vf_fc_3'))
        return net

    def visual_embed(self, X, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            X1 = tf.nn.tanh(slim.fully_connected(X, 200, scope='vf1_fc_0'))
            X2 = slim.fully_connected(X1, 64, scope='vf1_fc_1')
            net1 = slim.fully_connected(X1, 1, scope='vf1_fc_2')
            A = tf.transpose(net1, [1, 0])
            A = tf.nn.softmax(A)
            M = tf.matmul(A, X2)
        return M

    def label_embed(self, L, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(L, 200, scope='le_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 100,scope='le_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, 80,scope='le_fc_2'))
            net = tf.nn.tanh(slim.fully_connected(net, 64, scope='le_fc_3'))
        return net

    def label_classifier(self, X, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = slim.fully_connected(X, 18, scope='lc_fc_3')
        return net
        
    def domain_classifier(self, E, l, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            E = flip_gradient(E, l)
            net = slim.fully_connected(E, 32, scope='dc_fc_0')
            net = slim.fully_connected(net, 16, scope='dc_fc_1')
            net = slim.fully_connected(net, 2, scope='dc_fc_2')
        return net

    def train(self, sess):
        emb_train_op = tf.train.AdamOptimizer(learning_rate = self.model_params.lr_emb, beta1=0.7).minimize(self.emb_loss, var_list = self.le_vars+self.vf_vars + self.vf1_vars + self.rf_vars)
        domain_train_op = tf.train.AdamOptimizer(learning_rate = self.model_params.lr_domain, beta1=0.7).minimize(self.domain_class_loss, var_list = self.dc_vars)

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()
        start_time = time.time()

        for epoch in range(self.model_params.epoch):
            p = float(epoch) / self.model_params.epoch
            l = 2 / (1. + np.exp(-10. * p)) - 1
            for batch_feat, batch_vec, batch_labels, idx in self.data_iter.train_data():
                # create one-hot labels
                batch_labels_ = batch_labels - np.ones_like(batch_labels)
                label_binarizer = sklearn.preprocessing.LabelBinarizer()
                label_binarizer.fit(range(18))
                b = label_binarizer.transform(batch_labels_)  # b is the one-hot label
                adj_mat = np.dot(b, np.transpose(b))
                mask_mat = np.ones_like(adj_mat) - adj_mat
                img_sim_mat = mask_mat * cosine_similarity(batch_vec, batch_vec)
                img_neg_idx = np.argmax(img_sim_mat, axis=1).astype(int)

                sess.run([emb_train_op, domain_train_op],
                         feed_dict={self.tar_video: batch_feat,
                                    self.tar_img: batch_vec,
                                    self.neg_img: img_neg_idx,
                                    self.y: b,
                                    self.y_single: np.transpose([batch_labels]),
                                    self.l: l})

                video, img, emb_v, emb_w, v, w, y, triplet_loss_val = sess.run([self.tar_video,self.tar_img,self.emb_v,self.emb_w,self.logits_v,self.logits_w,self.y,self.triplet_loss],
                    feed_dict={self.tar_video: batch_feat,
                               self.tar_img: batch_vec,
                               self.neg_img: img_neg_idx,
                               self.y: b,
                               self.y_single: np.transpose([batch_labels]),
                               self.l: l})
                print('Epoch: [%2d][%4d/%4d] time: %4.4f, triplet_loss: %.8f' % (epoch, idx, self.data_iter.num_train_batch, time.time() - start_time, triplet_loss_val))


    def eval(self,sess):
        start = time.time()
        test_video_feats_trans = []
        test_img_vecs_trans = []
        test_labels = []
        for feats, vecs, labels, i in self.data_iter.test_data():
            feats_trans = sess.run(self.emb_v, feed_dict={self.tar_video: feats})
            vecs_trans = sess.run(self.emb_w, feed_dict={self.tar_img: vecs})
            test_labels = np.append(test_labels,labels)
            for ii in range(len(feats)):
                test_video_feats_trans.append(feats_trans[ii])
                test_img_vecs_trans.append(vecs_trans[ii])
        test_video_feats_trans = np.asarray(test_video_feats_trans)
        test_img_vecs_trans = np.asarray(test_img_vecs_trans)

        top_k = self.model_params.top_k
        avg_precs = []

        for k in top_k:
            for i in range(len(test_img_vecs_trans)):
                query_label = test_labels[i]

                # distances and sort by distances
                wv = test_img_vecs_trans[i]
                diffs = test_video_feats_trans/np.linalg.norm(test_video_feats_trans)- wv/np.linalg.norm(wv)
                dists = np.linalg.norm(diffs, axis=1)
                sorted_idx = np.argsort(dists)

                # for each k do top-k
                precs = []
                for topk in range(1, k + 1):
                    hits = 0
                    top_k = sorted_idx[0: topk]
                    if np.sum(query_label) != test_labels[top_k[-1]]:
                        continue
                    for ii in top_k:
                        retrieved_label = test_labels[ii]
                        if np.sum(retrieved_label) == query_label:
                            hits += 1
                    precs.append(float(hits) / float(topk))
                if len(precs) == 0:
                    precs.append(0)
                avg_precs.append(np.average(precs))
            mean_avg_prec = np.mean(avg_precs)
            print('[Eval - video2video] mAP: %d top %f in %4.4fs' % (k, mean_avg_prec, (time.time() - start)))




