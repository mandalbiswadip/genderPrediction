# train the network and classify
from __future__ import print_function
from __future__ import unicode_literals

import re
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from . import paths
from .paths import alphabet_dict, class_dict, seq_model_dir_path

dict_model = None

def addseqcolumns(columns = []):
    for i in range(paths.max_sentence_length):
        columns.append('letter_' + str(i))
    return columns

def remove_non_alphabates(word = '', gender = 'male'):
    regex = r'[^a-zA-Z]'
    match = re.findall(regex, word)
    if match:
        if gender == 'male':
            return 'biswadip'
        elif gender == 'female':
            return 'dipa'
    else:
        return word


class RNN:

    def __init__(self, column="Name", class_column = 'class'):
        self.name_column = column
        self.class_column = class_column

    def inputdata(self, male_dataset_path = paths.male_name_path, female_dataset_path = paths.female_name_path):

        male_dataset_path = pd.read_csv(male_dataset_path)
        female_dataset_path = pd.read_csv(female_dataset_path)
        male_dataset_path[self.class_column] = 'male'
        male_dataset_path[self.name_column] = male_dataset_path[self.name_column].apply(lambda row: remove_non_alphabates(row, 'male'))
        female_dataset_path[self.class_column] = 'female'
        female_dataset_path[self.name_column] = female_dataset_path[self.name_column].apply(lambda row: remove_non_alphabates(row, 'female'))

        self.dataset = male_dataset_path.append(female_dataset_path)
        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset)
        self.dataset[self.name_column] = self.dataset[self.name_column].apply(lambda row: row.strip())
        self.dataset[self.name_column] = self.dataset[self.name_column].apply(lambda row: row.lower())

        self.query_list = self.dataset[self.name_column]

    def input_single_data(self, word):
        word = word.lower()
        vec = []
        for i in range(paths.max_sentence_length):
            vec.append(self.char_embedding(word, i))
        self.test_x = [vec]
        self.num_classes = 2

    def char_embedding(self, word='abcd', index=0):
        vec_size = len(alphabet_dict)
        vec = [0]*vec_size
        if index < len(word):
            vec[alphabet_dict[word[index]]] = 1
        return np.array(vec)

    def class_embed(self, class_name='male'):
        vec = [0]*len(class_dict)
        vec[class_dict[class_name]] = 1
        return np.array(vec)

    def build_features(self):

        for i in range(paths.max_sentence_length):
            self.dataset['letter_' + str(i)] = self.dataset[self.name_column].apply(lambda row: self.char_embedding(row, i))
        self.dataset[self.class_column] = self.dataset[self.class_column].apply(lambda row: self.class_embed(row))

    def build_RNN_columns(self):
        self.seq_columns = []
        self.seq_columns = addseqcolumns(self.seq_columns)

    def build_dataset_RNN(self, split=0.9):
        self.train_len = int(split * self.dataset.shape[0])
        b = self.dataset[self.seq_columns].values
        m = []
        for i in range(b.shape[0]):
            row = []
            for j in range(b.shape[1]):
                row.append(b[i][j])
            m.append(row)
        self.seq_features = np.array(m)
        c = self.dataset[self.class_column].values
        n = []
        for i in range(c.shape[0]):
            row = []
            for j in range(2):
                row.append(c[i][j])
            n.append(row)
        self.class_dataset = np.array(n)
        print("seq features shape..", self.seq_features.shape)
        print("output shape..", self.class_dataset.shape)
        self.num_classes = len(class_dict)
        self.train_x, self.test_x = self.seq_features[:self.train_len], self.seq_features[self.train_len:]
        self.train_y, self.test_y = self.class_dataset[:self.train_len], self.class_dataset[self.train_len:]

    def build_RNN_variables(self, num_hidden):
        self.x_rnn = tf.placeholder(tf.float32, [None, 20, 26])
        self.y_rnn = tf.placeholder(tf.float32, [None, 2])

        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True)

        (output_fw, output_bw), self.states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=self.x_rnn,
                                                                              dtype=tf.float32)
        self.output = tf.concat([output_fw, output_bw], axis=2)

        self.val = tf.transpose(self.output, [1, 0, 2])

        self.last = tf.gather(self.val, int(self.val.get_shape()[0]) - 1)

        self.weight = tf.Variable(tf.truncated_normal([2 * num_hidden, self.num_classes]))
        self.bias = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))

    def build_RNN_loss(self):
        # include rest of the features

        logit = tf.matmul(self.last, self.weight) + self.bias

        self.pred = tf.nn.softmax(logit)

        self.entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.y_rnn))
        # l2 = 0.005 * sum(
        #   tf.nn.l2_loss(tf_var)
        # for tf_var in tf.trainable_variables()
        # if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        # )
        # self.entropy+=l2

        self.learning_rate = tf.placeholder(tf.float32, shape=())

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   300, 0.5, staircase=True)

        self.minimize = tf.train.AdamOptimizer(learning_rate).minimize(loss=self.entropy, global_step=global_step)

        mistakes = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y_rnn, 1))
        self.accuracy = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    def build_RNN_config(self):
        # TODO use these method for RNN config
        self.batch_size = 1000
        self.epoch = 500
        self.rate = 0.1

    def run_RNN_session(self):
        with open(paths.loss_file,'w') as file:
            with tf.Session() as ses:
                ses.run(tf.global_variables_initializer())
                # coord = tf.train.Coordinator()
                # threads = tf.train.start_queue_runners(coord=coord)
                saver = tf.train.Saver()

                batch_size = 500
                epoch = 500

                rate = 0.1
                num_of_batches = int(self.train_len / batch_size)

                for i in range(epoch):
                    start_time = time()
                    ptr = 0
                    avg_loss = 0
                    if epoch > 250:
                        rate = 0.1
                    for j in range(num_of_batches):

                        a = ses.run(self.minimize,
                                    feed_dict={self.x_rnn: self.train_x[ptr: ptr + batch_size],
                                               self.y_rnn: self.train_y[ptr: ptr + batch_size]
                                               # self.learning_rate: rate
                                               })
                        loss = ses.run(self.entropy, feed_dict={self.x_rnn: self.train_x[ptr: ptr + batch_size],
                                                                    self.y_rnn: self.train_y[ptr: ptr + batch_size]})
                        avg_loss = avg_loss + loss
                        ptr += batch_size

                    # _, loss = ses.run([minimize, entropy], feed_dict={x: train_x, y: train_y})
                    print("At epoch {} time: {}".format(i, time() - start_time))
                    print("loss..", str(avg_loss / num_of_batches))
                    file.write("%s\n" % str(avg_loss / num_of_batches))
                    if i % 10 == 0:
                        accur = ses.run(self.accuracy, feed_dict={self.x_rnn: self.test_x,
                                                                  self.y_rnn: self.test_y})
                        print("accur", accur)
                    if i % 100 == 0:
                        saver.save(ses, seq_model_dir_path + "model.clpt", global_step=epoch)

                # print(ses.run(pred,feed_dict={x: test_x, y: test_y,other_features_x:other_features_test}))
                # print(test_y)
                print("train accuracy..",
                      ses.run(self.accuracy, feed_dict={self.x_rnn: self.train_x, self.y_rnn: self.train_y}))

                accur, prediction = ses.run([self.accuracy, self.pred], {self.x_rnn: self.test_x,
                                                                         self.y_rnn: self.test_y})
                # accur = ses.run(accuracy, {x: train_x, y: train_y})

                print("accur..", accur)

                print("saving session..")
                saver.save(ses, seq_model_dir_path + "model.ckpt")

                print("classification report..", classification_report(self.test_y, prediction))
                print("confusion matrix..", confusion_matrix(self.test_y, prediction))


    def rerun_session(self):
        with tf.Session() as ses:
            ses.run(tf.global_variables_initializer())
            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(coord=coord)
            saver = tf.train.Saver()
            saver.restore(ses, seq_model_dir_path + "model.ckpt")



            prediction = ses.run( self.pred, { self.x_rnn: self.test_x })
            # accur = ses.run(accuracy, {x: train_x, y: train_y})
            pred_prob = []
            for row in prediction:
                pred_dict = {}
                pred_dict['male'] = row[class_dict['male']]
                pred_dict['female'] = row[class_dict['female']]
                pred_prob.append(pred_dict)
        return pred_prob


    def train_RNN(self, num_hidden=20, split=0.9):
        self.build_RNN_columns()
        print("columns defined..")
        self.build_dataset_RNN(split=split)
        print("dataset built..")
        self.build_RNN_variables(num_hidden=num_hidden)
        print("variables defined")
        self.build_RNN_loss()
        print("loss...")
        self.rerun_session()

    def get_RNN_features(self, dataset=pd.DataFrame()):

        seq_features = []

        seq_features = addseqcolumns(seq_features)

        data = dataset[seq_features].values

        m = []
        for i in range(data.shape[0]):
            row = []
            for j in range(data.shape[1]):
                row.append(data[i][j])
            m.append(row)
        data = np.array(m)
        saver = tf.train.Saver()
        with tf.Session() as ses:
            print("restoring rnn model..")
            saver.restore(ses, seq_model_dir_path + "model.ckpt")
            last = ses.run(self.last, feed_dict={self.x_rnn: data})

            for i in range(last.shape[1]):
                temp = [vector[i] for vector in last]
                se = pd.Series(temp)
                dataset["rnn_feature_" + str(i)] = se.values

            return dataset, last.shape[1]
def train():

    trainer = RNN()
    trainer.inputdata()
    trainer.build_features()
    trainer.train_RNN()

def classify(word):
    word = word.lower()
    trainer = RNN()
    trainer.input_single_data(word)
    trainer.build_RNN_variables(num_hidden=20)
    trainer.build_RNN_loss()
    pred = trainer.rerun_session()
    return pred

if __name__=='__main__':
    classify('Pabitra')