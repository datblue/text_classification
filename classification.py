# -*- encoding: utf-8 -*-

import utils
import os
import my_map
import preprocessing
from sklearn.externals import joblib
from io import open
import network
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
import sentence_embedding as sen_emb



class classification:
    def __init__(self, root_dir='.'):
        self.model = None
        self.root_dir = root_dir
        self.result_dir = os.path.join(self.root_dir, 'result')
        self.max_sentences = 50
        self.patience = 3


    def load(self, model):
        print('loading %s ...' % (model))
        if os.path.isfile(model):
            return joblib.load(model)
        else:
            return None


    def load_model(self):
        try:
            self.model = load_model('model/model.h5')
        except:
            self.model = None


    def load_training_vector(self):
        X_train = self.load('model/X_train.pkl')
        y_train = self.load('model/y_train.pkl')
        X_val = self.load('model/X_val.pkl')
        y_val = self.load('model/y_val.pkl')
        return X_train, y_train, X_val, y_val


    def load_testing_vector(self):
        X_test = self.load('model/X_test.pkl')
        y_test = self.load('model/y_test.pkl')
        return X_test, y_test


    def save(self, model, path):
        print('saving %s ...' % (path))
        utils.mkdir('model')
        joblib.dump(model, path, compress=True)
        return


    def save_model(self):
        utils.mkdir('model')
        self.model.save('model/model.h5')


    def save_training_vector(self, X_train, y_train, X_val, y_val):
        utils.mkdir('model')
        self.save(X_train, 'model/X_train.pkl')
        self.save(y_train, 'model/y_train.pkl')
        self.save(X_val, 'model/X_val.pkl')
        self.save(y_val, 'model/y_val.pkl')


    def save_testing_vector(self, X_test, y_test):
        utils.mkdir('model')
        self.save(X_test, 'model/X_test.pkl')
        self.save(y_test, 'model/y_test.pkl')


    def split_validation(self, samples_train):
        samples_val = {}
        for cat in samples_train:
            samples = samples_train[cat]
            boundary = int(round(0.9 * len(samples)))
            samples_val.update({cat : samples[boundary : ]})
            samples_train[cat] = samples[: boundary]
        return samples_val


    def training(self, data_train, data_test):
        n_labels = len(my_map.label2name)
        X_train, y_train, X_val, y_val = self.load_training_vector()
        if X_train is None or y_train is None \
                or X_val is None or y_val is None:
            samples_train = preprocessing.load_dataset_from_disk(
                data_train, max_sentences=self.max_sentences)
            samples_val = self.split_validation(samples_train)

            X_train, y_train = sen_emb.create_vector_data(samples_train, self.max_sentences)
            X_val, y_val = sen_emb.create_vector_data(samples_val, self.max_sentences)

            y_train = utils.convert_list_to_onehot(y_train, n_labels)
            y_val = utils.convert_list_to_onehot(y_val, n_labels)
            self.save_training_vector(X_train, y_train, X_val, y_val)
        self.fit(X_train, y_train, X_val, y_val)

        X_test, y_test = self.load_testing_vector()
        if X_test is None or y_test is None:
            samples_test = preprocessing.load_dataset_from_disk(
                data_test, max_sentences=self.max_sentences)
            X_test, y_test = sen_emb.create_vector_data(samples_test, self.max_sentences)
            self.save_testing_vector(X_test, y_test)
        self.evaluation(X_test, y_test)
        # self.save_model()


    def fit(self, X_train, y_train, X_val, y_val):
        print('build model...')
        # build network
        num_lstm_layer = 1
        num_hidden_node = 64
        dropout = 0.0
        embedding_size = sen_emb.emb_size
        self.model = network.building_network(embedding_size,
                                              num_lstm_layer, num_hidden_node, dropout,
                                              self.max_sentences,
                                              len(my_map.label2name))
        print 'Model summary...'
        print self.model.summary()
        print 'Training model...'
        early_stopping = EarlyStopping(patience=self.patience)
        self.model.fit(X_train, y_train, batch_size=32, epochs=100,
                       validation_data=(X_val, y_val),
                       callbacks=[early_stopping])


    def evaluation(self, X, y):
        y_pred = self.model.predict_classes(X, batch_size=32)
        accuracy = accuracy_score(y, y_pred)
        print('Accuracy = %.5f' % (accuracy))
        confusion = confusion_matrix(y, y_pred)
        print(confusion)


    def run(self, data_train, data_test):
        self.load_model()
        if self.model == None:
            self.training(data_train, data_test)


    def predict(self, list_document):
        pass


    def save_to_dir(self, list_document, labels):
    	utils.mkdir(self.result_dir)
        _ = map(lambda x: utils.mkdir(os.path.join(self.result_dir, x)), my_map.name2label.keys())
        for i in xrange(len(labels)):
            output_dir = os.path.join(self.result_dir, my_map.label2name[labels[i]])
            with open(os.path.join(output_dir, utils.id_generator()), 'w', encoding='utf-8') as fw:
                fw.write(unicode(list_document[i]))




if __name__ == '__main__':
    c = classification()
    c.run('dataset/train', 'dataset/test')