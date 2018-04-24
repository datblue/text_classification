# -*- encoding: utf-8 -*-

import utils
import os
import my_map
import preprocessing
from collections import Counter
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from io import open



class classification:
    def __init__(self, root_dir='.'):
        self.model = None
        self.vectorizer = None
        self.root_dir = root_dir
        self.result_dir = os.path.join(self.root_dir, 'result')


    def load(self, model):
        print('loading %s ...' % (model))
        if os.path.isfile(model):
            return joblib.load(model)
        else:
            return None


    def load_model(self):
        self.vectorizer = self.load('model/vectorizer.pkl')
        self.model = self.load('model/model.pkl')


    def load_training_vector(self):
        X_train = self.load('model/X_train.pkl')
        y_train = self.load('model/y_train.pkl')
        return X_train, y_train


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
        self.save(self.vectorizer, 'model/vectorizer.pkl')
        self.save(self.model, 'model/model.pkl')


    def save_training_vector(self, X_train, y_train):
        utils.mkdir('model')
        self.save(X_train, 'model/X_train.pkl')
        self.save(y_train, 'model/y_train.pkl')


    def save_testing_vector(self, X_test, y_test):
        utils.mkdir('model')
        self.save(X_test, 'model/X_test.pkl')
        self.save(y_test, 'model/y_test.pkl')


    def feature_extraction(self, X):
        if self.vectorizer == None:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.6, min_df=2)
            self.vectorizer.fit(X)
        return self.vectorizer.transform(X)


    def prepare_data(self, dataset):
        X = []; y = []
        for name, list_content in dataset.items():
            label = my_map.name2label[name]
            for content in list_content:
                y.append(label)
                X.append(content)
        return X, y


    def training(self, data_train, data_test):
        X_train, y_train = self.load_training_vector()
        if X_train == None or y_train == None:
            samples_train = preprocessing.load_dataset(data_train)
            X_train, y_train = self.prepare_data(samples_train)
            X_train = self.feature_extraction(X_train)
            self.save_training_vector(X_train, y_train)
        self.fit(X_train, y_train)

        X_test, y_test = self.load_testing_vector()
        if X_test == None or y_test == None:
            samples_test = preprocessing.load_dataset(data_test)
            X_test, y_test = self.prepare_data(samples_test)
            X_test = self.feature_extraction(X_test)
            self.save_testing_vector(X_test, y_test)
        self.evaluation(X_test, y_test)
        self.save_model()


    def fit(self, X, y):
        print('fit model...')
        self.model = LogisticRegressionCV(class_weight='balanced')
        self.model.fit(X, y)


    def evaluation(self, X, y):
        count = Counter(y)
        print(count)
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print('accuracy score = %.5f' % (accuracy))
        confusion = confusion_matrix(y, y_pred)
        print(confusion)


    def run(self, data_train, data_test):
        self.load_model()
        if self.model == None or self.vectorizer == None:
            self.training(data_train, data_test)


    def predict(self, list_document):
        docs = preprocessing.load_dataset_ex(list_document)
        X = self.feature_extraction(docs)
        return self.model.predict(X)


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