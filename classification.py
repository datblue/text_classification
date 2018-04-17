# -*- encoding: utf-8 -*-

import utils
import os
import my_map
import preprocessing
from collections import Counter
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class classification:
    def __init__(self):
        self.model = None
        self.vectorizer = None


    def load(self, model):
        print('loading %s ...' % (model))
        if os.path.isfile(model):
            return joblib.load(model)
        else:
            return None


    def load_model(self):
        self.vectorizer = self.load('model/vectorizer.pkl')
        self.model = self.load('model/model.pkl')


    def save(self, model, path):
        print('saving %s ...' % (path))
        utils.mkdir('model')
        joblib.dump(model, path)
        return


    def save_model(self):
        utils.mkdir('model')
        self.save(self.vectorizer, 'model/vectorizer.pkl')
        self.save(self.model, 'model/model.pkl')


    def feature_extraction(self, X):
        if self.vectorizer == None:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.6, min_df=1)
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
        samples_train = preprocessing.load_dataset(data_train)
        X_train, y_train = self.prepare_data(samples_train)
        X_train = self.feature_extraction(X_train)
        self.fit(X_train, y_train)

        samples_test = preprocessing.load_dataset(data_test)
        X_test, y_test = self.prepare_data(samples_test)
        X_test = self.feature_extraction(X_test)
        self.evaluation(X_test, y_test)
        self.save_model()


    def fit(self, X, y):
        print('fit model...')
        self.model = SVC(C=100, kernel='linear', class_weight='balanced')
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



if __name__ == '__main__':
    c = classification()
    c.run('dataset/train', 'dataset/test')