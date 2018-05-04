# -*- encoding: utf-8 -*-

import os, sys
from io import open
from tokenizer.tokenizer import Tokenizer
import utils
import unicodedata
import regex
import my_map
from pyvi.pyvi import ViPosTagger


dataset = 'dataset/train'
tokenized_dataset = 'dataset/train_tokenized'
# dataset = 'dataset/test'
# tokenized_dataset = 'dataset/test_tokenized'

tokenizer = Tokenizer()
r = regex.regex()


def tokenizer_dataset():
    utils.mkdir(tokenized_dataset)
    stack = os.listdir(dataset)
    print 'loading data in ' + dataset
    while (len(stack) > 0):
        file_name = stack.pop()
        file_path = os.path.join(dataset, file_name)
        if (os.path.isdir(file_path)):
            utils.push_data_to_stack(stack, file_path, file_name)
        else:
            print('\r%s' % (file_path)),
            sys.stdout.flush()
            with open(file_path, 'r', encoding='utf-16') as fp:
                content = unicodedata.normalize('NFKC', fp.read())
                content = r.run(tokenizer.predict(content))
                dir_name = utils.get_dir_name(file_path)
                output_dir = os.path.join(tokenized_dataset, dir_name)
                utils.mkdir(output_dir)
                name = os.path.basename(file_path)
                with open(os.path.join(output_dir, name), 'w', encoding='utf-8') as fw:
                    fw.write(content)
    print('')



def count_tokens():
    print('count tokens...')
    statistic = {name : {} for name in my_map.name2label.keys()}
    stack = os.listdir(tokenized_dataset)
    print 'loading data in ' + dataset
    while (len(stack) > 0):
        file_name = stack.pop()
        file_path = os.path.join(tokenized_dataset, file_name)
        if (os.path.isdir(file_path)):
            utils.push_data_to_stack(stack, file_path, file_name)
        else:
            print('\r%s' % (file_path)),
            sys.stdout.flush()
            with open(file_path, 'r', encoding='utf-8') as fp:
                label = utils.get_dir_name(file_path)
                for sen in fp:
                    sen = sen.strip()
                    tag = ViPosTagger.postagging(sen)
                    tokens = [tag[0][i] for i in xrange(len(tag[0])) if tag[1][i] == u'N']
                    update_count_tokens(statistic, label, tokens)


def update_count_tokens(statistic, label, tokens):
    for token in tokens:
        try:
            statistic[label][token] += 1
        except:
            try:
                statistic[label].update({token : 1})
            except:
                statistic.update({label : {token : 1}})





if __name__ == '__main__':
    tokenizer_dataset()
    count_tokens()