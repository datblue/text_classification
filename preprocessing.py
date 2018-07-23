# -*- encoding: utf-8 -*-

import regex
import os, sys
import my_map
import utils
from io import open
import unicodedata
from nlp_tools import tokenizer, spliter


r = regex.regex()


def get_max_sentences(dataset):
    max_sent = 0
    total_sent = []
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
                sentences = filter(lambda s: len(s) > 0, spliter.split(content))
                if len(sentences) > max_sent:
                    max_sent = len(sentences)
                total_sent.append(len(sentences))

    print('')
    print('max sentences = %d -- average sentences = %d' %
          (max_sent, sum(total_sent) / len(total_sent)))


def load_dataset_from_disk(dataset, max_sentences=100):
    samples = {k:[] for k in my_map.name2label.keys()}
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
                sentences = filter(lambda s: len(s) > 0, spliter.split(content))
                sentences = map(lambda s: r.run(tokenizer.predict(s)).lower(), sentences)
                dir_name = utils.get_dir_name(file_path)
                samples[dir_name].append(sentences[:max_sentences])
    print('')
    return samples


def load_dataset_from_list(list_samples, max_length):
    result = []
    for sample in list_samples:
        sentences = filter(lambda s: len(s) > 0, spliter.split(sample))
        sentences = map(lambda s: r.run(tokenizer.predict(s)).lower(), sentences)
        result.append(sentences[:max_length])
    return result




if __name__ == '__main__':
    get_max_sentences('dataset/train')