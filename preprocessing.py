# -*- encoding: utf-8 -*-

import regex
import os, sys
import my_map
import utils
from io import open
import unicodedata
from pyvi.pyvi import ViTokenizer
from nlp_tools import spliter


r = regex.regex()


def load_dataset_from_disk(dataset, max_length):
    list_samples = {k:[] for k in my_map.name2label.keys()}
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
                sentences = map(lambda s: r.run(ViTokenizer.tokenize(s)), sentences)
                content = u'\n'.join(sentences).lower()
                words = content.split()
                dir_name = utils.get_dir_name(file_path)
                list_samples[dir_name].append(words[:max_length])
    print('')
    return list_samples


def load_dataset_from_list(list_samples, max_length):
    result = []
    for sample in list_samples:
        sentences = filter(lambda s: len(s) > 0, spliter.split(sample))
        sentences = map(lambda s: r.run(ViTokenizer.tokenize(s)), sentences)
        sample = u'\n'.join(sentences).lower()
        words = sample.split()
        result.append(words[:max_length])
    return result




if __name__ == '__main__':
    load_dataset_from_disk('dataset/train', 500)