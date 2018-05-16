# -*- encoding: utf-8 -*-

import regex
import os, sys
import my_map
import utils
from io import open
import unicodedata
from pyvi.pyvi import ViTokenizer
import nlp_tools as nlp


r = regex.regex()


def load_dataset(dataset):
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
                content = []
                for sen in nlp.spliter.split(unicodedata.normalize('NFKC', fp.read())):
                    sen = r.run(ViTokenizer.tokenize(sen))
                    content.append(sen)
                dir_name = utils.get_dir_name(file_path)
                list_samples[dir_name].append(u'\n'.join(content))
    print('')
    return list_samples


def load_dataset_ex(list_samples):
    result = []
    for sample in list_samples:
        sample = r.run(ViTokenizer.tokenize(sample))
        result.append(sample)
    return result




if __name__ == '__main__':
    load_dataset('dataset/train')