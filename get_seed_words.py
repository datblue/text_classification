# -*- encoding: utf-8 -*-

import os, sys
from io import open
from tokenizer.tokenizer import Tokenizer
import utils
import unicodedata
import regex


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
                content = r.run(unicodedata.normalize('NFKC', fp.read()))
                # content = unicodedata.normalize('NFKC', fp.read())
                content = tokenizer.predict(content)
                dir_name = utils.get_dir_name(file_path)
                output_dir = os.path.join(tokenized_dataset, dir_name)
                utils.mkdir(output_dir)
                name = os.path.basename(file_path)
                with open(os.path.join(output_dir, name), 'w', encoding='utf-8') as fw:
                    fw.write(content)
    print('')


if __name__ == '__main__':
    tokenizer_dataset()
