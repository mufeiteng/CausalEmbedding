# -*- coding: utf-8 -*-
import codecs
import os
from tqdm import tqdm
from msworks.tools import project_source_path


if __name__ == '__main__':
    corpus_name, samples_type = 'bk', 'positives'
    corpus_type = '{}_data'.format(corpus_name)
    pair_file_path = os.path.join(project_source_path, '{}/{}/'.format(corpus_type, samples_type))
    # pair_file_path = '/home/feiteng/Music/bk_verb_negatives/'
    pair_files = os.listdir(pair_file_path)
    all_extracted_pairs_file = os.path.join(project_source_path, '{}/temp.txt'.format(corpus_type))
    with codecs.open(all_extracted_pairs_file, 'w', 'utf-8') as fout:
        for file_name in tqdm(pair_files):
            file_path_name = os.path.join(pair_file_path, file_name)
            if os.path.isfile(file_path_name):
                with codecs.open(file_path_name, 'r', 'utf-8') as fin:
                    for line in fin:
                        line = line.strip()
                        if line:
                            fout.write(line + '\n')
    fout.close()
    fin = codecs.open(all_extracted_pairs_file, 'r', 'utf-8')
    fout = codecs.open(os.path.join(project_source_path, '{}/{}_{}.txt'.format(corpus_type, corpus_name, samples_type)), 'w', 'utf-8')
    line = fin.readline()
    pre = ''
    while line:
        if line != pre:
            fout.write(line)
            pre = line
        line = fin.readline()
    fin.close()
    fout.close()
    os.remove(all_extracted_pairs_file)