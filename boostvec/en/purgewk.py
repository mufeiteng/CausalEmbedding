# -*- coding: utf-8 -*-
import spacy
from tools import project_source_path
import os
import codecs
import re
import multiprocessing
import sys
import nltk
from tqdm import tqdm


def purge_sent(s):
    s = re.sub(r'\(.*\)', '', s)
    return s


def judge_para_valid(s):
    if s.startswith('<doc') or s.startswith('/doc>') or s.startswith('curid=') or len(s) < 40:
        return False
    return True


def regex_split_en_para(paragraph):
    res = list()
    for s_str in paragraph.split('.'):
        if '?' in s_str:
            res.extend(s_str.split('?'))
        elif '!' in s_str:
            res.extend(s_str.split('!'))
        else:
            res.append(s_str)
    return [purge_sent(r) for r in res]


def multicore_split_wiki_file(seq, l, file_list):
    while not file_list.empty():
        l.acquire()
        name = file_list.get()
        l.release()
        print('Process {} is processing {} now.'.format(seq, name))
        fin = codecs.open(os.path.join(params['input_path'], name), 'r', 'utf-8')
        fout = codecs.open(os.path.join(params['output_path'], name), 'w', 'utf-8')
        paragraphs = fin.readlines()
        for paragraph in paragraphs:
            # sents = self.splitter.tokenize(paragraph)
            # # sents = self.regex_split_en_para(paragraph)
            # for sent in sents:
            #     if len(sent) > 40:
            #         fout.write(purge_sent(sent.strip()) + '\n')
            if not judge_para_valid(paragraph):
                continue
            fout.write(purge_sent(paragraph) + '\n')


def split_wiki_paragraph():
    file_names = os.listdir(params['input_path'])
    input_file_list = multiprocessing.Queue()
    for file_name in file_names:
        if os.path.isfile(os.path.join(params['input_path'], file_name)):
            input_file_list.put(file_name)
    lock = multiprocessing.Lock()
    thread_list = []
    for i in range(params['num_threads']):
        sthread = multiprocessing.Process(target=multicore_split_wiki_file, args=(str(i + 1), lock, input_file_list))
        thread_list.append(sthread)
    for th in thread_list:
        th.start()
    for th in thread_list:
        th.join()


def purge_wiki_line():
    print('threshold: {}.'.format(params['thres']))
    spacy_nlp = spacy.load('en_core_web_sm')
    # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    file_names = os.listdir(params['input_path'])
    file_names = [filename for filename in file_names if params['thres'][0] <= int(filename.split('_')[-1]) < params['thres'][1]]
    for filename in tqdm(file_names):
        fin = codecs.open(os.path.join(params['input_path'], filename), 'r', 'utf-8')
        fout = codecs.open(os.path.join(params['output_path'], filename), 'w', 'utf-8')
        document = fin.read()
        doc_set = spacy_nlp(document)
        count = 0
        for _doc in tqdm(doc_set.sents):
            s = re.sub(r"\(.*\)", '', _doc.text)
            s = re.sub(r"\[.*\]", '', s)
            s = re.sub("[\n<>\"|/?(){}@!%&*_+=“”．，；＇]", '', s)
            s = re.sub(' +', ' ', s)
            if len(s) > 50:
                fout.write(s + '\n')


if __name__ == '__main__':

    params = {
        'input_path': os.path.join(project_source_path, 'en_wiki/raw/'),
        'output_path': os.path.join(project_source_path, 'en_wiki/temp/'),
        'thres': [33000, 36000],
        'num_threads': 1,
    }
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            key = arg.split("=")[0][2:]
            val = arg.split("=")[1]
            params[key] = val

    # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    split_wiki_paragraph()
    # purge_wiki_line()

