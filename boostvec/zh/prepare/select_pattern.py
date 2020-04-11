from tools import project_source_path, WordCounter
import os
import codecs
from extraction.cn.parser import CnParseTree, LTPModel
import multiprocessing
import sys


class Handler(object):
    def __init__(self, ptree):
        self.ptree = ptree

    def process(self, filename):
        fin = codecs.open(os.path.join(params['input_path'], filename), 'r', 'utf-8')
        fout = codecs.open(os.path.join(params['output_path'], filename), 'w', 'utf-8')
        counter = WordCounter()
        line = fin.readline().strip()
        while line:
            try:
                sents = self.ptree.split_sentence(line, True)
                for sent in sents:
                    try:
                        self.ptree.create(sent)
                        words = [node.word for node in self.ptree.tree]
                        postags = [node.postag for node in self.ptree.tree]
                        for i in range(len(words)):
                            if postags[i] == 'v':
                                node, tag = self.ptree.tree[i], False
                                for child in node.children:
                                    if self.ptree.tree[child].relation == 'VOB':
                                        tag = True
                                        break
                                if tag:
                                    counter.add(words[i])
                    except Exception:
                        pass
            except Exception:
                pass
            line = fin.readline().strip()
        for k in counter:
            fout.write('{} {}\n'.format(k, counter[k]))


def process_file(seq, l, file_list):
    ptree = CnParseTree(ltp_model)
    while not file_list.empty():
        l.acquire()
        name = file_list.get()
        l.release()
        print('Process {} is processing {} now.'.format(seq, name))
        handler = Handler(ptree=ptree)
        handler.process(name)


if __name__ == '__main__':

    params = {
        'input_path': os.path.join(project_source_path, 'sg_data/raw'),
        'output_path': os.path.join(project_source_path, 'boostvec/zh/prepare/data2pa/'),
        'num_threads': 16,
    }
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            key = arg.split("=")[0][2:]
            val = arg.split("=")[1]
            params[key] = val

    ltp_model = LTPModel()
    items = os.listdir(params['input_path'])
    input_file_list = multiprocessing.Queue()
    for item in items:
        if os.path.isfile(os.path.join(params['input_path'], item)):
            input_file_list.put(item)
    lock = multiprocessing.Lock()
    thread_list = []
    for i in range(params['num_threads']):
        sthread = multiprocessing.Process(target=process_file, args=(str(i + 1), lock, input_file_list))
        thread_list.append(sthread)
    for th in thread_list:
        th.start()
    for th in thread_list:
        th.join()

    final_dict = dict()
    names = os.listdir(params['output_path'])
    for name in names:
        lines = codecs.open(os.path.join(params['output_path']), name).readlines()
        for line in lines:
            k, v = line.strip().split(' ')
            count = int(v)
            if k not in final_dict:
                final_dict[k] = count
            else:
                final_dict[k] += count
    res = sorted(final_dict.items(), key=lambda x: x[1], reverse=True)
    fout = codecs.open(os.path.join(project_source_path, 'boostvec/zh/prepare/allpattern.txt'), 'w', 'utf-8')
    for k, v in res:
        fout.write('{} {}\n'.format(k, v))
