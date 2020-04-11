# -*- coding: utf-8 -*-
import sys
from extraction.cn.parser import CnParseTree, LTPModel
from extraction.cn.canrules import *
from extraction.cn.causalrules import *
from extraction.cn.negrules import *
import multiprocessing


class Extractor(object):
    def __init__(self, ptree, file_name):
        self.ptree, self.filename = ptree, file_name

    def extract(self, func, sent):
        self.ptree.create(sent)
        results, response = func(ptree=self.ptree), []
        for res in results:
            cause, effect, cue = ' '.join(res['left']), ' '.join(res['right']), res['cue']
            pos_cause, pos_effect = ' '.join(res['left_pos']), ' '.join(res['right_pos'])
            response.append('----'.join([sent, cause, effect, pos_cause, pos_effect, cue]))
        return response

    def start(self):
        fin = codecs.open(os.path.join(params['input_path'], self.filename), 'r', 'utf-8')
        fout = codecs.open(os.path.join(params['output_path'], self.filename), 'w', 'utf-8')
        line = fin.readline()
        while line:
            sents = self.ptree.split_sentence(line)
            for sent in sents:
                if params['extract_type'] == 'candidates':
                    func = CandidateRules.extract
                elif params['extract_type'] == 'positives':
                    func = PositivesRules.extract
                else:
                    func = NegativeRules.extract
                try:
                    response = self.extract(func, sent)
                    for res in response:
                        fout.write(res+'\n')
                except Exception:
                    pass
            line = fin.readline()


def process_file(seq, l, file_list):
    ptree = CnParseTree(ltp_model)
    while not file_list.empty():
        l.acquire()
        name = file_list.get()
        l.release()
        print('Process {} is processing {} now.'.format(seq, name))
        extractor = Extractor(ptree=ptree, file_name=name)
        extractor.start()


if __name__ == '__main__':

    ltp_model = LTPModel()
    corpus_type = 'bk_data'
    samples_type = 'positives'  # samples_type in ['positives', 'negatives', 'candidates']
    params = {
        'input_path': project_source_path + '{}/raw/'.format(corpus_type),
        'output_path': project_source_path + '{}/{}/'.format(corpus_type, samples_type),
        'extract_type': samples_type,
        'num_threads': 7
    }
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            key = arg.split("=")[0][2:]
            val = arg.split("=")[1]
            params[key] = val

    print('boost {} in {}'.format(params['extract_type'], params['output_path']))

    items = os.listdir(params['input_path'])
    input_file_list = multiprocessing.Queue()
    for item in items:
        if os.path.isfile(os.path.join(params['input_path'], item)):
            input_file_list.put(item)

    lock = multiprocessing.Lock()

    thread_list = []
    for i in range(params['num_threads']):
        sthread = multiprocessing.Process(target=process_file, args=(str(i+1), lock, input_file_list))
        thread_list.append(sthread)
    for th in thread_list:
        th.start()
    for th in thread_list:
        th.join()

