import sys
# sys.path.append('/home/aszzy/Public/')
from extraction.en.causalrules import *
from extraction.en.parser import *
from nltk.stem import WordNetLemmatizer
from tools import *
from allennlp.predictors.biaffine_dependency_parser import BiaffineDependencyParserPredictor
# from allennlp.predictors.constituency_parser import ConstituencyParserPredictor


def extract_ce_mention_impl(ptree, extractor, output_list, l):
    response = []
    while not output_list.empty():
        l.acquire()
        output, s = output_list.get()
        l.release()
        ptree.create(output)
        res = extractor.extract(ptree, s)
        if res is not None:
            cause, effect, cue = res
            cause_w, cause_p = [child.text for child in cause], [child.pos for child in cause]
            effect_w, effect_p = [child.text for child in effect], [child.pos for child in effect]
            res = '----'.join([s, ' '.join(cause_w), ' '.join(cause_p), ' '.join(effect_w), ' '.join(effect_p), cue])
            response.append(res)
    return response


def extract_verb_causality():
    lemmatizer = WordNetLemmatizer()
    parser_dir = os.path.join(project_source_path, 'allennlp/')
    dependency = BiaffineDependencyParserPredictor.from_path(os.path.join(parser_dir, 'dependency_parser.tar.gz'))
    dep_analyser = DependencyAnalyser(dependency, cuda=True)
    global_ptree = DependencyTree()
    intra_extractor = CausalVerbRules(lemmatizer)
    manager = multiprocessing.Manager()
    output_sync_list = manager.Queue()
    lock = manager.Lock()
    workers = 15
    thres = [0, 6]
    path = os.path.join(project_source_path, 'enwiki/process/')
    input_path = os.path.join(path, 'raw/')
    output_path = os.path.join(path, 'output/')
    files = os.listdir(input_path)
    for name in files:
        print('process file {}.'.format(name))
        pool = multiprocessing.Pool()
        fout = codecs.open(os.path.join(output_path, name), 'w', 'utf-8')
        fin = codecs.open(os.path.join(input_path, name), 'r', 'utf-8')
        lines = fin.readlines()
        batches = generate_batches(lines, 128)
        for batch in batches:
            outputs = dep_analyser.predict_batch(batch.tolist())
            start = time()
            count = 0
            # for idx, item in enumerate(outputs):
            #     output_sync_list.put((item, batch[idx].strip()))
            #
            # results = []
            # for i in range(workers):
            #     results.append(pool.apply_async(
            #         extract_ce_mention, (global_ptree, global_extractor, output_sync_list, lock)
            #     ))
            # pool.close()
            # pool.join()
            # for result in results:
            #     sents = result.get()
            #     for sent in sents:
            #         fout.write('{}\n'.format(sent))

            for idx, output in enumerate(outputs):
                global_ptree.create(output)
                sent = batch[idx].strip()
                res = intra_extractor.extract(global_ptree, sent)
                if res is not None:
                    cause, effect, cue = res
                    cause_w, cause_p = [child.text for child in cause], [child.pos for child in cause]
                    effect_w, effect_p = [child.text for child in effect], [child.pos for child in effect]
                    res = '----'.join([sent, ' '.join(cause_w), ' '.join(cause_p), ' '.join(effect_w), ' '.join(effect_p), cue])
                    fout.write('{}\n'.format(res))
                    count += 1
            end = time()
            print('batch use time {} s, count {}.'.format((end - start), count))


def extract_conj_causality():
    input_dir = os.path.join(project_source_path, 'enwiki/large_split/')
    output_dir = os.path.join(project_source_path, 'enwiki/extract_output/')
    threshold, num_threads = [0, 1000], 16
    items = os.listdir(input_dir)
    # items = [item for item in items if threshold[0] <= int(item) < threshold[1]]
    input_file_list = multiprocessing.Queue()
    for item in items:
        if os.path.isfile(os.path.join(input_dir, item)):
            input_file_list.put(item)

    lock = multiprocessing.Lock()
    thread_list = []
    for i in range(num_threads):
        sthread = multiprocessing.Process(target=process_file, args=(str(i + 1), lock, input_file_list))
        thread_list.append(sthread)
    for th in thread_list:
        th.start()
    for th in thread_list:
        th.join()


def process_file(seq, l, file_list):
    while not file_list.empty():
        l.acquire()
        name = file_list.get()
        l.release()
        print('Process {} is processing {} now.'.format(seq, name))
        fin = open(os.path.join(input_dir, name), 'r', encoding='gb18030', errors='ignore')
        fout = open(os.path.join(output_dir, name), 'w', encoding='gb18030', errors='ignore')
        prev = ''
        line = fin.readline().strip()
        while line:
            response = CausalConjRules.extract(prev, line)
            if len(response) > 0:
                for res in response:
                    arg1_tokens, arg2_tokens, cue, sent = res
                    arg1_words, arg1_postags = zip(*arg1_tokens)
                    arg2_words, arg2_postags = zip(*arg2_tokens)
                    s = '----'.join([sent, ' '.join(arg1_words), ' '.join(arg2_words), ' '.join(arg1_postags), ' '.join(arg2_postags), cue])
                    fout.write('{}\n'.format(s))
            prev = line
            line = fin.readline().strip()


if __name__ == '__main__':
    input_dir = os.path.join(project_source_path, 'enwiki/large_split/')
    output_dir = os.path.join(project_source_path, 'enwiki/train_dataset/')
    threshold, num_threads = [0, 1000], 16

    items = os.listdir(input_dir)
    # items = [item for item in items if threshold[0] <= int(item) < threshold[1]]

    input_file_list = multiprocessing.Queue()
    for item in items:
        if os.path.isfile(os.path.join(input_dir, item)):
            input_file_list.put(item)

    lock = multiprocessing.Lock()
    thread_list = []
    for i in range(num_threads):
        sthread = multiprocessing.Process(target=process_file, args=(str(i + 1), lock, input_file_list))
        thread_list.append(sthread)
    for th in thread_list:
        th.start()
    for th in thread_list:
        th.join()
