# -*- coding: utf-8 -*-
import spacy
from spacy import displacy
from tools import project_source_path, sharp_causal_verb_set
from boostvec.en.prepare.parser import EnParseTree
import os
import codecs
import re
import sys
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from boostvec.en.prepare.extractor import EnExtractor


if __name__ == '__main__':
    params = {
        'input_path': os.path.join(project_source_path, 'en_wiki/raw/'),
        'output_path': os.path.join(project_source_path, 'boostvec/en/prepare/verb2pp/'),
        'thres': [6000, 7000]
    }
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            key = arg.split("=")[0][2:]
            val = arg.split("=")[1]
            params[key] = val

    print('threshold in 61: {}.'.format(params['thres']))
    spacy_nlp = spacy.load('en_core_web_sm')
    lemmatizer = WordNetLemmatizer()
    # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    extractor = EnExtractor(
        sent_splitter_model=None, ptree=EnParseTree(), stem_model=lemmatizer
    )
    file_names = os.listdir(params['input_path'])
    file_names = [filename for filename in file_names if params['thres'][0] <= int(filename.split('_')[-1]) < params['thres'][1]]
    for filename in tqdm(file_names):
        fin = codecs.open(os.path.join(params['input_path'], filename), 'r', 'utf-8')
        fout = codecs.open(os.path.join(params['output_path'], filename), 'w', 'utf-8')
        document = fin.read()
        doc_set = spacy_nlp(document)
        count = 0
        for _doc in tqdm(doc_set.sents):
            try:
                res = extractor.extract_from_doc(_doc)
                if len(res) == 0:
                    continue
                sent = _doc.sent
                for r in res:
                    fout.write(_doc.text.strip()+'----'+ r + '\n')
            except Exception:
                pass
