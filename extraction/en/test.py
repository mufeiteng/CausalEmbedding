# -*- coding: utf-8 -*-
import spacy
from msworks.tools import project_source_path
from boostvec.en.prepare.parser import EnParseTree
from extraction.en.causalrules import EnCausalRules
import os
import codecs
import re
import sys
from nltk.stem import WordNetLemmatizer
"""
The event also had the secondary purpose of memorialising workers killed as a result of the Haymarket affair.
The Commune was the result of an uprising in Paris after France was defeated in the Franco-Prussian War.

"""

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
    ptree = EnParseTree()
    # tokenizer = nltk.graph.load('tokenizers/punkt/english.pickle')
    causalRules = EnCausalRules(stemmer=lemmatizer)
    fin = codecs.open(os.path.join(project_source_path, 'en_wiki/en_wiki_test'), 'r', 'utf-8')
    key_words = '(.+) (?:cause|causes|caused|causing|lead|leads|led|leading|result|results|resulted|resulting|create|creates|created|creating) (.+)'
    # lines = fin.readlines()
    # document = ''
    # for line in lines:
    #     if re.match(key_words, line):
    #         document += line
    # doc_set = spacy_nlp(document)
    # count = 0
    # for _doc in tqdm(doc_set.sents):
    #     ptree.create(_doc)
    #     res = causalRules.extract(ptree)
    #     for r in res:
    #         print(r)
    cause_rules = '(.+) (?:cause|causes|caused|causing) (.+)'
    c = 0
    lines = fin.readlines()
    for line in lines:
        g = re.match(cause_rules, line)
        if g:
            if re.match('(.+) (?:is|was) caused by (.+)', line):
                print(line)
