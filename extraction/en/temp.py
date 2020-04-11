# -*- coding: utf-8 -*-
import spacy
from tools import project_source_path
from boostvec.en.prepare.parser import EnParseTree
from extraction.en.causalrules import EnCausalRules
import os
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
    causalRules = EnCausalRules(stemmer=lemmatizer)
    # s = 'The wide range of salinities in the North Atlantic is caused by the asymmetry of the northern subtropical gyre and the large number of contributions from a wide range of sources: Labrador Sea, Norwegian-Greenland Sea, Mediterranean, and South Atlantic Intermediate Water.'
    s = 'Similarities centered on the principles that life involves suffering, that suffering is caused by desire , and that the extinction of desire leads to liberation.'
    doc = spacy_nlp(s)
    count = 0
    ptree.create(doc)
    res = causalRules.extract(ptree)
    for r in res:
        print(r)
