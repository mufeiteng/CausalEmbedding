import codecs
import os
from tools import project_source_path
import json


path = os.path.join(project_source_path, 'causalembedding/siminfer/en/')
fin1 = codecs.open(os.path.join(path, 'wp_sim_resources.txt'), 'r', 'utf-8')
fin2 = codecs.open(os.path.join(path, 'wp_sim_use_wordnet.txt'), 'r', 'utf-8')
resource = dict()
for line in fin1:
    res = line.strip().split('----')
    w1, d = res[0], json.loads(res[1])
    if w1 not in resource:
        resource[w1] = d
    for w2 in d:
        if w2 not in resource[w1]:
            resource[w1] = dict()
        resource[w1][w2] = float(d[w2])
fin1.close()

for line in fin2:
    res = line.strip().split('----')
    w1, d = res[0], json.loads(res[1])
    if w1 not in resource:
        resource[w1] = d
    for w2 in d:
        if w2 not in resource[w1]:
            resource[w1] = dict()
        resource[w1][w2] = float(d[w2])
fin2.close()

fout = codecs.open(os.path.join(path, 'wp_sim_combine.txt'), 'w', 'utf-8')
for w in resource:
    fout.write('{}----{}\n'.format(w, json.dumps(resource[w])))