import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.font_manager import fontManager
import codecs
import numpy as np
from tools import project_source_path
import os
import random
from sklearn.metrics import auc


def draw_curve(data, params, name):
    plt.cla()
    plt.figure(figsize=(14, 9))

    # axes.set_title('p-r curve')
    for i in range(len(data)):
        points_list = np.array(data[i])

        recall, precision = [float(d[0]) for d in points_list], [float(d[1]) for d in points_list]
        plt.plot(
            recall, precision, label=params[i][0], color=params[i][1], linestyle=params[i][2],
            marker=params[i][3], ms=10, markevery=params[i][4]
        )

    plt.legend(loc='upper right', prop={'size': 22})
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.tick_params(labelsize=18)
    plt.xlim(0, None)
    plt.ylim(0.4, None)
    # , bbox_inches = 'tight'
    plt.rcParams['savefig.dpi'] = 1000
    # plt.show()
    plt.savefig('/home/feiteng/Documents/sources/{}.png'.format(name), bbox_inches='tight')


if __name__ == '__main__':
    """
    bmm color: teal, c, coral, green, goldenrod, dimgrey, steelblue
    """
    """
    ablation: noweight
    """
    params = {
       
        # 'en_sota': ('BMM', 'c', '-.', '>', 0.2),
        # 'pairwise-matching': ('Pairwise-Matching', 'dimgrey', '-.', '>', 0.2),
        'fl_max_11': ('Max-Matching', 'steelblue', '-.', 's', 0.2),
        # 'att-new': ('Attentive-Matching', 'coral', '--', 'X', (0.0, 0.05)),
        # 'mmplus_53': ('MMPlus', 'green', '-.', '*', 0.1),
        # 'bmm7_37': ('BMMPlus37', 'red', '-.', '>', 0.2),
        # 'bmm7_38': ('BMMPlus38', 'green', '-.', '>', 0.2),

        # 'mm7_new_42': ('MMPlusSota', 'green', '-', 's', 0.2),

        # 'Lookup_baseline': ('Look-up', 'olive', '-.', '*', 0.1),
        # 'Vanilla': ('vEmbed', 'green', '-.', 'o', 0.2),
        # 'Causal': ('cEmbed', 'teal', '-.', '>', 0.2),
        # 'Causal_bidir': ('cEmbedBi', 'goldenrod', '-.', 'd', 0.2),
        # 'Causal_bidir_pmi': ('cEmbedBiNoise', 'dimgrey', '-.', '*', 0.2),
        #
        'en_sota': ('BMM', 'c', '-', '^', 0.1),

        'noweight1_4': ('noweight1_4', 'red', '-', 's', 0.1),
        'noweight2_4': ('BMM-weighting', 'coral', '--', 'd', 0.1),

        'bmm_nodirection_8': ('BMM-direction', 'black', '-.', '>', 0.1),
        # 'fl_max_11': ('Max-Matching', 'black', '-.', 's', 0.1),
        # 'att-new': ('Att-Matching', 'black', ':', 'x', (0.45, 0.05)),
        # # 'max-matching': ('Max', 'red', '-'),
        # # 'pairwise-matching': ('Pairwise-Matching', 'dimgrey', '-.'),
        # # 'Lookup_baseline': ('Look-up', 'olive', '-.'),
        # 'Vanilla': ('vEmbed', 'black', '--', '*', 0.1),
        # 'Causal': ('cEmbed', 'black', '-', '>', (0.0, 0.1)),
        # 'Causal_bidir': ('cEmbedBi', 'black', '-.', 'd', (0.45, 0.1)),
        # 'Causal_bidir_pmi': ('cEmbedBiNoise', 'black', '--', 'o', 0.1),

        # 'extract_max': ('extract_max', 'blue', '-'),
        # 'extract_att': ('extract_att', 'dimgrey', '-.'),

    }

    points_data, parameters = [], []
    path = os.path.join(project_source_path, 'prcurve/')
    files = os.listdir(path)
    names = [f.strip().split('.')[1] for f in files]
    ordered_file = [
        'en_sota',
        # 'fl_max_11',
        # 'att-new',
        # 'pairwise-matching',
        # 'noweight1_4',
       
        # 'Causal_bidir_pmi',
        #
        # 'Causal_bidir',
        # 'Causal',
        #
        # # 'Lookup_baseline',
        # 'Vanilla',

    ]
    for f in ordered_file:
        if f not in params:
            continue
        print(f)
        lines = codecs.open(os.path.join(path, 'points.{}'.format(f)), 'r', 'utf-8').readlines()
        points = [line.strip().split('\t') for line in lines]
        points_data.append(points)
        parameters.append(params[str(f)])

    draw_curve(points_data, parameters, 'PRCurve')
