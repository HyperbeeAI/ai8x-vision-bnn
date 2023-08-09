###########################################################################
# Computer vision - Binary neural networks demo software by HyperbeeAI.   #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. main@shallow.ai #
###########################################################################
import torch, matplotlib, os, sys, argparse
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.use('Agg')

from datetime import date

# bizden
from generate_leaderboard import get_checkpoints_except_reproduce, get_checkpoint_top1s_sizes

def main():
    cp_full_paths, cp_name_list = get_checkpoints_except_reproduce()
    print('')
    print('Found checkpoints (except reproduce checkpoints) at these locations:')
    for cp_path in cp_full_paths:
        print(cp_path)

    print('')
    print('Gathering hardware-mode top-1 accuracy and size info from each checkpoint')
    cp_best_top1s, cp_sizes_in_bytes, cp_sizes_in_bytes_max78000, cp_sizes_antipodal = get_checkpoint_top1s_sizes(cp_full_paths, cp_name_list)

    print('')
    print('Generating results graph under documentation, with timestamp')

    ###############################################
    ### Hardcoded
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim((80,400))
    ax.set_ylim((53,68.0))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel('Size [KBytes]', fontsize=15)
    ax.set_ylabel('Validation set accuracy [%]', fontsize=15)
    ###############################################

    color_maxim   = np.asarray([30,30,255])/256
    color_shallow = np.asarray([255,30,30])/256
    for i, name in enumerate(cp_name_list):
        if('maxim' in name):
            color = color_maxim
            annot = 'm'+name[5:8]
        elif('shallow' in name):
            color = color_shallow
            annot = 's'+name[7:10]
        else:
            print('')
            print('whose model is this?! ->', name)
            print('exiting')
            print('')
            sys.exit()


        if(cp_sizes_antipodal[i]):
            ax.scatter(cp_sizes_in_bytes[i]/1000.0, cp_best_top1s[i], color = color, s = 70, linestyle='None', alpha=0.2)
            ax.scatter(cp_sizes_in_bytes_max78000[i]/1000.0, cp_best_top1s[i], color = color, s = 70, linestyle='None', alpha=0.8)
            ax.plot([cp_sizes_in_bytes[i]/1000.0, cp_sizes_in_bytes_max78000[i]/1000.0], [cp_best_top1s[i], cp_best_top1s[i]], color = color, linestyle='dashed')
        else:
            ax.scatter(cp_sizes_in_bytes[i]/1000.0, cp_best_top1s[i], color = color, s = 70, linestyle='None', alpha=0.8)

        #annot_position_x = cp_sizes_in_bytes[i]/1000.0-10
        #annot_position_y = cp_best_top1s[i]+0.6
        #ax.text(annot_position_x, annot_position_y, annot, fontsize=11, color=color)

    custom_lines = [Line2D([0], [0], color=color_maxim, lw=4),
                    Line2D([0], [0], color=color_shallow, lw=4)]
    ax.legend(custom_lines, ['maxim', 'shallow'], loc='upper left', fontsize=12)
    plt.title('Models for CIFAR-100', fontsize=15)
    
    today      = date.today()
    dd         = today.strftime("%Y-%m-%d")
    graph_path = 'documentation/'+dd+'-results-graph.png'
    plt.savefig(graph_path)

    print('')
    print('Saved graph under', graph_path)
    print('')

if __name__ == '__main__':
    main()
