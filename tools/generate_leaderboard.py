###########################################################################
# Computer vision - Binary neural networks demo software by HyperbeeAI.   #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. main@shallow.ai #
###########################################################################
import torch, matplotlib, os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def get_checkpoints_except_reproduce():
    # get a list of all available checkpoint folder names except the ones that have reproduce in it
    temporary  = sorted(os.listdir("./checkpoints"))
    checkpoint_full_paths = []
    checkpoint_name_list  = []
    for name in temporary:
        fullname = os.path.join("./checkpoints",name)
        if ((os.path.isdir(fullname)) and ('reproduce' not in fullname)):
            checkpoint_full_paths.append(os.path.abspath(fullname))
            checkpoint_name_list.append(name)

    return checkpoint_full_paths, checkpoint_name_list

def get_checkpoint_top1s_sizes(checkpoint_full_paths, checkpoint_name_list):
    checkpoint_best_top1s               = []
    checkpoint_sizes_in_bytes           = []
    checkpoint_sizes_in_bytes_max78000  = []
    checkpoint_sizes_antipodal          = []
    for i, cp in enumerate(checkpoint_full_paths):
        nn = os.path.join(cp, 'hardware_checkpoint.pth.tar')
        if(os.path.isfile(nn)):
            a  = torch.load(nn)
        else:
            print("Hardware checkpoint does not exist for:", checkpoint_name_list[i])
            checkpoint_best_top1s.append(None)
            checkpoint_sizes_in_bytes.append(None)
            checkpoint_sizes_in_bytes_max78000.append(None)
            checkpoint_sizes_antipodal.append(None)
            continue

        checkpoint_best_top1s.append(a['extras']['best_top1'])

        layer_keys = []
        layers = []
        for key in a['state_dict'].keys():
            fields = key.split('.')    
            if(fields[0] not in layer_keys):
                layer_keys.append(fields[0])
                layers.append({'key': fields[0], 
                               'weight_bits':None, 
                               'bias_bits':None, 
                               'adjust_output_shift':None, 
                               'output_shift':None, 
                               'quantize_activation':None, 
                               'shift_quantile':None, 
                               'weight': None, 
                               'bias':None })
                idx = -1
            else:
                idx = layer_keys.index(fields[0])

            if((fields[1]=='weight_bits') or \
               (fields[1]=='output_shift') or \
               (fields[1]=='bias_bits') or \
               (fields[1]=='quantize_activation') \
               or (fields[1]=='adjust_output_shift') \
               or (fields[1]=='shift_quantile')):
                layers[idx][fields[1]] = a['state_dict'][key].cpu().numpy();
            elif(fields[1]=='op'):
                layers[idx][fields[2]] = a['state_dict'][key].cpu().numpy();
            else:
                print('[ERROR]: Unknown field. Exiting', file=f)
                print('[ERROR]: Unknown field. Exiting')
                sys.exit()

        size_in_bytes          = 0.0
        size_in_bytes_max78000 = 0.0 ## this keeps track of antipodal layers as 2b, bad hack

        ## info flag that tells if there are any antipodal layers in the network, 
        ## this triggers viewing size_in_bytes_max78000 rather than size_in_bytes
        ## bad hack, needs to change at some point
        antipodal              = False 
        
        for layer in layers:
            ### Burak: handle antipodal layers
            ### Burak: implicit assumption -> all networks have bias
            if(layer['weight_bits'][0]==2):
                # antipodal 2-bit, count these as 1-bit
                if(len(np.unique(layer['weight'])) == 2): 
                    size_in_bytes +=    (layer['weight_bits'][0]/(2.0*8.0))*layer['weight'].size + (layer['bias_bits'][0]/8.0)*layer['bias'].size
                    size_in_bytes_max78000 += (layer['weight_bits'][0]/8.0)*layer['weight'].size + (layer['bias_bits'][0]/8.0)*layer['bias'].size
                    antipodal = True
                    continue

            newsize = (layer['weight_bits'][0]/8.0)*layer['weight'].size + (layer['bias_bits'][0]/8.0)*layer['bias'].size
            size_in_bytes += newsize
            size_in_bytes_max78000 += newsize;

        checkpoint_sizes_in_bytes.append(size_in_bytes)
        checkpoint_sizes_in_bytes_max78000.append(size_in_bytes_max78000)
        checkpoint_sizes_antipodal.append(antipodal)

    return checkpoint_best_top1s, checkpoint_sizes_in_bytes, checkpoint_sizes_in_bytes_max78000, checkpoint_sizes_antipodal

def main():

    cp_full_paths, cp_name_list = get_checkpoints_except_reproduce()
    print('')
    print('Found checkpoints (except reproduce checkpoints) at these locations:')
    for cp_path in cp_full_paths:
        print(cp_path)

    print('')
    print('Gathering hardware-mode top-1 accuracy and size info from each checkpoint')
    cp_best_top1s, cp_sizes_in_bytes, cp_sizes_in_bytes_max78000, cp_sizes_antipodal = get_checkpoint_top1s_sizes(cp_full_paths, cp_name_list)

    print("")
    print('Leaderboard')
    print('--------------------------------')
    for i, cp in enumerate(cp_name_list):
        print("Name          : ", cp)
        if(cp_best_top1s[i] is not None):
            print("Top-1 accuracy: ", np.round(100*cp_best_top1s[i])/100)
            if(cp_sizes_antipodal[i]):
                print("Size (KBytes) : ", cp_sizes_in_bytes_max78000[i]/1000.0, ', but has "-1/+1 only" 2b layers, so this would be:', cp_sizes_in_bytes[i]/1000.0, 'KBytes on MAX78002')
            else:
                print("Size (KBytes) : ", cp_sizes_in_bytes[i]/1000.0)
        else:
            print("Top-1 accuracy: ", cp_best_top1s[i])
            if(cp_sizes_antipodal[i]):
                print("Size (KBytes) : ", cp_sizes_in_bytes_max78000[i], ', MARK: has some antipodal 2b layers')
            else:
                print("Size (KBytes) : ", cp_sizes_in_bytes[i])
        print("")

if __name__ == '__main__':
    main()
