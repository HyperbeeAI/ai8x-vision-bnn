###########################################################################
# Computer vision - Binary neural networks demo software by HyperbeeAI.   #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai #
###########################################################################
import torch, matplotlib, os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def generate_histogram_for_quantized_layer(layer_key, layer_weight, layer_bias, checkpoint_type, histograms_folderpath):
    histogram_folder_exists = os.path.isdir(histograms_folderpath)
    if not histogram_folder_exists:
        os.makedirs(histograms_folderpath)

    matplotlib.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(20, 10))
    ww = layer_weight.flatten();
    bb = layer_bias.flatten();
    
    ww_max = np.amax(ww)
    ww_min = np.amin(ww)
    ww_unq = len(np.unique(ww))

    bb_max = np.amax(bb)
    bb_min = np.amin(bb)
    bb_unq = len(np.unique(bb))

    if checkpoint_type=='hardware':
        ww_num_bins = ww_unq*3
        bb_num_bins = bb_unq*3
        ww_max_lim = ww_max+1;
        bb_max_lim = bb_max+1/16384;
    elif checkpoint_type=='training':
        ww_num_bins = min(ww_unq*3,800)
        bb_num_bins = min(bb_unq*3,800)
        ww_max_lim = ww_max+1/128;
        bb_max_lim = bb_max+1/128;

    axs[0].grid(True)
    axs[0].set_title('weight', fontdict={'fontsize': 22, 'fontweight': 'medium'})
    axs[0].hist(ww, range=(ww_min, ww_max_lim), bins=ww_num_bins, align='left')

    axs[1].grid(True)
    axs[1].set_title('bias',   fontdict={'fontsize': 22, 'fontweight': 'medium'})
    axs[1].hist(bb, range=(bb_min, bb_max_lim), bins=bb_num_bins, align='left')

    filename = os.path.join(histograms_folderpath,layer_key + '.jpg')
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Print out model statistics file and optionally also save weight/bias histogram figures for each layer')
    parser.add_argument('-c','--checkpoint-name', help='Name of folder under the checkpoints folder for which you want to generate a model statistics file', required=True)
    parser.add_argument('-q','--checkpoint-type', help='checkpoint type can be either a hardware or training checkpoint.', required=True)
    parser.add_argument('-g','--generate-histograms', help='Add this flag if you want to save jpg  figures inside the checkpoint folder for histograms of bias and weight values of each layer in the network', action='store_true', default=False, required=False)
    args = vars(parser.parse_args())

    checkpoint_folder = os.path.join('checkpoints',args['checkpoint_name']);
    if(os.path.isdir(checkpoint_folder)):
        print('')
        print('Found checkpoint folder')
    else:
        print('')
        print('Could not find checkpoint folder. Please check that:')
        print('1- you are running this script from the top level of the repository, and')
        print('2- the checkpoint folder you gave the name for exists (needs to be created manually)')
        sys.exit();

    checkpoint_type = args['checkpoint_type']
    if(checkpoint_type=='hardware'):
        print('')
        print('Searching for a hardware_checkpoint.pth.tar')
        print('')
        check_for_bit_errors = True;
    elif(checkpoint_type=='training'):
        print('')
        print('Searching for a training_checkpoint.pth.tar')
        print('')
        check_for_bit_errors = False;
    else:
        print('')
        print('Something is wrong, we dont know of a',checkpoint_type, 'checkpoint. Perhaps a misspelling?' )
        print('')
        sys.exit()

    checkpoint_filename = checkpoint_type+'_checkpoint.pth.tar';

    a = torch.load(os.path.join(checkpoint_folder,checkpoint_filename))

    flag_generate_histograms = args['generate_histograms']
    if(flag_generate_histograms):
        print('[INFO]: Will generate histograms')

    with open(os.path.join(checkpoint_folder,'statistics_'+checkpoint_type+'_checkpoint'), 'w') as f:
        print('[INFO]: Generating statistics file')
        print('Top:', file=f)
        for key in a.keys():
            print('  ', key, file=f)

        if( 'arch' not in a.keys()):
            print('[ERROR]: there is no key named arch in this checkpoint', file=f)
            print('[ERROR]: there is no key named arch in this checkpoint')
            #sys.exit()
        if( 'state_dict' not in a.keys()):
            print('[ERROR]: there is no key named state_dict in this checkpoint', file=f)
            print('[ERROR]: there is no key named state_dict in this checkpoint')
            #sys.exit()
        if( 'extras' not in a.keys()):
            print('[ERROR]: there is no key named extras in this checkpoint', file=f)
            print('[ERROR]: there is no key named extras in this checkpoint')
            #sys.exit()

        print('-------------------------------------', file=f)
        print('arch:',  a['arch'], file=f)

        print('-------------------------------------', file=f)
        print('extras:',  a['extras'], file=f)

        print('-------------------------------------', file=f)
        print('state_dict:', file=f)

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

        for layer in layers:
            print('  ', layer['key'], file=f)
            print('     output_shift:        ', layer['output_shift'], file=f)
            print('     adjust_output_shift: ', layer['adjust_output_shift'], file=f)
            print('     quantize_activation: ', layer['quantize_activation'], file=f)
            print('     shift_quantile:      ', layer['shift_quantile'], file=f)
            print('     weight bits:         ', layer['weight_bits'], file=f)
            print('     bias_bits:           ', layer['bias_bits'], file=f)

            print('     bias', file=f)
            print('        total # of elements, shape:', np.size(layer['bias']), ',', list(layer['bias'].shape), file=f)
            print('        # of unique elements:      ', len(np.unique(layer['bias'])), file=f)
            print('        min, max, mean:', np.amin(layer['bias']), ', ', np.amax(layer['bias']), ', ', np.mean(layer['bias']), file=f)
            if((len(np.unique(layer['bias'])) > 2**layer['bias_bits']) and (check_for_bit_errors)):
                print('', file=f)
                print('[WARNING]: # of unique elements in bias tensor is more than that allowed by bias_bits.', file=f)
                print('           This might be OK, since Maxim deployment repository right shifts these.', file=f)
                print('', file=f)
                print('')
                print('[WARNING]: # of unique elements in bias tensor is more than that allowed by bias_bits.')
                print('           This might be OK, since Maxim deployment repository right shifts these.')
                print('           Check stats file for details.')
                print('')
            print('     weight', file=f)
            print('        total # of elements, shape:', np.size(layer['weight']), ',', list(layer['weight'].shape), file=f)
            print('        # of unique elements:      ', len(np.unique(layer['weight'])), file=f)
            print('        min, max, mean:', np.amin(layer['weight']), ', ', np.amax(layer['weight']), ', ', np.mean(layer['weight']), file=f)

            if((len(np.unique(layer['weight'])) > 2**layer['weight_bits']) and (check_for_bit_errors)):
                print('', file=f)
                print('[ERROR]: # of unique elements in weight tensor is more than that allowed by weight_bits.', file=f)
                print('         This is definitely not OK, weights are used in HW as is.', file=f)
                print('         Exiting.', file=f)
                print('', file=f)
                print('')
                print('[ERROR]: # of unique elements in weight tensor is more than that allowed by weight_bits.')
                print('         This is definitely not OK, weights are used in HW as is.')
                print('         Exiting.')
                print('')
                sys.exit()
            if(flag_generate_histograms):
                generate_histogram_for_quantized_layer(layer['key'], layer['weight'], layer['bias'], checkpoint_type, os.path.join(checkpoint_folder, 'histograms_'+checkpoint_type+'_checkpoint'))
                print('[INFO]: saved histograms for layer', layer['key'])


if __name__ == '__main__':
    main()
