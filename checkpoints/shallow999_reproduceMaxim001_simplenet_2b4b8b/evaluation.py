###########################################################################
# Computer vision - Binary neural networks demo software by HyperbeeAI.   #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. main@shallow.ai #
###########################################################################
import torch, sys, time
import torch.nn as nn
import torch.optim as optim

# bizden
import layers, models, dataloader
from library.utils import compute_batch_accuracy, compute_set_accuracy

bs = 100;
train_loader, test_loader = dataloader.load_cifar100(batch_size=bs, num_workers=6, shuffle=False, act_8b_mode=False);

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = models.maxim_simplenet()
model  = model.to(device) 

weight_dictionary = {}
weight_dictionary['conv1' ] = 8;
weight_dictionary['conv2' ] = 4;
weight_dictionary['conv3' ] = 2;
weight_dictionary['conv4' ] = 2;
weight_dictionary['conv5' ] = 2;
weight_dictionary['conv6' ] = 2;
weight_dictionary['conv7' ] = 2;
weight_dictionary['conv8' ] = 2;
weight_dictionary['conv9' ] = 2;
weight_dictionary['conv10'] = 2;
weight_dictionary['conv11'] = 4;
weight_dictionary['conv12'] = 4;
weight_dictionary['conv13'] = 4;
weight_dictionary['conv14'] = 4;

layer_attributes = []
for layer_string in dir(model):
    if(layer_string in weight_dictionary):
        layer_attribute = getattr(model, layer_string)
        print('Folding BN for:', layer_string)
        layer_attribute.configure_layer_base(weight_bits=weight_dictionary[layer_string], bias_bits=8, shift_quantile=0.985)
        layer_attribute.mode_fpt2qat('qat');
        setattr(model, layer_string, layer_attribute)
model.to(device) # somehow new parameters are left out, so they need a reload to the GPU

# then, load trained checkpoint
checkpoint = torch.load('training_checkpoint.pth.tar');
model.load_state_dict(checkpoint['state_dict'])

print('')
print('Computing test set accuracy, training checkpoint')
test_acc = compute_set_accuracy(model, test_loader)

print('')
print('Test accuracy:', test_acc*100.0)
print('')

train_loader, test_loader = dataloader.load_cifar100(batch_size=bs, num_workers=6, shuffle=False, act_8b_mode=True);

# then, qat2hw
model  = model.to(device) 
for layer_string in dir(model):
	layer_attribute = getattr(model, layer_string)
	if isinstance(layer_attribute, layers.shallow_base_layer):
		print('Generating HW parameters for:', layer_string)
		layer_attribute.mode_qat2hw('eval');
		setattr(model, layer_string, layer_attribute)
model.to(device) # somehow new parameters are left out, so they need a reload

print('')
print('Computing test set accuracy, hardware checkpoint')
test_acc = compute_set_accuracy(model, test_loader)

torch.save({
            'epoch': 123456789,
            'extras': {'best epoch':123456789, 'best_top1':100*test_acc.cpu().numpy(), 'clipping_method':'MAX_BIT_SHIFT', 'current_top1':100*test_acc.cpu().numpy()},
            'state_dict': model.state_dict(),
            'arch': 'ai85nascifarnet'
            }, 'hardware_checkpoint.pth.tar')

print('')
print('Test accuracy:', test_acc*100.0)
