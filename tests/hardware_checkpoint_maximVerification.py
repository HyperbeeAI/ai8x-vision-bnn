###########################################################################
# Computer vision - Binary neural networks demo software by HyperbeeAI.   #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai #
###########################################################################
import sys, torch
sys.path.append(".") ## works only when running from repo top layer
import layers
import models
import dataloader

bs = 250;
train_loader, test_loader = dataloader.load_cifar100(batch_size=bs, num_workers=1, shuffle=True, act_8b_mode=True);

print('')
print('Check: maxim checkpoints loaded into our model definitions, see test accuracy.')
print('       We expect approx. 64.32 for NAS, 55.76 for simplenet')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('')
print('Device:', device)

print('')
print('NAS Model')
mm = models.maxim_nas()
mm = mm.to(device) 

## make mode eval here (easy since we set all layer weights to 8b)
for layer_string in dir(mm):
	layer_attribute = getattr(mm, layer_string)
	if isinstance(layer_attribute, layers.shallow_base_layer):
		layer_attribute.configure_layer_base(weight_bits=8, bias_bits=8, shift_quantile=0.99)
		layer_attribute.mode_fpt2qat('qat');
		layer_attribute.mode_qat2hw('eval')
		setattr(mm, layer_string, layer_attribute)

checkpoint = torch.load('checkpoints/maxim000_nas_8b/hardware_checkpoint.pth.tar')
mm.load_state_dict(checkpoint['state_dict'])
mm = mm.to(device) 

dataiter = iter(test_loader)
ma = 0;
for i in range(0,int(10000/bs)):
	images , labels = dataiter.next()
	images = images.to(device) 
	labels =labels.to(device) 
	out =  mm(images)
	ma  += torch.sum(torch.argmax(out,dim=1)==labels)
print('Test Accuracy:', (ma)/10000*100)

print('')
print('Simplenet Mixed Precision Model')
mm = models.maxim_simplenet()
mm = mm.to(device) 

## make mode eval here (not that easy, layers are 2b/4b/8b)
# replace that weird policy thing here with explicit settings
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
for layer_string in dir(mm):
	if(layer_string in weight_dictionary):
		layer_attribute = getattr(mm, layer_string)
		layer_attribute.configure_layer_base(weight_bits=weight_dictionary[layer_string], bias_bits=8, shift_quantile=1.0)
		layer_attribute.mode_fpt2qat('qat');
		layer_attribute.mode_qat2hw('eval')
		setattr(mm, layer_string, layer_attribute)

checkpoint = torch.load('checkpoints/maxim001_simplenet_2b4b8b/hardware_checkpoint.pth.tar')
mm.load_state_dict(checkpoint['state_dict'], strict=False)
mm = mm.to(device) 

dataiter = iter(test_loader)
ma = 0;
for i in range(0,int(10000/bs)):
	images , labels = dataiter.next()
	images = images.to(device) 
	labels =labels.to(device) 
	out =  mm(images)
	ma  += torch.sum(torch.argmax(out,dim=1)==labels)
print('Test Accuracy:', (ma)/10000*100)
