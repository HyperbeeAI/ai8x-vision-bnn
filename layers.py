###########################################################################
# Computer vision - Binary neural networks demo software by HyperbeeAI.   #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. main@shallow.ai #
###########################################################################
import torch, sys
import torch.nn as nn
import numpy as np
from torch.autograd import Function
from functions import quantization, clamping_qa, clamping_hw, calc_out_shift

###################################################
### Base layer for conv/linear, 
###    enabling quantization-related mechanisms
class shallow_base_layer(nn.Module):
    def __init__(
            self,
            quantization_mode = 'fpt', # 'fpt', 'qat', 'qat_ap' and 'eval'
            pooling_flag      = None,  # boolean flag for now, only maxpooling of 2-pools with stride 2
            operation_module  = None,  # torch nn module for keeping and updating conv/linear parameters 
            operation_fcnl    = None,  # torch nn.functional for actually doing the operation
            activation_module = None,  # torch nn module for relu/abs
            batchnorm_module  = None,  # torch nn module for batchnorm, see super
            output_width_30b  = False  # boolean flag that chooses between "bigdata" (32b) and normal (8b) activation modes for MAX78000
    ):
        super().__init__()

        ###############################################################################
        # Initialize stuff that won't change throughout the model's lifetime here
        # since this place will only be run once (first time the model is declared)
        if(pooling_flag==True):
            self.pool   = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.pool   = None

        ### Burak: we have to access and change (forward pass) and also train (backward pass) parameters .weight and .bias for the operations
        ###        therefore we keep both a functional and a module for Conv2d/Linear. The name "op" is mandatory for keeping params in Maxim 
        ###        checkpoint format.
        self.op         = operation_module
        self.op_fcn     = operation_fcnl
        self.act        = activation_module
        self.bn         = batchnorm_module
        self.wide       = output_width_30b

        ###############################################################################
        # Initialize stuff that will change during mode progression (FPT->QAT->Eval/HW).
        self.mode               = quantization_mode;
        self.quantize_Q_ud_8b   = None
        self.quantize_Q_ud_wb   = None
        self.quantize_Q_ud_bb   = None
        self.quantize_Q_ud_ap   = None
        self.quantize_Q_d_8b    = None
        self.quantize_Q_u_wb    = None
        self.quantize_Q_ud_wide = None
        self.quantize_Q_d_wide  = None
        self.clamp_C_qa_8b      = None
        self.clamp_C_qa_bb      = None
        self.clamp_C_qa_wb      = None
        self.clamp_C_hw_8b      = None
        self.clamp_C_qa_wide    = None
        self.clamp_C_hw_wide    = None

        ### Burak: these aren't really trainable parameters, but they're logged in the Maxim checkpoint format. It seems they marked 
        ###        them as "non-trainable parameters" to get them automatically saved in the state_dict
        self.output_shift        = nn.Parameter(torch.Tensor([ 0 ]), requires_grad=False) ### Burak: we called this los, this varies, default:0
        self.weight_bits         = nn.Parameter(torch.Tensor([ 8 ]), requires_grad=False) ### Burak: we called this wb, this varies, default:8
        self.bias_bits           = nn.Parameter(torch.Tensor([ 8 ]), requires_grad=False) ### Burak: this is always 8
        self.quantize_activation = nn.Parameter(torch.Tensor([ 0 ]), requires_grad=False) ### Burak: this is 0 in FPT, 1 in QAT & eval/hardware, default: fpt
        self.adjust_output_shift = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) ### Burak: this is 1 in FPT & QAT, 0 in eval/hardware, default: fpt
        self.shift_quantile      = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) ### Burak: this varies, default:1 (naive)

        ###############################################################################
        # Do first mode progression (to the default)
        ### Burak: this recognizes that layer configuration is done via a function, 
        ###        thus, can be done again in training time for mode progression
        weight_bits      = self.weight_bits
        bias_bits        = self.bias_bits
        shift_quantile   = self.shift_quantile
        self.configure_layer_base( weight_bits, bias_bits, shift_quantile )

    # This will be called during mode progression to set fields,
    # check workflow-training-modes.png in doc for further info.
    # sets functions for all modes though, not just the selected mode
    def configure_layer_base(self, weight_bits, bias_bits, shift_quantile):
        # quantization operators
        self.quantize_Q_ud_8b   = quantization(xb = 8,           mode ='updown' , wide=False) # 8 here is activation bits
        self.quantize_Q_ud_wb   = quantization(xb = weight_bits, mode ='updown' , wide=False) 
        self.quantize_Q_ud_bb   = quantization(xb = bias_bits,   mode ='updown' , wide=False) 
        self.quantize_Q_ud_ap   = quantization(xb = 2,        mode ='updown_ap' , wide=False) # 2 here is dummy, mode antipodal overrides xb
        self.quantize_Q_d_8b    = quantization(xb = 8,           mode ='down'   , wide=False) # 8 here is activation bits
        self.quantize_Q_u_wb    = quantization(xb = weight_bits, mode ='up'     , wide=False)
        self.quantize_Q_ud_wide = quantization(xb = 8,           mode ='updown' , wide=True)  # 8 here is activation bits, but its wide, so check inside
        self.quantize_Q_d_wide  = quantization(xb = 8,           mode ='down'   , wide=True)  # 8 here is activation bits, but its wide, so check inside
        
        # clamping operators
        self.clamp_C_qa_8b    = clamping_qa(xb = 8,           wide=False) # 8 here is activation bits
        self.clamp_C_qa_bb    = clamping_qa(xb = bias_bits,   wide=False)
        self.clamp_C_qa_wb    = clamping_qa(xb = weight_bits, wide=False)
        self.clamp_C_hw_8b    = clamping_hw(xb = 8,           wide=False) # 8 here is activation bits
        self.clamp_C_qa_wide  = clamping_qa(xb = None,        wide=True)  # None to avoid misleading info on the # of bits, check inside
        self.clamp_C_hw_wide  = clamping_hw(xb = None,        wide=True)  # None to avoid misleading info on the # of bits, check inside

        # state variables
        self.weight_bits     = nn.Parameter(torch.Tensor([ weight_bits    ]), requires_grad=False)
        self.bias_bits       = nn.Parameter(torch.Tensor([ bias_bits      ]), requires_grad=False)
        self.shift_quantile  = nn.Parameter(torch.Tensor([ shift_quantile ]), requires_grad=False)

    # This will be called during mode progression, during training
    def mode_fpt2qat(self, quantization_mode):
        # just fold batchnorms
        if(self.bn is not None):
            w_fp = self.op.weight.data
            b_fp = self.op.bias.data
    
            running_mean_mu     = self.bn.running_mean
            running_var         = self.bn.running_var
            running_stdev_sigma = torch.sqrt(running_var + 1e-20)
    
            w_hat = w_fp * (1.0 / (running_stdev_sigma*4.0)).reshape((w_fp.shape[0],) + (1,) * (len(w_fp.shape) - 1))
            b_hat = (b_fp - running_mean_mu)/(running_stdev_sigma*4.0)
    
            self.op.weight.data = w_hat
            self.op.bias.data   = b_hat
            self.bn             = None
        else:
            pass
            #print('This layer does not have batchnorm')
        self.mode                = quantization_mode;
        self.quantize_activation = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) ### Burak: this is 0 in FPT, 1 in QAT & eval/hardware
        self.adjust_output_shift = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) ### Burak: this is 1 in FPT & QAT, 0 in eval/hardware

    # This will be called during mode progression after training, for eval
    def mode_qat2hw(self, quantization_mode):
        w_hat = self.op.weight.data
        b_hat = self.op.bias.data

        shift = -self.output_shift.data;
        s_o   = 2**(shift)
        wb    = self.weight_bits.data.cpu().numpy()[0]
       
        w_clamp = [-2**(wb-1)  , 2**(wb-1)-1 ]
        b_clamp = [-2**(wb+8-2), 2**(wb+8-2)-1] # 8 here is activation bits

        w = w_hat.mul(2**(wb -1)).mul(s_o).add(0.5).floor()
        w = w.clamp(min=w_clamp[0],max=w_clamp[1])

        b = b_hat.mul(2**(wb -1 + 7)).mul(s_o).add(0.5).floor()
        b = b.clamp(min=b_clamp[0],max=b_clamp[1])

        self.op.weight.data      = w
        self.op.bias.data        = b
        self.mode                = quantization_mode;
        self.quantize_activation = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) ### Burak: this is 0 in FPT, 1 in QAT & eval/hardware
        self.adjust_output_shift = nn.Parameter(torch.Tensor([ 0 ]), requires_grad=False) ### Burak: this is 1 in FPT & QAT, 0 in eval/hardware

    def mode_qat_ap2hw(self, quantization_mode):
        w_hat = self.op.weight.data
        b_hat = self.op.bias.data

        shift = -self.output_shift.data;
        s_o   = 2**(shift)
        wb    = self.weight_bits.data.cpu().numpy()[0]

        if(wb==2):
            w = self.quantize_Q_ud_ap(w_hat).mul(2.0)
        else:
            w_clamp = [-2**(wb-1)  , 2**(wb-1)-1 ]
            w = w_hat.mul(2**(wb -1)).mul(s_o).add(0.5).floor()
            w = w.clamp(min=w_clamp[0],max=w_clamp[1])

        b_clamp = [-2**(wb+8-2), 2**(wb+8-2)-1] # 8 here is activation bits
        b = b_hat.mul(2**(wb -1 + 7)).mul(s_o).add(0.5).floor()
        b = b.clamp(min=b_clamp[0],max=b_clamp[1])

        self.op.weight.data      = w
        self.op.bias.data        = b
        self.mode                = quantization_mode;
        self.quantize_activation = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) ### Burak: this is 0 in FPT, 1 in QAT & eval/hardware
        self.adjust_output_shift = nn.Parameter(torch.Tensor([ 0 ]), requires_grad=False) ### Burak: this is 1 in FPT & QAT, 0 in eval/hardware

        
    def forward(self, x):
        if(self.pool is not None):
            x = self.pool(x)

        if(self.mode == 'fpt'):
            # pre-compute stuff
            w_fp = self.op.weight
            b_fp = self.op.bias

            # actual forward pass
            x = self.op_fcn(x, w_fp, b_fp, self.op.stride, self.op.padding)
            if(self.bn is not None):
                x = self.bn(x)     # make sure var=1 and mean=0
                x = x / 4.0        # since BN is only making sure var=1 and mean=0, 1/4 is to keep everything within [-1,1] w/ hi prob.
            if(self.act is not None):
                x = self.act(x)
            if((self.wide) and (self.act is None)):
                x = self.clamp_C_qa_wide(x)
            else:
                x = self.clamp_C_qa_8b(x)

            # save stuff (los is deactivated in fpt)
            self.output_shift        = nn.Parameter(torch.Tensor([ 0 ]), requires_grad=False) # functional, used in Maxim-friendly checkpoints
            self.quantize_activation = nn.Parameter(torch.Tensor([ 0 ]), requires_grad=False) # ceremonial, for Maxim-friendly checkpoints
            self.adjust_output_shift = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) # ceremonial, for Maxim-friendly checkpoints

        elif(self.mode == 'qat'):
            ###############################################################################
            ## ASSUMPTION: batchnorms are already folded before coming here. Check doc,  ##
            ## the parameters with _fp and with _hat are of different magnitude          ##
            ###############################################################################

            # pre-compute stuff
            w_hat = self.op.weight
            b_hat = self.op.bias
            los  = calc_out_shift(w_hat.detach(), b_hat.detach(), self.shift_quantile.detach())            
            s_w  = 2**(-los)
            s_o  = 2**(los)
            w_hat_q = self.clamp_C_qa_wb(self.quantize_Q_ud_wb(w_hat*s_w));
            b_hat_q = self.clamp_C_qa_bb(self.quantize_Q_ud_bb(b_hat*s_w));

            # actual forward pass
            x = self.op_fcn(x, w_hat_q, b_hat_q, self.op.stride, self.op.padding)
            x = x*s_o
            if(self.act is not None):
                x = self.act(x)
            if((self.wide) and (self.act is None)):
                x = self.quantize_Q_ud_wide(x)
                x = self.clamp_C_qa_wide(x)
            else:
                x = self.quantize_Q_ud_8b(x)
                x = self.clamp_C_qa_8b(x)

            # save stuff
            self.output_shift        = nn.Parameter(torch.Tensor([ los ]), requires_grad=False) # functional, used in Maxim-friendly checkpoints

        elif(self.mode == 'qat_ap'):
            ###############################################################################
            ## ASSUMPTION: batchnorms are already folded before coming here. Check doc,  ##
            ## the parameters with _fp and with _hat are of different magnitude          ##
            ###############################################################################

            # pre-compute stuff
            w_hat = self.op.weight
            b_hat = self.op.bias
            los  = calc_out_shift(w_hat.detach(), b_hat.detach(), self.shift_quantile.detach())            
            s_w  = 2**(-los)
            s_o  = 2**(los)
            ##############################################
            # This is the only difference from qat
            if(self.weight_bits.data==2):
                w_hat_q = self.quantize_Q_ud_ap(w_hat*s_w);
            else:
                w_hat_q = self.clamp_C_qa_wb(self.quantize_Q_ud_wb(w_hat*s_w));
            ##############################################
            b_hat_q = self.clamp_C_qa_bb(self.quantize_Q_ud_bb(b_hat*s_w));

            # actual forward pass
            x = self.op_fcn(x, w_hat_q, b_hat_q, self.op.stride, self.op.padding)
            x = x*s_o
            if(self.act is not None):
                x = self.act(x)
            if((self.wide) and (self.act is None)):
                x = self.quantize_Q_ud_wide(x)
                x = self.clamp_C_qa_wide(x)
            else:
                x = self.quantize_Q_ud_8b(x)
                x = self.clamp_C_qa_8b(x)

            # save stuff
            self.output_shift        = nn.Parameter(torch.Tensor([ los ]), requires_grad=False) # functional, used in Maxim-friendly checkpoints

        elif(self.mode == 'eval'):
            #####################################################################################
            ## ASSUMPTION: parameters are already converted to HW before coming here.Check doc ##
            #####################################################################################

            # pre-compute stuff
            w = self.op.weight
            b = self.op.bias
            los  = self.output_shift
            s_o  = 2**(los)
            w_q = self.quantize_Q_u_wb(w);
            b_q = self.quantize_Q_u_wb(b); # yes, wb, not a typo, they need to be on the same scale

            # actual forward pass
            x = self.op_fcn(x, w_q, b_q, self.op.stride, self.op.padding) # convolution / linear
            x = x*s_o
            if(self.act is not None):
                x = self.act(x)
            if((self.wide) and (self.act is None)):
                x = self.quantize_Q_d_wide(x)
                x = self.clamp_C_hw_wide(x)
            else:
                x = self.quantize_Q_d_8b(x)
                x = self.clamp_C_hw_8b(x)

            # nothing to save, this was a hardware-emulated evaluation pass
        else:
            print('wrong quantization mode. should have been one of {fpt, qat, eval}. exiting')
            sys.exit()

        return x


class conv(shallow_base_layer):
    def __init__(
            self,
            C_in_channels      = None,    # number of input channels
            D_out_channels     = None,    # number of output channels
            K_kernel_dimension = None,    # square kernel dimension
            padding            = None,    # amount of pixels to pad on one side (other side is symmetrically padded too)
            pooling            = False,   # boolean flag for now, only maxpooling of 2-pools with stride 2
            batchnorm          = False,   # boolean flag for now, no trainable affine parameters 
            activation         = None,    # 'relu' is the only choice for now
            output_width_30b   = False    # boolean flag that chooses between "bigdata" (32b) and normal (8b) activation modes for MAX78000
    ):
        pooling_flag = pooling

        if(activation is None):
            activation_fcn = None;
        elif(activation == 'relu'):
            activation_fcn = nn.ReLU(inplace=True);
        else:
            print('wrong activation type in model. only {relu} is acceptable. exiting')
            sys.exit()

        ### Burak: only a module is enough for BN since we neither need to access internals in forward pass, nor train anything (affine=False)
        if(batchnorm):
        	batchnorm_mdl  = nn.BatchNorm2d(D_out_channels, eps=1e-05, momentum=0.05, affine=False)
        else:
        	batchnorm_mdl  = None;

        operation_mdl  = nn.Conv2d(C_in_channels, D_out_channels, kernel_size=K_kernel_dimension, stride=1, padding=padding, bias=True);
        operation_fcn  = nn.functional.conv2d

        super().__init__(
            pooling_flag       = pooling_flag,
            activation_module  = activation_fcn,
            operation_module   = operation_mdl,
            operation_fcnl     = operation_fcn,
            batchnorm_module   = batchnorm_mdl,
            output_width_30b   = output_width_30b
        )

def linear_functional(x, weight, bias, _stride, _padding):
    # dummy linear function that has same arguments as conv
    return nn.functional.linear(x, weight, bias)

class fullyconnected(shallow_base_layer):
    def __init__(
            self,
            in_features        = None,    # number of output features
            out_features       = None,    # number of output features
            pooling            = False,   # boolean flag for now, only maxpooling of 2-pools with stride 2
            batchnorm          = False,   # boolean flag for now, no trainable affine parameters 
            activation         = None,    # 'relu' is the only choice for now
            output_width_30b   = False    # boolean flag that chooses between "bigdata" (32b) and normal (8b) activation modes for MAX78000
    ):
 
        pooling_flag = pooling

        if(activation is None):
            activation_fcn = None;
        elif(activation == 'relu'):
            activation_fcn = nn.ReLU(inplace=True);
        else:
            print('wrong activation type in model. only {relu} is acceptable. exiting')
            sys.exit()

        ### Burak: only a module is enough for BN since we neither need to access internals in forward pass, nor train anything (affine=False)
        if(batchnorm):
        	batchnorm_mdl  = nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.05, affine=False)
        else:
        	batchnorm_mdl  = None;

        operation_mdl  = nn.Linear(in_features, out_features, bias=True);
        operation_fcn  = linear_functional

        super().__init__(
            pooling_flag       = pooling_flag,
            activation_module  = activation_fcn,
            operation_module   = operation_mdl,
            operation_fcnl     = operation_fcn,
            batchnorm_module   = batchnorm_mdl,
            output_width_30b   = output_width_30b
        )
 
        # Define dummy arguments to make Linear and conv compatible in shallow_base_layer.
        # the name "op" here refers to op in super, i.e., in base_layer
        self.op.stride = None
        self.op.padding = None
