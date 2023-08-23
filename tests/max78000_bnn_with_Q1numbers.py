###########################################################################
# Computer vision - Binary neural networks demo software by HyperbeeAI.   #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai #
###########################################################################
import torch

batch_size      = 1
num_rows        = 32
num_cols        = 32
input_channels  = 30
output_channels = 100
kernel_dim      = 3
same_padding    = kernel_dim // 2 

## input activations, random, 8-bit [-128,+127]
u = (1/4)*torch.randn((batch_size, input_channels, num_rows, num_cols)) ## somewhat close to the [-1,1] range
u = u.mul((2**7))                                                 ## expand to the [-128, +128] range (not quantized)
min_act = -(2**(8-1))                                             ## compute min value for quantized activation
max_act = 2**(8-1)-1                                              ## compute max value for quantized activation
u = u.add(0.5).floor().clamp(min=min_act, max=max_act)            ## quantize to 8-bit, 2s complement, clamp to [-128, +127]


## weight, random, -1/+1
x = (1/4)*torch.rand((output_channels, input_channels, kernel_dim, kernel_dim))  ## somewhat close to the [-1,1] range
for kr in range(0, kernel_dim):
    for kc in range(0, kernel_dim):
        for ki in range(0, input_channels):
            for ko in range(0, output_channels):
                element = x[ko,ki,kr,kc]
                if(element > 0.0):
                    x[ko,ki,kr,kc] = 1           ## quantize to +1
                else:
                    x[ko,ki,kr,kc] = -1          ## quantize to -1


## bias, random, 8-bit [-128,+127]
b = (1/4)*torch.randn((output_channels))                 ## somewhat close to the [-1,1] range
b = b.mul((2**7))                                        ## expand to the [-128, +128] range (not quantized)
b = b.add(0.5).floor().clamp(min=min_act, max=max_act)   ## quantize to 8-bit, 2s complement, clamp to [-128, +127]


## output with -1/+1
y_act = torch.nn.functional.conv2d(u,x,bias=b, padding=same_padding)     ## operation
y_act = y_act.mul(128)                                                   ## apply s_q
y_act = y_act.mul(2**(0))                                                ## apply s_o
y_act = y_act.div(128).add(0.5).floor()                                  ## apply f
#y_act = y_act.clamp(min=min_act, max=max_act)                            ## apply 8-bit clamp
y_act = torch.nn.functional.relu(y_act)                                  ## apply relu

## output emulation with -1/0 dictionary
allm1 = -torch.abs(x)     ## generate all -1 kernel
zeta = x.add(-1).div(2.0) ## generate zeta kernel

y_emu1 = torch.nn.functional.conv2d(2*u,zeta, bias=b,padding=same_padding) ## operation
y_emu1 = y_emu1.mul(128)                                                   ## apply s_q
y_emu1 = y_emu1.mul(2**(0))                                                ## apply s_o
y_emu1 = y_emu1.div(128).add(0.5).floor()                                  ## apply f
#y_emu1 = y_emu1.clamp(min=min_act, max=max_act)                            ## apply 8-bit clamp

y_emu2 = torch.nn.functional.conv2d(u   , allm1 , padding=same_padding)    ## operation
y_emu2 = y_emu2.mul(128)                                                   ## apply s_q
y_emu2 = y_emu2.mul(2**(0))                                                ## apply s_o
y_emu2 = y_emu2.div(128).add(0.5).floor()                                  ## apply f
#y_emu2 = y_emu2.clamp(min=min_act, max=max_act)                            ## apply 8-bit clamp

y_emu = y_emu1 - y_emu2                                                  ## residual subtract
y_emu = y_emu.add(0.5).floor()                                           ## apply f
#y_emu = y_emu.clamp(min=min_act, max=max_act)                            ## apply 8-bit clamp

y_emu  = torch.nn.functional.relu(y_emu)                                   ## apply relu

print('actual output:')
print('')
print(y_act)
print('')
print('emulated output:')
print('')
print(y_emu)
print('')
print('difference:', torch.sum(torch.abs(y_act-y_emu)).numpy())