###########################################################################
# Computer vision - Binary neural networks demo software by HyperbeeAI.   #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. main@shallow.ai #
###########################################################################
import torch.nn as nn
import layers

class maxim_nas(nn.Module):
    def __init__(
            self,
            num_classes  =100,
            num_channels =3,
            dimensions   =(32, 32),
            bias         =True,
            **kwargs
    ):
        super().__init__()

        ### Burak: disable word wrap in your editor to see this "table" for conv layers properly 
        ###        mark that all layers are 'same' padding.
        ###                          input ch     | out ch | kernel dim | padding
        self.conv1_1 = layers.conv( num_channels,     64,     3,           1,   pooling=False, batchnorm=True, activation='relu')
        self.conv1_2 = layers.conv(           64,     32,     1,           0,   pooling=False, batchnorm=True, activation='relu')
        self.conv1_3 = layers.conv(           32,     64,     3,           1,   pooling=False, batchnorm=True, activation='relu')
        self.conv2_1 = layers.conv(           64,     32,     3,           1,   pooling=True , batchnorm=True, activation='relu')
        self.conv2_2 = layers.conv(           32,     64,     1,           0,   pooling=False, batchnorm=True, activation='relu')
        self.conv3_1 = layers.conv(           64,    128,     3,           1,   pooling=True , batchnorm=True, activation='relu')
        self.conv3_2 = layers.conv(          128,    128,     1,           0,   pooling=False, batchnorm=True, activation='relu')
        self.conv4_1 = layers.conv(          128,     64,     3,           1,   pooling=True , batchnorm=True, activation='relu')
        self.conv4_2 = layers.conv(           64,    128,     3,           1,   pooling=False, batchnorm=True, activation='relu')
        self.conv5_1 = layers.conv(          128,    128,     1,           0,   pooling=True , batchnorm=True, activation='relu')
        self.fc      = layers.fullyconnected(512, num_classes, output_width_30b=True, pooling=False, batchnorm=False, activation=None)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv5_1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class maxim_nas_large(nn.Module):
    def __init__(
            self,
            num_classes  =100,
            num_channels =3,
            dimensions   =(32, 32),
            bias         =True,
            **kwargs
    ):
        super().__init__()

        ### Burak: disable word wrap in your editor to see this "table" for conv layers properly 
        ###        mark that all layers are 'same' padding.
        ###                          input ch     | out ch | kernel dim | padding
        self.conv1_1 = layers.conv( num_channels,    128,     3,           1,   pooling=False, batchnorm=True, activation='relu')
        self.conv1_2 = layers.conv(          128,    128,     1,           0,   pooling=False, batchnorm=True, activation='relu')
        self.conv1_3 = layers.conv(          128,    256,     3,           1,   pooling=False, batchnorm=True, activation='relu')
        self.conv2_1 = layers.conv(          256,    128,     3,           1,   pooling=True , batchnorm=True, activation='relu')
        self.conv2_2 = layers.conv(          128,    128,     1,           0,   pooling=False, batchnorm=True, activation='relu')
        self.conv3_1 = layers.conv(          128,     64,     3,           1,   pooling=True , batchnorm=True, activation='relu')
        self.conv3_2 = layers.conv(           64,    256,     1,           0,   pooling=False, batchnorm=True, activation='relu')
        self.conv4_1 = layers.conv(          256,    128,     3,           1,   pooling=True , batchnorm=True, activation='relu')
        self.conv4_2 = layers.conv(          128,     64,     3,           1,   pooling=False, batchnorm=True, activation='relu')
        self.conv5_1 = layers.conv(           64,    128,     1,           0,   pooling=True , batchnorm=True, activation='relu')
        self.fc      = layers.fullyconnected(512, num_classes, output_width_30b=True, pooling=False, batchnorm=False, activation=None)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv5_1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class maxim_simplenet(nn.Module):
    def __init__(
            self,
            num_classes  =100,
            num_channels =3,
            dimensions   =(32, 32),
            bias         =True,
            **kwargs
    ):
        super().__init__()

        ### Burak: disable word wrap in your editor to see this "table" for conv layers properly 
        ###        mark that all layers are 'same' padding.
        ###                       input ch      | out ch  | kernel dim | padding
        self.conv1 = layers.conv( num_channels,     16,       3,           1, pooling=False, batchnorm=True, activation='relu')
        self.conv2 = layers.conv(           16,     20,       3,           1, pooling=False, batchnorm=True, activation='relu')
        self.conv3 = layers.conv(           20,     20,       3,           1, pooling=False, batchnorm=True, activation='relu')
        self.conv4 = layers.conv(           20,     20,       3,           1, pooling=False, batchnorm=True, activation='relu')
        self.conv5 = layers.conv(           20,     20,       3,           1, pooling=True , batchnorm=True, activation='relu')
        self.conv6 = layers.conv(           20,     20,       3,           1, pooling=False, batchnorm=True, activation='relu')
        self.conv7 = layers.conv(           20,     44,       3,           1, pooling=False, batchnorm=True, activation='relu')
        self.conv8 = layers.conv(           44,     48,       3,           1, pooling=True , batchnorm=True, activation='relu')
        self.conv9 = layers.conv(           48,     48,       3,           1, pooling=False, batchnorm=True, activation='relu')
        self.conv10= layers.conv(           48,     96,       3,           1, pooling=True , batchnorm=True, activation='relu')
        self.conv11= layers.conv(           96,    512,       1,           0, pooling=True , batchnorm=True, activation='relu')
        self.conv12= layers.conv(          512,    128,       1,           0, pooling=False, batchnorm=True, activation='relu')
        self.conv13= layers.conv(          128,    128,       3,           1, pooling=True , batchnorm=True, activation='relu')
        self.conv14= layers.conv(          128, num_classes,  1,           0, output_width_30b=True, pooling=False, batchnorm=False, activation=None) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = x.view(x.size(0), -1)
        return x