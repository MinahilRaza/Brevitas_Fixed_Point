# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author: Minahil Raza


from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
import torch


total_bits = 8   #width for weights and activations
n = 7            #fractional part
class LeNet5(Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = qnn.QuantConv2d(in_channels= 1,
                                     out_channels= 20,
                                     kernel_size= 3,
                                     padding= 1,
                                     bias= False,
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width= total_bits,
                                     weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                     weight_scaling_impl_type=ScalingImplType.CONST,
                                     weight_scaling_const=1.0)
        self.relu1 = qnn.QuantReLU(quant_type=QuantType.INT, 
                                   bit_width=8, 
                                   max_val= 1- 1/128.0,
                                   restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                   scaling_impl_type=ScalingImplType.CONST )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.conv2 = qnn.QuantConv2d(in_channels= 20,
                                     out_channels= 50,
                                     kernel_size= 3,
                                     padding= 1,
                                     bias= False,
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width=8,
                                     weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                     weight_scaling_impl_type=ScalingImplType.CONST,
                                     weight_scaling_const=1.0 )

        self.relu2 = qnn.QuantReLU(quant_type=QuantType.INT, 
                                   bit_width=8, 
                                   max_val= 1- 1/128.0,
                                   restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                   scaling_impl_type=ScalingImplType.CONST )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        """
        # for 32-bit precision FC layers
        self.fc1   = nn.Linear(7*7*50, 500)

        self.relu3 = nn.ReLU()

        self.fc2   = nn.Linear(500,10)

        """
        

        # for fixed-point precision FC layers
        self.fc1   = qnn.QuantLinear(7*7*50, 500,
                                     bias= True,
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width=32,
                                     weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                     weight_scaling_impl_type=ScalingImplType.CONST,
                                     weight_scaling_const=1.0)
        

        self.relu3 = qnn.QuantReLU(quant_type=QuantType.INT, 
                                   bit_width=8, 
                                   max_val= 1- 1/128.0,
                                   restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                   scaling_impl_type=ScalingImplType.CONST )

        self.fc2   = qnn.QuantLinear(500, 10,
                                     bias= True,
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width=8,
                                     weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                     weight_scaling_impl_type=ScalingImplType.CONST,
                                     weight_scaling_const=1.0)
    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out



