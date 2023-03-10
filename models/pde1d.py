"""
pde1d.py

PDE VAE model (PDEAutoEncoder module) for fitting data with 1 spatial dimension.
"""
import collections
from itertools import repeat

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_single = _ntuple(1, "_single")

def unfold(x, axis=-2, kernel_size=3, strides=1):
    # print(x.shape)
    if len(x.shape) == 3:
        N = None
        C, H, W = x.shape
        x = x.reshape([1, C, H, W])
    else:
        N, C, H, W = x.shape
    
    kernel_size = [1, kernel_size] if axis==-1 else [kernel_size, 1]
    
    if N is None:
        x = F.unfold(x, kernel_sizes=kernel_size, strides=strides).squeeze_()
        x = x.reshape([C, max(kernel_size), -1])
        x = x.transpose([0, 2, 1])
        x = x.reshape([C, (H-kernel_size[0])//strides+1, (W-kernel_size[1])//strides+1, max(kernel_size)])
    else:
        x = F.unfold(x, kernel_sizes=kernel_size, strides=strides)
        x = x.reshape([N, C, max(kernel_size), -1])
        x = x.transpose([0, 1, 3, 2])
        x = x.reshape([N, C, (H-kernel_size[0])//strides+1, (W-kernel_size[1])//strides+1, max(kernel_size)])
        
    return x

class PeriodicPad1d(nn.Layer):
    def __init__(self, pad, dim=-1):
        super().__init__()
        self.pad = pad
        self.dim = dim

    def forward(self, x):
        if self.pad > 0:
            front_padding = x.slice([self.dim], [x.shape[self.dim]-self.pad], [x.shape[self.dim]])
            back_padding = x.slice([self.dim], [0], [self.pad])
            x = paddle.concat([front_padding, x, back_padding], axis=self.dim)

        return x

class AntiReflectionPad1d(nn.Layer):
    def __init__(self, pad, dim=-1):
        super().__init__()
        self.pad = pad
        self.dim = dim

    def forward(self, x):
        if self.pad > 0:
            front_padding = -x.slice([self.dim], [0], [self.pad]).flip([self.dim])
            back_padding = -x.slice([self.dim], [x.shape[self.dim]-self.pad], [x.shape[self.dim]]).flip([self.dim])
            x = paddle.concat([front_padding, x, back_padding], axis=self.dim)

        return x


class DynamicConv1d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, boundary_cond='periodic'):

        super().__init__()

        self.kernel_size = _single(kernel_size) # x
        # x
        self.stride = _single(stride) # not implemented
        self.padding = _single(padding)
        self.dilation = _single(dilation) # not implemented

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.boundary_cond = boundary_cond

        if self.padding[0] > 0 and boundary_cond == 'periodic':
            assert self.padding[0] == int((self.kernel_size[0]-1)/2)
            self.pad = PeriodicPad1d(self.padding[0], dim=-2)
        else:
            self.pad = None

    def forward(self, input, weight, bias):
        # x
        # print(input.shape)
        y = input.transpose([0, 2, 1])

        if self.pad is not None:
            output_size = input.shape[-1]
            y = self.pad(y)
        else:
            output_size = input.shape[-1] - (self.kernel_size[0]-1)
        # print(y.shape)
        image_patches = unfold(y, axis=-2, kernel_size=self.kernel_size[0], strides=self.stride[0])
        image_patches = image_patches.transpose([0, 1, 3, 2]) # x
        y = paddle.matmul(image_patches.reshape([-1, output_size, self.in_channels * self.kernel_size[0]]),
                          weight.reshape([-1, self.in_channels * self.kernel_size[0], self.out_channels])
                        )
        if bias is not None:
            y = y + bias.reshape([-1, 1, self.out_channels])

        return y.transpose([0, 2, 1])


class ConvPropagator(nn.Layer):
    def __init__(self, hidden_channels, linear_kernel_size, nonlin_kernel_size, data_channels, stride=1,
                 linear_padding=0, nonlin_padding=0, dilation=1, groups=1, prop_layers=1, prop_noise=0., boundary_cond='periodic'):

        self.data_channels = data_channels
        self.prop_layers = prop_layers
        self.prop_noise = prop_noise
        self.boundary_cond = boundary_cond

        assert nonlin_padding == int((nonlin_kernel_size-1)/2)
        if boundary_cond == 'crop' or boundary_cond == 'dirichlet0':
            self.padding = int((2+prop_layers)*nonlin_padding)

        super(ConvPropagator, self).__init__()

        self.conv_linear = DynamicConv1d(data_channels, data_channels, linear_kernel_size, stride,
                                    linear_padding, dilation, groups, boundary_cond) if linear_kernel_size > 0 else None

        self.conv_in = DynamicConv1d(data_channels, hidden_channels, nonlin_kernel_size, stride,
                                    nonlin_padding, dilation, groups, boundary_cond)

        self.conv_out = DynamicConv1d(hidden_channels, data_channels, nonlin_kernel_size, stride,
                                    nonlin_padding, dilation, groups, boundary_cond)

        if prop_layers > 0:
            self.conv_prop = nn.LayerList([DynamicConv1d(hidden_channels, hidden_channels, nonlin_kernel_size, stride,
                                            nonlin_padding, dilation, groups, boundary_cond)
                                            for i in range(prop_layers)])

        # self.cutoff = Parameter(torch.Tensor([1]))
        cutoff = paddle.to_tensor([1.])
        self.cutoff = paddle.create_parameter(shape=cutoff.shape,
                                              dtype=str(cutoff.numpy().dtype),
                                              default_initializer=paddle.nn.initializer.Assign(cutoff))

    def _target_pad_1d(self, y, y0):
        return paddle.concat([y0[:,:,:self.padding], y, y0[:,:,-self.padding:]], axis=-1)

    def _antireflection_pad_1d(self, y, dim):
        front_padding = -y.slice([dim], [0], [self.padding]).flip([dim])
        back_padding = -y.slice([dim], [y.shape[dim]-self.padding], [y.shape[dim]]).flip([dim])
        return paddle.concat([front_padding, y, back_padding], axis=dim)
        
    def _f(self, y, linear_weight, linear_bias, in_weight, in_bias, 
                    out_weight, out_bias, prop_weight, prop_bias):
        y_lin = self.conv_linear(y, linear_weight, linear_bias) if self.conv_linear is not None else 0

        y = self.conv_in(y, in_weight, in_bias)
        y = F.relu(y)
        for j in range(self.prop_layers):
            y = self.conv_prop[j](y, prop_weight[:,j], prop_bias[:,j])
            y = F.relu(y)
        y = self.conv_out(y, out_weight, out_bias)

        return y + y_lin

    def forward(self, y0, linear_weight, linear_bias, 
                in_weight, in_bias, out_weight, out_bias, prop_weight, prop_bias, depth):
        if self.boundary_cond == 'crop':
            # requires entire target solution as y0 for padding purposes
            assert len(y0.shape) == 4
            assert y0.shape[1] == self.data_channels
            assert y0.shape[2] == depth
            y_pad = y0[:,:,0]
            y = y0[:,:,0, self.padding:-self.padding]
        elif self.boundary_cond == 'periodic' or self.boundary_cond == 'dirichlet0':
            assert len(y0.shape) == 3
            assert y0.shape[1] == self.data_channels
            y = y0
        else:
            raise ValueError("Invalid boundary condition.")

        f = lambda y: self._f(y, linear_weight, linear_bias, in_weight, in_bias, 
                                        out_weight, out_bias, prop_weight, prop_bias)

        y_list = []
        for i in range(depth):
            if self.boundary_cond == 'crop':
                if i > 0:
                    y_pad = self._target_pad_1d(y, y0[:,:,i])
            elif self.boundary_cond == 'dirichlet0':
                y_pad = self._antireflection_pad_1d(y, -1)
            elif self.boundary_cond == 'periodic':
                y_pad = y

            ### Euler integrator
            dt = 1e-6 # NOT REAL TIME STEP, JUST HYPERPARAMETER
            noise = self.prop_noise * paddle.randn(y.shape) if (self.training and self.prop_noise > 0) else 0 # x
            y = y + self.cutoff * paddle.tanh((dt * f(y_pad)) / self.cutoff) + noise

            y_list.append(y)

        return paddle.stack(y_list, axis=-2)


class PDEAutoEncoder(nn.Layer):
    def __init__(self, param_size=1, data_channels=1, data_dimension=1, hidden_channels=16, 
                        linear_kernel_size=0, nonlin_kernel_size=5, prop_layers=1, prop_noise=0., 
                        boundary_cond='periodic', param_dropout_prob=0.1, debug=False):

        assert data_dimension == 1
        
        super(PDEAutoEncoder, self).__init__()

        self.param_size = param_size
        self.data_channels = data_channels
        self.hidden_channels = hidden_channels
        self.linear_kernel_size = linear_kernel_size
        self.nonlin_kernel_size = nonlin_kernel_size
        self.prop_layers = prop_layers
        self.boundary_cond = boundary_cond
        self.param_dropout_prob = param_dropout_prob
        self.debug = debug

        if param_size > 0:
            ### 2D Convolutional Encoder
            if boundary_cond =='crop' or boundary_cond == 'dirichlet0':
                pad_input = [0, 0, 0, 0]
                pad_func = PeriodicPad1d # can be anything since no padding is added
            elif boundary_cond == 'periodic':
                pad_input = [1, 2, 4, 8]
                pad_func = PeriodicPad1d
            else:
                raise ValueError("Invalid boundary condition.")

            self.encoder = nn.Sequential(   pad_func(pad_input[0]),
                                            nn.Conv2D(data_channels, 4, kernel_size=3, dilation=1),
                                            nn.ReLU(),

                                            pad_func(pad_input[1]),
                                            nn.Conv2D(4, 16, kernel_size=3, dilation=2),
                                            nn.ReLU(),

                                            pad_func(pad_input[2]),
                                            nn.Conv2D(16, 64, kernel_size=3, dilation=4),
                                            nn.ReLU(),

                                            pad_func(pad_input[3]),
                                            nn.Conv2D(64, 64, kernel_size=3, dilation=8),
                                            nn.ReLU(),
                                            )
            self.encoder_to_param = nn.Sequential(nn.Conv2D(64, param_size, kernel_size=1, stride=1))
            self.encoder_to_logvar = nn.Sequential(nn.Conv2D(64, param_size, kernel_size=1, stride=1))

            ### Parameter to weight/bias for dynamic convolutions
            if linear_kernel_size > 0:
                self.param_to_linear_weight = nn.Sequential( nn.Linear(param_size, 4 * data_channels * data_channels),
                                        nn.ReLU(),
                                        nn.Linear(4 * data_channels * data_channels, 
                                                    data_channels * data_channels * linear_kernel_size)
                                        )

            self.param_to_in_weight = nn.Sequential( nn.Linear(param_size, 4 * data_channels * hidden_channels),
                                    nn.ReLU(),
                                    nn.Linear(4 * data_channels * hidden_channels, 
                                                data_channels * hidden_channels * nonlin_kernel_size)
                                    )
            self.param_to_in_bias = nn.Sequential( nn.Linear(param_size, 4 * hidden_channels),
                                    nn.ReLU(),
                                    nn.Linear(4 * hidden_channels, hidden_channels)
                                    )

            self.param_to_out_weight = nn.Sequential( nn.Linear(param_size, 4 * data_channels * hidden_channels),
                                    nn.ReLU(),
                                    nn.Linear(4 * data_channels * hidden_channels, 
                                                data_channels * hidden_channels * nonlin_kernel_size)
                                    )
            self.param_to_out_bias = nn.Sequential( nn.Linear(param_size, 4 * data_channels),
                                    nn.ReLU(),
                                    nn.Linear(4 * data_channels, data_channels)
                                    )

            if prop_layers > 0:
                self.param_to_prop_weight = nn.Sequential( nn.Linear(param_size, 4 * prop_layers * hidden_channels * hidden_channels),
                                        nn.ReLU(),
                                        nn.Linear(4 * prop_layers * hidden_channels * hidden_channels, 
                                                    prop_layers * hidden_channels * hidden_channels * nonlin_kernel_size)
                                        )
                self.param_to_prop_bias = nn.Sequential( nn.Linear(param_size, 4 * prop_layers * hidden_channels),
                                        nn.ReLU(),
                                        nn.Linear(4 * prop_layers * hidden_channels, prop_layers * hidden_channels)
                                        )

        ### Decoder/PDE simulator
        self.decoder = ConvPropagator(hidden_channels, linear_kernel_size, nonlin_kernel_size, data_channels, 
                                        linear_padding=int((linear_kernel_size-1)/2), 
                                        nonlin_padding=int((nonlin_kernel_size-1)/2), 
                                        prop_layers=prop_layers, prop_noise=prop_noise, boundary_cond=boundary_cond)
        
    def forward(self, x, y0, depth):

        if self.param_size > 0:
            assert len(x.shape) == 4
            assert x.shape[1] == self.data_channels

            ### 2D Convolutional Encoder
            encoder_out = self.encoder(x)
            
            logvar = self.encoder_to_logvar(encoder_out)
            logvar_size = logvar.shape
            logvar = logvar.reshape([logvar_size[0], logvar_size[1], -1])
            params = self.encoder_to_param(encoder_out).reshape([logvar_size[0], logvar_size[1], -1])

            if self.debug:
                raw_params = params

            # Parameter Spatial Averaging Dropout
            if self.training and self.param_dropout_prob > 0:
                mask = paddle.bernoulli(paddle.full_like(logvar, self.param_dropout_prob))
                mask[mask > 0] = float("inf")
                logvar = logvar + mask

            # Inverse variance weighted average of params
            weights = F.softmax(-logvar, axis=-1)
            params = (params * weights).sum(axis=-1)

            # Compute logvar for inverse variance weighted average with a correlation length correction
            correlation_length = 31 # estimated as receptive field of the convolutional encoder
            logvar = -paddle.logsumexp(-logvar, axis=-1) \
                        + paddle.log(paddle.to_tensor(
                            max(1, (1 - self.param_dropout_prob)
                                * min(correlation_length, logvar_size[-2])
                                * min(correlation_length, logvar_size[-1])),
                            dtype=logvar.dtype))

            ### Variational autoencoder reparameterization trick
            if self.training:
                stdv = (0.5 * logvar).exp()

                # Sample from unit normal
                z = params + stdv * paddle.randn(stdv.shape)
            else:
                z = params

            ### Parameter to weight/bias for dynamic convolutions
            if self.linear_kernel_size > 0:
                linear_weight = self.param_to_linear_weight(z)
                linear_bias = None
            else:
                linear_weight = None
                linear_bias = None

            in_weight = self.param_to_in_weight(z)
            in_bias = self.param_to_in_bias(z)

            out_weight = self.param_to_out_weight(z)
            out_bias = self.param_to_out_bias(z)

            if self.prop_layers > 0:
                prop_weight = self.param_to_prop_weight(z).reshape([-1, self.prop_layers,
                                    self.hidden_channels * self.hidden_channels * self.nonlin_kernel_size])
                prop_bias = self.param_to_prop_bias(z).reshape([-1, self.prop_layers, self.hidden_channels])
            else:
                prop_weight = None
                prop_bias = None

        else: # if no parameter used
            linear_weight = None
            linear_bias = None
            in_weight = None
            in_bias = None
            out_weight = None
            out_bias = None
            prop_weight = None
            prop_bias = None
            params = None
            logvar = None

        ### Decoder/PDE simulator
        y = self.decoder(y0, linear_weight, linear_bias, in_weight, in_bias, out_weight, out_bias, 
                                prop_weight, prop_bias, depth)

        if self.debug:
            return y, params, logvar, [in_weight, in_bias, out_weight, out_bias, prop_weight, prop_bias], \
                    weights.reshape([logvar_size]), raw_params.reshape([logvar_size])

        return y, params, logvar
