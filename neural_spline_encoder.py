import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

K = 20  # 100

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


# Taken from: https://github.com/bayesiains/nsf/blob/master/nde/transforms/splines/rational_quadratic.py
def unconstrained_rational_quadratic_spline(inputs,
                                            unnormalized_widths,
                                            unnormalized_heights,
                                            unnormalized_derivatives,
                                            inverse=False,
                                            tails='linear',
                                            tail_bound=1.,
                                            min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                            min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                            min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == 'linear':
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))

    # print(inside_interval_mask)
    # print(unnormalized_widths.shape)

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    return outputs, logabsdet

# Taken from: https://github.com/bayesiains/nsf/blob/master/nde/transforms/splines/rational_quadratic.py
def rational_quadratic_spline(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse=False,
                              left=0., right=1., bottom=0., top=1.,
                              min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                              min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                              min_derivative=DEFAULT_MIN_DERIVATIVE):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise transforms.InputOutsideDomain()

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives
                                             + input_derivatives_plus_one
                                             - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives
                                              + input_derivatives_plus_one
                                              - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet  # -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2)
                                     + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet


def get_layers(D, hidden, is_spline=False):
    if is_spline:
        hidden = hidden + [D*(3*K-1)]
    else:
        hidden = hidden + [D]
    ret = [nn.Linear(D, hidden[0])]
    for i in range(len(hidden) - 1):
        ret += [nn.ReLU(), nn.Linear(hidden[i], hidden[i+1])]
    return ret


class Spline(nn.Module):

    def __init__(self, D, hidden):
        super(Spline, self).__init__()
        layers = get_layers(D, hidden, is_spline=True)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Scale(nn.Module):

    def __init__(self, D, hidden):
        super(Scale, self).__init__()
        layers = get_layers(D, hidden) + [nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Translate(nn.Module):

    def __init__(self, D, hidden):
        super(Translate, self).__init__()
        layers = get_layers(D, hidden)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FlowLayer(nn.Module):

    def __init__(self, D, hidden, mask=None):
        super(FlowLayer, self).__init__()
        self.D = D
        self.hidden = hidden
        # self.s = Scale(D, hidden)
        # self.t = Translate(D, hidden)
        self.spline = Spline(D, hidden)
        mask = torch.from_numpy(mask)
        mask.requires_grad_(False)
        self.register_buffer('mask', mask)

    def forward(self, z):
        spline_params = self.spline(self.mask * z)
        spline_params = spline_params.reshape(-1, z.shape[1], 3*K-1)
        widths, heights, derivatives = spline_params[:, :, :K], spline_params[:, :, K:2*K], spline_params[:, :, 2*K:]
        x_tmp, logdet_tmp = unconstrained_rational_quadratic_spline(z, widths, heights, derivatives, False, tail_bound=10.0)
        x = self.mask * z + (1 - self.mask) * x_tmp
        logdet = torch.sum(logdet_tmp * (1 - self.mask), dim=1)
        # s = self.s(self.mask * z)
        # t = self.t(self.mask * z)
        # x = self.mask * z + (1 - self.mask) * (z * torch.exp(s) + t)
        # logdet = torch.sum(s * (1 - self.mask), dim=1)
        return x, logdet

    def inverse(self, x):
        spline_params = self.spline(self.mask * x)
        spline_params = spline_params.reshape(-1, x.shape[1], 3*K-1)
        widths, heights, derivatives = spline_params[:, :, :K], spline_params[:, :, K:2*K], spline_params[:, :, 2*K:]
        z_tmp, logdet_tmp = unconstrained_rational_quadratic_spline(x, widths, heights, derivatives, True, tail_bound=10.0)
        z = self.mask * x + (1 - self.mask) * z_tmp
        logdet = torch.sum(logdet_tmp * (1 - self.mask), dim=1)
        # print(self.mask.shape, z.shape, logdet.shape)
        # exit(0)
        # s = self.s(self.mask * x)
        # t = self.t(self.mask * x)
        # z = self.mask * x + (1 - self.mask) * ((x - t) * torch.exp(-s))
        # logdet = torch.sum(s * (1 - self.mask), dim=1)
        return z, logdet


class BatchNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.batch_mean = None
        self.batch_var = None

    def inverse(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps
            self.set_batch_stats_func(x.detach())
        else:
            m = self.batch_mean
            v = self.batch_var

        z = (x - m) / torch.sqrt(v)
        z = z * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(-self.gamma + 0.5 * torch.log(v))
        return z, log_det

    def forward(self, z):
        m = self.batch_mean
        v = self.batch_var
        x = (z - self.beta) * torch.exp(-self.gamma) * torch.sqrt(v) + m
        log_det = torch.sum(-self.gamma + 0.5 * torch.log(v))
        return x, log_det

    def set_batch_stats_func(self, x):
        if self.batch_mean is None:
            self.batch_mean = x.mean(dim=0)
            self.batch_var = x.var(dim=0) + self.eps
        else:
            self.batch_mean = 0.9 * self.batch_mean + 0.1 * x.mean(dim=0)
            self.batch_var = 0.9 * self.batch_var + 0.1 * (x.var(dim=0) + self.eps)


class NeuralSplineEncoder(nn.Module):

    def __init__(self, p_x, D, hidden, k, masks=None):
        super(NeuralSplineEncoder, self).__init__()
        self.p_x = p_x
        self.D = D
        self.hidden = hidden
        self.k = k
        if masks is None:
            masks = [
                np.array([(j + i) % 2 for j in range(D)])
                for i in range(k)
            ]
        self.layers = []
        for i in range(k):
            self.layers += [FlowLayer(D, hidden, masks[i])]
        self.layers = nn.ModuleList(self.layers)

    def inverse(self, x):
        if self.p_x is not None:
            sum_logp = self.p_x.log_prob(x)
        else:
            sum_logp = 0
        z = x
        for layer in reversed(self.layers):
            z, logdet = layer.inverse(z)
            sum_logp = sum_logp + logdet
        return z, sum_logp

    def forward(self, z):
        sum_logp = 0
        x = z
        for layer in self.layers:
            x, logdet = layer.forward(x)
            sum_logp = sum_logp + logdet
        if self.p_x is not None:
            sum_logp = sum_logp + self.p_x.log_prob(x)
        return x, sum_logp
