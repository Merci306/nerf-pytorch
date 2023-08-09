import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_ 
# from .snn_cuda import LIFSpike

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class MLP(nn.Module):
    def __init__(self, input_ch, W, D, skips):
        super(MLP, self).__init__()
        self.skips = skips
        self.mlp = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )

    def forward(self, x):
        # Your forward pass implementation here
        return self.mlp

class LIFModule(nn.Module):
    def __init__(self, dim, lif_bias=True, proj_drop=0.,
                 lif=-1, lif_fix_tau=False, lif_fix_vth=False, lif_init_tau=0.25, lif_init_vth=-1.):
        super().__init__()
        self.dim = dim
        self.lif = lif
        self.fn1=nn.Linear(dim,dim,bias=lif_bias)
        self.fn2=nn.Linear(dim,dim,bias=lif_bias)
        self.fn3=nn.Linear(dim,dim,bias=lif_bias)
        self.fn4= nn.Linear(dim,dim,bias=lif_bias)
        self.fn4= nn.Linear(dim,dim,bias=lif_bias)
        self.fn5= nn.Linear(dim,dim,bias=lif_bias)
        self.actn = F.relu
        self.norm1= MyNorm(dim)
        self.norm2= MyNorm(dim)
        self.norm3= MyNorm(dim)
        self.lif1 = LIFSpike(lif=lif, fix_tau=lif_fix_tau, fix_vth=lif_fix_vth,
                               init_tau=lif_init_tau, init_vth=lif_init_vth, dim=2)
        self.lif2 = LIFSpike(lif=lif, fix_tau=lif_fix_tau, fix_vth=lif_fix_vth,
                               init_tau=lif_init_tau, init_vth=lif_init_vth, dim=3)
            
        

    def forward(self, x):
        x=self.fn1(x)
        x=self.norm1(x)
        x= self.actn(x)
        x = self.fn2(x)
        x = self.norm2(x)
        x = self.actn(x)
        x_lr = self.lif1(x)
        # x_lr = x
        # x_td = x
        x_td = self.lif2(x)
        x_lr = self.fn3(x_lr)
        x_td = self.fn4(x_td)
        x_lr = self.actn(x_lr)
        x_td = self.actn(x_td)
        x = x_lr + x_td
        x = self.norm3(x)
        x = self.fn5(x)
        return x
class PatchMerging(nn.Module):
    def __init__(self) -> None:
        super().__init__()
class PatchEmbed(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
class  LIFSpike(nn.Module):
    def __init__(self,
                 lif, fix_tau=False, fix_vth=False, init_tau=0.25, init_vth=-1., dim=2):
        super(LIFSpike, self).__init__()
        self.lif = lif
        self.dim = dim
        if fix_tau:
            self.tau = init_tau
        else:
            self.tau = torch.nn.Parameter(torch.Tensor([init_tau]))
        if fix_vth:
            self.Vth = init_vth
        else:
            self.Vth = torch.nn.Parameter(torch.Tensor([init_vth]))
        assert dim == 2 or dim == 3

    def forward(self, x):
        if self.lif == -1:
            return x
        out = _lif_cuda(x, self.tau, self.Vth, self.lif, self.dim)
        return out
def MyNorm(dim):
    return nn.GroupNorm(1, dim)      
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,norm_layer=nn.LayerNorm, lif_bias=True,lif=-1, lif_fix_tau=False, lif_fix_vth=False, lif_init_tau=0.25, lif_init_vth=-1.,drop=0.):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.norm = norm_layer(input_ch)
        self.pts_linears = MLP(input_ch, W, D, skips)
        self.lif_module = LIFModule(input_ch, lif_bias=lif_bias, proj_drop=drop,
                               lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                               lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)
        self.lif_module2 = LIFModule(W, lif_bias=lif_bias, proj_drop=drop,
                               lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                               lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)
        self.norm1 = norm_layer(W)
        self.norm2 = norm_layer(W//2)
        # self.pts_linears = nn.ModuleList(
        #     [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        h = self.norm(h)
        h = self.lif_module(h)
        for i, l in enumerate(self.pts_linears.mlp):
            

            h = self.pts_linears.mlp[i](h)
            h = self.norm1(h)
            h = self.lif_module2(h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        

        if self.use_viewdirs:
            H=h
            alpha=self.norm1(H)
            alpha = self.alpha_linear(H)
            feature = self.feature_linear(H)
            H = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                H = self.views_linears[i](H)
                H = F.relu(H)
            rgb = self.rgb_linear(H)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
# 2022.06.27-Changed for building SNN-MLP
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn

from collections import namedtuple
import cupy
from string import Template
import math

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_lif_kernel_h = kernel_loop + '''
extern "C"
__global__ void lif_forward_kernel_h(
const ${Dtype}* bottom_data, const ${Dtype}* tau, const ${Dtype}* vth, ${Dtype}* top_data, bool* flag, ${Dtype}* o) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int s = ${lif};
    const int lifh = (${height} + s - 1) / s;
    const int lifhh = ${height} / s;
    const int n = index / ${channels} / lifh / ${width};
    const int c = (index / lifh / ${width}) % ${channels};
    const int h = (index / ${width}) % lifh;
    const int w = index % ${width};

    ${Dtype} u = 0;

    const int offset = ((n * ${channels} + c) * ${height} + h * s) * ${width} + w;
    for(int j = 0; j < s; j++) {
      const int toffset = offset + j * ${width};
      if(toffset < ${numel} && h * s + j < lifhh * s) {
        u = tau[0] * o[toffset] + bottom_data[toffset];
        flag[toffset] = u > vth[0];
        if(j < s - 1)
          o[toffset + ${width}] = flag[toffset] ? 0 : u;
        top_data[toffset] = flag[toffset] ? u : vth[0];
      } else if(toffset < ${numel} && h * s + j < ${height}) {
        flag[toffset] = 1;
        top_data[toffset] = bottom_data[toffset];
      }
    }
  }
}
'''

_lif_kernel_w = kernel_loop + '''
extern "C"
__global__ void lif_forward_kernel_w(
const ${Dtype}* bottom_data, const ${Dtype}* tau, const ${Dtype}* vth, ${Dtype}* top_data, bool* flag, ${Dtype}* o) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int s = ${lif};
    const int lifw = (${width} + s - 1) / s;
    const int lifww = ${width} / s;
    const int n = index / ${channels} / ${height} / lifw;
    const int c = (index / ${height} / lifw) % ${channels};
    const int h = (index / lifw) % ${height};
    const int w = index % lifw;

    ${Dtype} u = 0;

    const int offset = ((n * ${channels} + c) * ${height} + h) * ${width} + w * s;
    for(int j = 0; j < s; j++) {
      const int toffset = offset + j;
      if(toffset < ${numel} && w * s + j < lifww * s) {
        u = tau[0] * o[toffset] + bottom_data[toffset];
        flag[toffset] = u > vth[0];
        if(j < s - 1)
          o[toffset + 1] = flag[toffset] ? 0 : u;
        top_data[toffset] = flag[toffset] ? u : vth[0];
      } else if(toffset < ${numel} && w * s + j < ${width}) {
        flag[toffset] = 1;
        top_data[toffset] = bottom_data[toffset];
      }
    }
  }
}
'''

_lif_kernel_backward_grad_input_h = kernel_loop + '''
extern "C"
__global__ void lif_backward_grad_input_kernel_h(
    const ${Dtype}* const top_diff, const ${Dtype}* const tau, const ${Dtype}* const flag, const ${Dtype}* const tmpo, ${Dtype}* const bottom_diff, ${Dtype}* const tau_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int s = ${lif};
    const int lifh = (${height} + s - 1) / s;
    const int lifhh = ${height} / s;
    const int n = index / ${channels} / lifh / ${width};
    const int c = (index / lifh / ${width}) % ${channels};
    const int h = (index / ${width}) % lifh;
    const int w = index % ${width};

    ${Dtype} tmp_bottom[${lif}];
    ${Dtype} tmp_tau[${lif}];
    
    const int offset = ((n * ${channels} + c) * ${height} + h * s) * ${width} + w;
    for(int j = 0; j < s; j++) {
      const int toffset = offset + j * ${width};
      if(toffset < ${numel} && h * s + j < lifhh * s) {
        tmp_bottom[j] = flag[toffset] * top_diff[toffset];
        if(j == 0) {
          tmp_tau[j] = 0;
        } else {
          tmp_tau[j] = tmp_tau[j - 1] * tau[0] * (1 - flag[toffset - ${width}]) + tmpo[toffset];
        }
        tau_diff[toffset] = top_diff[toffset] * flag[toffset] * tmp_tau[j];
      }
    }
    if(offset + (s - 1) * ${width} < ${numel} && h * s + s - 1 < lifhh * s)
      bottom_diff[offset + (s - 1) * ${width}] = tmp_bottom[s - 1];
    for(int j = s - 2; j >= 0; j--) {
      const int toffset = offset + j * ${width};
      if(toffset + ${width} < ${numel} && h * s + j + 1 < lifhh * s) {
        tmp_bottom[j] += tmp_bottom[j + 1] * (1 - flag[toffset]) * tau[0];
        bottom_diff[toffset] = tmp_bottom[j];
      } if(toffset < ${numel} && h * s + j < lifhh * s) {
        bottom_diff[toffset] = tmp_bottom[j];
      } else if(toffset < ${numel} && h * s + j < ${height}) {
        bottom_diff[toffset] = top_diff[toffset];
      }
    }
  }
}
'''

_lif_kernel_backward_grad_input_w = kernel_loop + '''
extern "C"
__global__ void lif_backward_grad_input_kernel_w(
    const ${Dtype}* const top_diff, const ${Dtype}* const tau, const ${Dtype}* const flag, const ${Dtype}* const tmpo, ${Dtype}* const bottom_diff, ${Dtype}* const tau_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int s = ${lif};
    const int lifw = (${width} + s - 1) / s;
    const int lifww = ${width} / s;
    const int n = index / ${channels} / ${height} / lifw;
    const int c = (index / ${height} / lifw) % ${channels};
    const int h = (index / lifw) % ${height};
    const int w = index % lifw;

    ${Dtype} tmp_bottom[${lif}];
    ${Dtype} tmp_tau[${lif}];

    const int offset = ((n * ${channels} + c) * ${height} + h) * ${width} + w * s;
    for(int j = 0; j < s; j++) {
      const int toffset = offset + j;
      if(toffset < ${numel} && w * s + j < lifww * s) {
        tmp_bottom[j] = flag[toffset] * top_diff[toffset];
        if(j == 0) {
          tmp_tau[j] = 0;
        } else {
          tmp_tau[j] = tmp_tau[j - 1] * tau[0] * (1 - flag[toffset - 1]) + tmpo[toffset];
        }
        tau_diff[toffset] = top_diff[toffset] * flag[toffset] * tmp_tau[j];
      }
    }
    if(offset + s - 1 < ${numel} && w * s + s - 1 < lifww * s)
        bottom_diff[offset + s - 1] = tmp_bottom[s - 1];
    for(int j = s - 2; j >= 0; j--) {
      const int toffset = offset + j;
      if(toffset + 1 < ${numel} && w * s + j + 1 < lifww * s) {
        tmp_bottom[j] += tmp_bottom[j + 1] * (1 - flag[toffset]) * tau[0];
        bottom_diff[toffset] = tmp_bottom[j];
      } else if(toffset < ${numel} && w * s + j < lifww * s) {
        bottom_diff[toffset] = tmp_bottom[j];
      } else if(toffset < ${numel} && w * s + j < ${width}) {
        bottom_diff[toffset] = top_diff[toffset];
      }
    }
  }
}
'''

class _lif_h(Function):
    @staticmethod
    def forward(ctx, input, tau, vth, lif, dim):
        assert input.dim() == 4 and input.is_cuda
        batch_size, channels, height, width = input.size()

        output = torch.zeros_like(input)#input.new(batch_size, channels, height, width)
        flag = torch.zeros_like(input).type(torch.bool)#input.new(batch_size, channels, height, width).type(torch.bool)
        tmpo = torch.zeros_like(input)#input.new(batch_size, channels, height, width)

        n = batch_size * channels * int(math.ceil(height / lif)) * width

        with torch.cuda.device_of(input):
            f = load_kernel('lif_forward_kernel_h', _lif_kernel_h, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels, 
                            height=height, width=width,
                            lif=lif, numel=output.numel()
                            )
            f(block=(CUDA_NUM_THREADS,1,1),
              grid=(GET_BLOCKS(n),1,1),
              args=[input.data_ptr(), tau.data_ptr(), vth.data_ptr(), output.data_ptr(), flag.data_ptr(), tmpo.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input, tau, vth, flag, tmpo)
        ctx.lif, ctx.dim = lif, dim
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        input, tau, vth, flag, tmpo = ctx.saved_tensors
        flag = flag.type(input.dtype)
        lif, dim = ctx.lif, ctx.dim
        batch_size, channels, height, width = input.size()

        grad_input = None
        grad_tau = None
        grad_vth = None
        n = batch_size * channels * int(math.ceil(height / lif)) * width

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels,
                   height=height, width=width, nthreads=n, numel=grad_output.numel(),
                   lif=lif
              )
        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_vth = ((1. - flag) * grad_output).sum().unsqueeze(0).contiguous()
                grad_input = torch.zeros_like(input)#input.new(input.size())
                grad_tau = torch.zeros_like(input)#input.new(input.size())

                f = load_kernel('lif_backward_grad_input_kernel_h',
                                _lif_kernel_backward_grad_input_h, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[grad_output.data_ptr(), tau.data_ptr(), flag.data_ptr(), tmpo.data_ptr(), grad_input.data_ptr(), grad_tau.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                grad_tau = grad_tau.sum().unsqueeze(0).contiguous()
        return grad_input, grad_tau, grad_vth, None, None
    

class _lif_w(Function):
    @staticmethod
    def forward(ctx, input, tau, vth, lif, dim):
        assert input.dim() == 4 and input.is_cuda
        batch_size, channels, height, width = input.size()

        output = torch.zeros_like(input)
        flag = torch.zeros_like(input).type(torch.bool)
        tmpo = torch.zeros_like(input)

        n = batch_size * channels * int(math.ceil(width / lif)) * height

        with torch.cuda.device_of(input):
            f = load_kernel('lif_forward_kernel_w', _lif_kernel_w, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels, 
                            height=height, width=width,
                            lif=lif, numel=output.numel()
                            )

            f(block=(CUDA_NUM_THREADS,1,1),
              grid=(GET_BLOCKS(n),1,1),
              args=[input.data_ptr(), tau.data_ptr(), vth.data_ptr(), output.data_ptr(), flag.data_ptr(), tmpo.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input, tau, vth, flag, tmpo)
        ctx.lif, ctx.dim = lif, dim
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        input, tau, vth, flag, tmpo = ctx.saved_tensors
        flag = flag.type(input.dtype)
        lif, dim = ctx.lif, ctx.dim
        batch_size, channels, height, width = input.size()

        grad_input = None
        grad_tau = None
        grad_vth = None
        n = batch_size * channels * int(math.ceil(width / lif)) * height

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels,
                   height=height, width=width, nthreads=n, numel=grad_output.numel(),
                   lif=lif
              )
        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_vth = ((1. - flag) * grad_output).sum().unsqueeze(0).contiguous()
                grad_input = torch.zeros_like(input)#input.new(input.size())
                grad_tau = torch.zeros_like(input)#input.new(input.size())
                f = load_kernel('lif_backward_grad_input_kernel_w',
                                _lif_kernel_backward_grad_input_w, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[grad_output.data_ptr(), tau.data_ptr(), flag.data_ptr(), tmpo.data_ptr(), grad_input.data_ptr(), grad_tau.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                grad_tau = grad_tau.sum().unsqueeze(0).contiguous()
        return grad_input, grad_tau, grad_vth, None, None


def _lif_cuda(input, tau, vth, lif, dim):
    """ involution kernel
    """
    assert dim == 2 or dim == 3

    if input.is_cuda:
        if dim == 2:
            out = _lif_h.apply(input, tau, vth, lif, dim)
        elif dim == 3:
            out = _lif_w.apply(input, tau, vth, lif, dim)
    else:
        raise NotImplementedError
    return out


class LIFSpike(nn.Module):
    def __init__(self,
                 lif, fix_tau=False, fix_vth=False, init_tau=0.25, init_vth=-1., dim=2):
        super(LIFSpike, self).__init__()
        self.lif = lif
        self.dim = dim
        if fix_tau:
            self.tau = init_tau
        else:
            self.tau = torch.nn.Parameter(torch.Tensor([init_tau]))
        if fix_vth:
            self.Vth = init_vth
        else:
            self.Vth = torch.nn.Parameter(torch.Tensor([init_vth]))
        assert dim == 2 or dim == 3

    def forward(self, x):
        if self.lif == -1:
            return x
        out = _lif_cuda(x, self.tau, self.Vth, self.lif, self.dim)
        return out



