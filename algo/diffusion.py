import torch
from torch import nn
import diffusers
from algo.diffusers.src.diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from algo.diffusers.src.diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from algo.diffusers.src.diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from networks.mlp import MLP
import math
from einops import rearrange, repeat

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLPWrapper(MLP):
    def __init__(self, channels, feature_dim, seq_length=1, *args, **kwargs):
        self.channels = channels
        self.seq_length = 1
        self.time_embedding_dim = 32
        input_dim = channels * seq_length + 32 + feature_dim
        super().__init__(input_dim = input_dim,*args, **kwargs)
        self.embedding = SinusoidalPosEmb(self.time_embedding_dim)
    
    def forward(self, x, cond, t):
        x = rearrange(x, 'b c l -> b (c l)')
        t = self.embedding(t)
        net_input = [x, cond, t]
        return rearrange(super().forward(torch.cat(net_input, dim = -1)), 'b (c l) -> b c l', l=self.seq_length)

class GaussianDiffusion1D(nn.Module):
    def __init__(self, model, config) -> None:
        super().__init__()
        self.config = config
        self.model = model
        if config.scheduler_type == 'DDPMScheduler':
            self.scheduler = DDPMScheduler(**config.scheduler)
        elif config.scheduler_type == 'EulerAncestralDiscreteScheduler':
            self.scheduler = EulerAncestralDiscreteScheduler(**config.scheduler)
        elif config.scheduler_type == 'EulerDiscreteScheduler':
            self.scheduler = EulerDiscreteScheduler(**config.scheduler)
        # self.scheduler.
        self.timesteps = config.scheduler.num_train_timesteps
        # self.scheduler.set_timesteps(self.timesteps, device=config.device)
        self.inference_timesteps = config.num_inference_timesteps
        self.prediction_type = config.scheduler.prediction_type
        
    def forward(self, *args, **kwargs):
        return self.calculate_loss(*args, **kwargs)
    
    def calculate_loss(self, x, cond):
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)
        noised_x = self.scheduler.add_noise(x, noise, t)
        
        pred = self.model(noised_x, cond, t)
        if self.prediction_type == 'epsilon':
            target = noise
        elif self.prediction_type == 'sample':
            target = x
        elif self.prediction_type == 'v_prediction':
            target = self.scheduler.get_velocity(x, noise, t)
        else:
            raise NotImplementedError()
        
        loss = (pred - target).square().mean()
        
        return loss
    
    def sample(self, cond):
        x = torch.randn(cond.shape[0], self.model.channels, self.model.seq_length, device=cond.device)
        self.scheduler.set_timesteps(self.inference_timesteps, device=cond.device)
        
        for t in self.scheduler.timesteps:
            t_pad = torch.full((x.shape[0],), t.item(), device=x.device, dtype=torch.long)
            model_output = self.model(x, cond, t_pad)
            x = self.scheduler.step(model_output, t, x).prev_sample
            
        return x
    
    def log_prob(self, x, cond):
        raise NotImplementedError()