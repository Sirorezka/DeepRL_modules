import torch
import torch.nn as nn
import torch.distributions as tdist

import math


class NoisyLinear(nn.Module):
    ## Factorized NoisyLinear
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ## weights: lin = W * X + b
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.ma_sigma = nn.Parameter(torch.Tensor(out_features))  
        self.mb_sigma = nn.Parameter(torch.Tensor(in_features)) 
        self.anoise = nn.Parameter(torch.tensor(out_features, dtype=torch.float))
        self.bnoise = nn.Parameter(torch.tensor(in_features, dtype=torch.float))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))            
        else:
            self.register_parameter('bias', None)


        self.reset_parameters()

    def reset_parameters(self):
        ## https://arxiv.org/pdf/1706.10295.pdf
        p = self.in_features
        sigma0 = 0.5
        sigma = sigma0 / math.sqrt(p)
        eps = 1e-1

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        nn.init.uniform_(self.ma_sigma, sigma * 0.99, sigma * 1.01)
        nn.init.uniform_(self.mb_sigma, sigma * 0.99, sigma * 1.01)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)      

    def forward(self, input):
        if not self.training:
          return nn.functional.linear(input, self.weight, self.bias)

        randa = self.anoise.data.normal_() * self.ma_sigma
        randa = torch.sign(randa) * torch.sqrt(torch.abs(randa))

        randb = self.bnoise.data.normal_() * self.mb_sigma
        randb = torch.sign(randb) * torch.sqrt(torch.abs(randb))

        randw = torch.matmul(randa.unsqueeze(1), randb.unsqueeze(0))

        weight = self.weight + randw
        bias = self.bias + randa

        return nn.functional.linear(input, weight, bias)



class NoisyLinear2(nn.Module):
    ## Unfactorized
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ## weights
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.ma_sigma = nn.Parameter(torch.Tensor(out_features, in_features))  
        self.anoise = nn.Parameter(torch.Tensor(out_features,in_features))
        self.bnoise = nn.Parameter(torch.Tensor(out_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=False)
            self.mb_sigma = nn.Parameter(torch.Tensor(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            self.register_parameter('mb_sigma', None)

        self.reset_parameters()

    def reset_parameters(self):
        ## https://arxiv.org/pdf/1706.10295.pdf
        p = self.in_features
        sigma0 = 0.017

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.ma_sigma, sigma0)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            nn.init.constant_(self.mb_sigma, sigma0)            

    def forward(self, input):
        if not self.training:
          return nn.functional.linear(input, self.weight, self.bias)

        normrand =  self.anoise.data.normal_() * self.ma_sigma    
        weight = self.weight + normrand

        normrand =  self.bnoise.data.normal_() * self.mb_sigma        
        bias = self.bias + normrand

        return nn.functional.linear(input, weight, bias)
