"""Paper reference snapshot: TEDDCF + adaptive Self/Delay S0 + BGST.

This file defines the complete forward architecture and input preprocessing.
It has no training/evaluation/checkpoint entry point and does not start jobs.
Classes/functions below are verbatim AST-selected definitions from the project,
with imports consolidated and UpstreamTEDDCF aliased to the original TEDDCF.
Train-fitted climate, channel-specific directed delay graphs and BGST candidate
supports are external frozen inputs; graph fitting is outside model forward.
Unused modules inherited from upstream are retained for state_dict fidelity:
pre_h and adp_f are constructed but NOT called by JointNative.forward.
Only SelfExpert and DelayExpert are instantiated; no periodic/local expert.
Source files may contain those other experts, but they are excluded here.

DATA FLOW (all flow predictions before final inverse scaling are standardized):
  Raw history X: [B,N,12,2], channel0=destination_count=inflow,
  channel1=origin_count=outflow; N=1024 in the actual experiments.
  CPU coif1 level1 DWT on time axis -> reconstructed low/high -> per-channel
  train-window mean/std normalization of each reconstructed branch.
  Clock: [B,N,12,2] time-of-day and weekday, auxiliary to two flow channels.
  Each branch: concat flow+clock -> Conv2d(4,96,1x1); clock embedding96;
  concatenate embeddings -> [B,192,N,12]. Both branches share start_conv/Temb.
  Low: 3 temporal attention layers -> TrendGCN -> Z_L [B,192,N,12];
       GLU(Z_L)+Z_L is used by backbone regression, NOT the expert Z_L tap.
  High: causal TemporalConvNet -> EventGCN -> Z_H [B,192,N,12].
  Base: ReLU(Z_H+GLU(Z_L)+Z_L) -> Conv2d(192,24,1x12)
        -> reshape [B,N,12,2] = Y_base.
  Self: shared node-independent GRU([Z_L+Z_H, raw channel history]);
        last hidden -> Linear/Tanh/Linear -> Linear(192,12), each channel
        call returns [B,12,N,1]; concatenate/transpose -> S [B,N,12,2].
  Delay: shared lag embedding and evidence encoder on Z_H plus per-channel
         evidence; frozen channel-specific directed multilag edges; weighted
         source-to-target aggregation -> Linear(192,12); same joint layout D.
  Gate: [mean_t Z_L, mean_t Z_H, last trend/high-frequency evidence per flow]
        [B,N,388] -> Linear(388,32)/SiLU/Linear(32,4)
        -> [B,N,2 flows,2 experts] -> softmax OVER EXPERTS.
        Gate is sample/node/channel dependent, shared over all 12 horizons.
  R_S = w_S*(S-Y_base); R_D = w_D*D.
  BGST: mean_t Z_L -> Linear(192,32)/LayerNorm -> background context;
        Z_H -> Conv2d(192,32,1) -> depthwise causal temporal branches
        dilations1/2/4 (receptive fields3/5/9) -> [B,N,3,12,32].
        Background FiLM conditions query/key; sparse neighbors K<=8,
        lags0/1/2/3; score includes q,k,q*k,q-k, lag,scale,horizon embeddings.
        For each horizon softmax jointly over neighbor/scale/lag candidates
        PLUS zero-logit null candidate (null contributes zero message).
        Weighted projected source values -> shared Linear(32,2) head;
        BGST correction = tanh(beta)*raw_BG [B,12,N,2], transposed to BNT2.
  Y = Y_base + alpha*(R_S+R_D) + tanh(beta)*raw_BG.
  alpha is an unconstrained learned scalar; beta learned scalar starts at0.
  Gate last layer starts at0 -> initial per-expert weights0.5, then learned.
  Inflow/outflow are inverse-scaled and evaluated from the same joint output.

Activation checkpointing (native stages, experts, BGST node blocks) and the
node chunk_size=16 are execution/memory devices, not additional model branches.
Expert evidence uses observed history only. Future target labels never enter
model forward. Raw evidence and normalized branch inputs are distinct inputs.
Training batch64 is an execution setting, not a fixed architecture dimension B.

Usage: import this file and instantiate JointFull with frozen graph inputs.
PyTorch/NumPy/PyWavelets are required. No project-specific import is required.
"""
from __future__ import annotations

import math
import hashlib
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pywt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint as torch_checkpoint

MAX_TOP_K = 8


# SOURCE: teddcf/layers.py:9
class AttentionLayer(nn.Module):
    """Multi-head scaled dot-product attention used by TEDDCF."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 8,
        mask: bool = False,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        target_length = query.shape[-2]
        source_length = key.shape[-2]
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)
        query = torch.cat(
            torch.split(query, self.head_dim, dim=-1),
            dim=0,
        )
        key = torch.cat(
            torch.split(key, self.head_dim, dim=-1),
            dim=0,
        ).transpose(-1, -2)
        value = torch.cat(
            torch.split(value, self.head_dim, dim=-1),
            dim=0,
        )
        score = (query @ key) / (self.head_dim**0.5)
        if self.mask:
            mask = torch.ones(
                target_length,
                source_length,
                dtype=torch.bool,
                device=query.device,
            ).tril()
            score.masked_fill_(~mask, -torch.inf)
        output = torch.softmax(score, dim=-1) @ value
        output = torch.cat(
            torch.split(output, batch_size, dim=0),
            dim=-1,
        )
        return self.out_proj(output)


# SOURCE: teddcf/layers.py:71
class SelfAttentionLayer(nn.Module):
    """Residual self-attention and feed-forward block used by TEDDCF."""

    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        mask: bool = False,
    ):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        dim: int = -2,
    ) -> torch.Tensor:
        x = x.transpose(dim, -2)
        residual = x
        output = self.dropout1(self.attn(x, x, x))
        output = self.ln1(residual + output)
        residual = output
        output = self.dropout2(self.feed_forward(output))
        output = self.ln2(residual + output)
        return output.transpose(dim, -2)


# SOURCE: baselines/TEDDCF-main/TEDDCF-main/TEDDCF/model.py:10
class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):#PEMS08: 192
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):


        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out#[64,192,170,12]


# SOURCE: baselines/TEDDCF-main/TEDDCF-main/TEDDCF/model.py:29
class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()
        #S08:time 288 features 96
        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))#[288 96]
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))#[7 96]
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        #x #in:[64,12,170,3]
        day_emb = x[..., 1]

        time_day = self.time_day[(day_emb[:, :, :] * self.time).type(torch.LongTensor)]


        time_day = time_day.transpose(1, 2).contiguous()

        week_emb = x[..., 2]


        time_week = self.time_week[(week_emb[:, :, :]).type(torch.LongTensor)]#[64,12,170,96]
        time_week = time_week.transpose(1, 2).contiguous()#torch.Size([64, 170, 12, 96])


        tem_emb = time_day + time_week#[64,170,12,96]

        tem_emb = tem_emb.permute(0,3,1,2)#[64,96,170,12]

        return tem_emb


# SOURCE: baselines/TEDDCF-main/TEDDCF-main/TEDDCF/model.py:64
class Diffusion_GCN(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.diffusion_step = diffusion_step#1
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))#[192,192,(1,1)]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):

        out = []
        for i in range(0, self.diffusion_step):#1
            if adj.dim() == 3:
                x = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
                out.append(x)
            elif adj.dim() == 2:
                x = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
                out.append(x)
        x = torch.cat(out, dim=1)
        x = self.conv(x)
        output = self.dropout(x)
        return output


# SOURCE: baselines/TEDDCF-main/TEDDCF-main/TEDDCF/model.py:87
class EventGraph_Fusion(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2,1)
        
    def forward(self, x):
        adj_dyn_1 = torch.softmax(
            F.relu(
                torch.einsum("bcnt, cm->bnm", x, self.memory).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        adj_f = torch.cat([(adj_dyn_1).unsqueeze(-1)] + [(adj_dyn_2).unsqueeze(-1)], dim=-1)

        adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)

        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1]*0.8), dim=-1)

        mask = torch.zeros_like(adj_f)

        mask.scatter_(-1, topk_indices, 1)

        adj_f = adj_f * mask

        return adj_f


# SOURCE: baselines/TEDDCF-main/TEDDCF-main/TEDDCF/model.py:124
class EventGCN(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()

        self.conv = nn.Conv2d(channels,channels,(1,1))
        self.generator = EventGraph_Fusion(channels, num_nodes, diffusion_step, dropout)
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)
        self.emb = emb

    def forward(self, x):

        skip = x
        x = self.conv(x)
        adj_dyn = self.generator(x)
        x = self.gcn(x, adj_dyn)
        x = x*self.emb + skip

        return x


# SOURCE: baselines/TEDDCF-main/TEDDCF-main/TEDDCF/model.py:143
class TrendGCN(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()

        self.conv = nn.Conv2d(channels,channels,(1,1))
        self.generator = TrendGraph_Fusion(channels, num_nodes, diffusion_step, dropout)
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)
        self.emb = emb


    def forward(self, x):

        skip = x
        x = self.conv(x)
        adj_dyn = self.generator(x)
        x = self.gcn(x, adj_dyn)
        x = x*self.emb + skip

        return x


# SOURCE: baselines/TEDDCF-main/TEDDCF-main/TEDDCF/model.py:164
class TrendGraph_Fusion(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.memory = nn.Parameter(
            torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2, 1)
        self.E_adaptive = nn.Parameter(torch.randn(num_nodes, 10))

    def forward(self, x):
        # adj_dyn_1 = torch.softmax(
        #     F.relu(
        #         torch.einsum("bcnt, cm->bnm", x, self.memory).contiguous()
        #         / math.sqrt(x.shape[1])
        #     ),
        #     -1,
        # )

        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        adj_adp = F.softmax(F.relu(torch.mm(self.E_adaptive, self.E_adaptive.transpose(0, 1))), dim=1)

        adj_adp_expanded = adj_adp.unsqueeze(0)

        adj_adp = adj_adp_expanded.repeat(x.shape[0], 1, 1)

        adj_f = torch.cat([(adj_dyn_2).unsqueeze(-1)] + [(adj_adp).unsqueeze(-1)], dim=-1)

        adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)

        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1] * 0.8), dim=-1)

        mask = torch.zeros_like(adj_f)

        mask.scatter_(-1, topk_indices, 1)

        adj_f = adj_f * mask

        return adj_f


# SOURCE: baselines/TEDDCF-main/TEDDCF-main/TEDDCF/model.py:209
class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


# SOURCE: baselines/TEDDCF-main/TEDDCF-main/TEDDCF/model.py:220
class TemporalConvNet(nn.Module):
    def __init__(self, features, kernel_size=2, dropout=0.2, levels=1):
        super(TemporalConvNet, self).__init__()

        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(features, features, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, xh):
        xh = self.tcn(xh)
        return xh
    pass


# SOURCE: baselines/TEDDCF-main/TEDDCF-main/TEDDCF/model.py:242
class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()



        self.res_ln = res_ln
        self.L = len(fea) - 1#2
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i+1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):

        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L-1:
                x = F.relu(x)


        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x


# SOURCE: baselines/TEDDCF-main/TEDDCF-main/TEDDCF/model.py:267
class Adaptive_Fusion(nn.Module):
    def __init__(self, heads, dims):
        super(Adaptive_Fusion, self).__init__()
        features = dims  # 192
        self.h = heads  # 8
        self.d = int(dims / heads) # 16

        self.qlfc = FeedForward([features, features])
        self.khfc = FeedForward([features, features])
        self.vhfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features, features, features], True)

    def forward(self, xl, xh, Mask=True):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        '''
        # xl += te
        # xh += te

        query = self.qlfc(xl)  # [B,T,N,F]
        keyh = torch.relu(self.khfc(xh))  # [B,T,N,F]
        valueh = torch.relu(self.vhfc(xh))  # [B,T,N,F]

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]
        keyh = torch.cat(torch.split(keyh, self.d, -1), 0).permute(0, 2, 3, 1)  # [k*B,N,d,T]
        valueh = torch.cat(torch.split(valueh, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]

        attentionh = torch.matmul(query, keyh)  # [k*B,N,T,T]

        if Mask:
            batch_size = xl.shape[0]
            num_steps = xl.shape[1]
            num_vertexs = xl.shape[2]
            mask = torch.ones(num_steps, num_steps).to(xl.device)  # [T,T]
            mask = torch.tril(mask)  # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1)  # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1) * torch.ones_like(attentionh).to(xl.device)  # [k*B,N,T,T]
            attentionh = torch.where(mask, attentionh, zero_vec)

        attentionh /= (self.d ** 0.5)  # scaled
        attentionh = F.softmax(attentionh, -1)  # [k*B,N,T,T]

        value = torch.matmul(attentionh, valueh)  # [k*B,N,T,d]

        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1).permute(0, 2, 1, 3)  # [B,T,N,F]
        value = self.ofc(value)
        value = value + xl

        value = self.ln(value)

        return self.ff(value)  # [64,12,170,128]


# SOURCE: baselines/TEDDCF-main/TEDDCF-main/TEDDCF/model.py:327
class TEDDCF(nn.Module):
    def __init__(
        self, device, input_dim, num_nodes, channels, granularity, dropout=0.1
    ):
        super().__init__()

        self.device = device
        self.num_nodes = num_nodes
        self.output_len = 12
        self.input_len = 12
        self.heads = 8
        diffusion_step = 1

        self.Temb = TemporalEmbedding(granularity, channels)

        self.start_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=channels, kernel_size=(1, 1)
        )


        self.glu = GLU(channels*2, dropout)

        self.regression_layer = nn.Conv2d(
            channels*2, self.output_len, kernel_size=(1, self.output_len)
        )

        self.temporal_conv = TemporalConvNet(channels*2)
        self.pre_h = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1,1))
        self.adp_f = Adaptive_Fusion(self.heads, channels*2)

        num_layers = 3
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(channels*2, feed_forward_dim=256, num_heads=4, dropout=0.1)
                for _ in range(num_layers)  # 3
            ]
        )
        self.xh_emb = nn.Parameter(torch.randn(channels*2, num_nodes, 12))
        self.xh_dgcn = EventGCN(channels*2, num_nodes, diffusion_step=1, dropout=0.1,emb=self.xh_emb)

        self.xl_emb = nn.Parameter(torch.randn(channels*2, num_nodes, 12))
        self.xl_dgcn = TrendGCN(channels*2, num_nodes, diffusion_step=1, dropout=0.1, emb=self.xl_emb)


    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, inputxl, inputxh):

        xl = inputxl
        xh = inputxh

        # Encoder
        # Data Embedding
        time_embl = self.Temb(inputxl.permute(0, 3, 2, 1))
        time_embh = self.Temb(inputxh.permute(0, 3, 2, 1))
        #t = self.start_conv(x)#[64,96,170,12]
        xl = torch.cat([self.start_conv(xl)] + [time_embl], dim=1)
        xh = torch.cat([self.start_conv(xh)] + [time_embh], dim=1)



        xl = xl.permute(0, 3, 2, 1)
        for attn in self.attn_layers_t:
            xl = attn(xl, dim=1)
        xl = xl.permute(0, 3, 2, 1)

        xl = self.xl_dgcn(xl)
        xl = self.glu(xl) + xl


        xh = self.temporal_conv(xh)


        xh = self.xh_dgcn(xh)

        #simple plus
        x_all = xh + xl
        #STwave_fusion
        # xl = xl.transpose(1, 3)
        # xh = self.pre_h(xh.transpose(1,3))#[64,12,170,192]
        # x_all = self.adp_f(xl, xh)
        # x_all = x_all.transpose(1, 3)

        prediction = self.regression_layer(F.relu(x_all))


        return prediction


UpstreamTEDDCF = TEDDCF

# SOURCE: teddcf/v2/model.py:9
class DeviceSafeTemporalEmbedding(nn.Module):
    def __init__(
        self,
        *,
        granularity: int,
        time_day: nn.Parameter,
        time_week: nn.Parameter,
    ) -> None:
        super().__init__()
        self.granularity = int(granularity)
        self.time_day = time_day
        self.time_week = time_week

    @classmethod
    def from_upstream(cls, module: nn.Module) -> "DeviceSafeTemporalEmbedding":
        return cls(
            granularity=int(module.time),
            time_day=module.time_day,
            time_week=module.time_week,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        day_index = (x[..., 1] * self.granularity).to(
            device=x.device,
            dtype=torch.long,
        )
        week_index = x[..., 2].to(device=x.device, dtype=torch.long)
        if torch.any(day_index < 0) or torch.any(day_index >= self.granularity):
            raise ValueError("time-of-day index is outside configured granularity")
        if torch.any(week_index < 0) or torch.any(week_index > 6):
            raise ValueError("weekday index must be in [0,6]")
        time_day = self.time_day[day_index].transpose(1, 2).contiguous()
        time_week = self.time_week[week_index].transpose(1, 2).contiguous()
        return (time_day + time_week).permute(0, 3, 1, 2)


# SOURCE: teddcf/v2/model.py:45
class NativeCFModel(UpstreamTEDDCF):
    """Upstream 12->12 TEDDCF with only device-safe tensor adaptation."""

    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        num_nodes: int,
        channels: int,
        granularity: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            device=device,
            input_dim=input_dim,
            num_nodes=num_nodes,
            channels=channels,
            granularity=granularity,
            dropout=dropout,
        )
        self.Temb = DeviceSafeTemporalEmbedding.from_upstream(self.Temb)

    def forward(self, inputxl: torch.Tensor, inputxh: torch.Tensor) -> torch.Tensor:
        expected = (3, self.num_nodes, 12)
        if inputxl.ndim != 4 or tuple(inputxl.shape[1:]) != expected:
            raise ValueError(f"inputxl must have shape [B,{expected[0]},{expected[1]},12]")
        if inputxh.shape != inputxl.shape:
            raise ValueError("inputxh must match inputxl")
        return super().forward(inputxl, inputxh)


# SOURCE: teddcf/scratch_t5.py:36
class MemorySafeNative(NativeCFModel):
    """Same modules/order as upstream; recompute stages during backpropagation."""
    def stage(self, module, value):
        if self.training and torch.is_grad_enabled():
            return checkpoint(module,value,use_reentrant=False,preserve_rng_state=True)
        return module(value)

    def forward(self,low,high):
        expected=(3,self.num_nodes,12)
        if tuple(low.shape[1:])!=expected or low.shape!=high.shape:
            raise ValueError('native 12-to-12 input shape mismatch')
        tl=self.Temb(low.permute(0,3,2,1))
        th=self.Temb(high.permute(0,3,2,1))
        low=torch.cat([self.start_conv(low),tl],dim=1)
        high=torch.cat([self.start_conv(high),th],dim=1)
        low=low.permute(0,3,2,1)
        for attn in self.attn_layers_t:
            low=self.stage(lambda value, layer=attn: layer(value,dim=1),low)
        low=low.permute(0,3,2,1)
        low=self.stage(self.xl_dgcn,low)
        low=self.stage(self.glu,low)+low
        high=self.stage(self.temporal_conv,high)
        high=self.stage(self.xh_dgcn,high)
        return self.regression_layer(torch.relu(high+low))


# SOURCE: teddcf/scratch_t5.py:108
def indices(length):
    origins = np.arange(11, length - 12)
    h, y = origins[:, None] + np.arange(-11, 1), origins[:, None] + np.arange(1, 13)
    n = len(origins)
    train, test = round(n * .6), round(n * .2)
    return h, y, {'train': np.arange(train), 'val': np.arange(train, n-test),
                  'test': np.arange(n-test, n)}


# SOURCE: teddcf/scratch_t5.py:117
def pad_order(order, batch=64):
    return np.concatenate([order, np.repeat(order[-1], (-len(order)) % batch)])


# SOURCE: teddcf/v2/s0/graphs.py:14
@dataclass(frozen=True)
class SparseGraph:
    graph_id: str
    edges: np.ndarray
    weights: np.ndarray
    lags: np.ndarray
    source_role: str


# SOURCE: teddcf/v2/s0/graphs.py:23
def _top_k(top_k: int) -> int:
    value = int(top_k)
    if value < 1 or value > MAX_TOP_K:
        raise ValueError(f"top_k must be in [1,{MAX_TOP_K}]")
    return value


# SOURCE: teddcf/v2/s0/graphs.py:30
def _empty_lags(size: int) -> np.ndarray:
    return np.zeros((size,), dtype=np.int64)


# SOURCE: teddcf/v2/s0/graphs.py:34
def _graph(graph_id: str, edges: Iterable[tuple[int, int]], weights: Iterable[float], source_role: str, lags=None) -> SparseGraph:
    edge_array = np.asarray(list(edges), dtype=np.int64)
    if edge_array.size == 0:
        edge_array = np.empty((0, 2), dtype=np.int64)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("sparse graph edges must be two-dimensional")
    weight_array = np.asarray(list(weights), dtype=np.float32)
    lag_array = _empty_lags(edge_array.shape[0]) if lags is None else np.asarray(list(lags), dtype=np.int64)
    if weight_array.shape != (edge_array.shape[0],) or lag_array.shape != (edge_array.shape[0],):
        raise ValueError("sparse graph edge fields have incompatible shapes")
    if not np.isfinite(weight_array).all():
        raise ValueError("sparse graph weights must be finite")
    return SparseGraph(graph_id, edge_array, weight_array, lag_array, source_role)


# SOURCE: teddcf/v2/s0/graphs.py:49
def build_identity_graph(*, nodes: int, top_k: int = 1) -> SparseGraph:
    if _top_k(top_k) != 1:
        raise ValueError("identity graph top_k must be exactly one self edge per node")
    if int(nodes) <= 0:
        raise ValueError("nodes must be positive")
    edges = [(node, node) for node in range(int(nodes))]
    return _graph("self_identity", edges, [1.0] * len(edges), "structural")


# SOURCE: teddcf/v2/s0/experts.py:16
@dataclass(frozen=True)
class S0StateEvidence:
    climate_baseline: torch.Tensor
    climate_deviation: torch.Tensor
    recent_residual: torch.Tensor
    trend_deviation: torch.Tensor
    high_frequency_change: torch.Tensor
    node_history: torch.Tensor
    periodic_time: torch.Tensor

    def replace(self, **changes) -> "S0StateEvidence":
        return replace(self, **changes)

    def select(self, item: slice | int) -> "S0StateEvidence":
        return S0StateEvidence(*(
            getattr(self, field)[item] for field in self.__dataclass_fields__
        ))


# SOURCE: teddcf/v2/s0/experts.py:35
def _resize_series(value: torch.Tensor, steps: int) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError("S0 node evidence must be [B,N,T]")
    if value.shape[-1] == steps:
        return value
    return F.interpolate(value, size=steps, mode="linear", align_corners=False)


# SOURCE: teddcf/v2/s0/experts.py:43
def _coerce_evidence(
    evidence: S0StateEvidence | torch.Tensor,
    *,
    z: torch.Tensor,
    horizon: int,
) -> S0StateEvidence:
    if isinstance(evidence, S0StateEvidence):
        fields = (
            evidence.climate_baseline,
            evidence.climate_deviation,
            evidence.recent_residual,
            evidence.trend_deviation,
            evidence.high_frequency_change,
            evidence.node_history,
        )
        if any(value.shape[:2] != (z.shape[0], z.shape[2]) for value in fields):
            raise ValueError("S0 evidence batch/node shape differs")
        if evidence.periodic_time.ndim != 3 or evidence.periodic_time.shape[:2] != (z.shape[0], z.shape[-1]):
            raise ValueError("S0 periodic evidence must be [B,T,4]")
        if evidence.periodic_time.shape[-1] != 4:
            raise ValueError("S0 periodic evidence requires four phase channels")
        return S0StateEvidence(
            *(_resize_series(value, int(z.shape[-1])) for value in fields),
            periodic_time=evidence.periodic_time,
        )
    if evidence.ndim != 4 or evidence.shape[0] != z.shape[0] or evidence.shape[1] != horizon or evidence.shape[2] != z.shape[2]:
        raise ValueError("expert evidence must be [B,H,N,E]")
    series = evidence[..., 0].transpose(1, 2).contiguous()
    series = _resize_series(series, int(z.shape[-1]))
    return S0StateEvidence(
        climate_baseline=series,
        climate_deviation=series,
        recent_residual=series,
        trend_deviation=series,
        high_frequency_change=series,
        node_history=series,
        periodic_time=z.new_zeros((z.shape[0], z.shape[-1], 4)),
    )


# SOURCE: teddcf/v2/s0/experts.py:83
def _validate_z(z: torch.Tensor, channels: int) -> None:
    if z.ndim != 4 or int(z.shape[1]) != int(channels):
        raise ValueError("expert input must be [B,C,N,T]")


# SOURCE: teddcf/v2/s0/experts.py:88
def _validate_graph(graph: SparseGraph, allowed: tuple[str, ...], context: str) -> None:
    if graph.graph_id not in allowed:
        raise ValueError(f"{context} expert requires a {context} graph")


# SOURCE: teddcf/v2/s0/experts.py:104
def _forecast(features: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
    return projection(features).permute(0, 2, 1).unsqueeze(-1)


# SOURCE: teddcf/v2/s0/experts.py:108
def _checkpoint_expert(module: nn.Module, function, *args: torch.Tensor) -> torch.Tensor:
    if module.training and torch.is_grad_enabled():
        return torch_checkpoint(
            function,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
        )
    return function(*args)


# SOURCE: teddcf/v2/s0/experts.py:193
class DelayExpert(nn.Module):
    temporal_encoder_kind = "lag_embedding"
    spatial_encoder_kind = "lag_conditioned_directed_message_transform"

    def __init__(self, *, channels: int, horizon: int, max_lag_steps: int = 12):
        super().__init__()
        self.channels = int(channels)
        self.horizon = int(horizon)
        self.max_lag_steps = int(max_lag_steps)
        self.lag_embedding = nn.Embedding(self.max_lag_steps + 1, self.channels)
        self.evidence_encoder = nn.Linear(2, self.channels, bias=False)
        self.message_transform = nn.Sequential(
            nn.Linear(self.channels * 2, self.channels * 2, bias=False), nn.SiLU(),
            nn.Linear(self.channels * 2, self.channels, bias=False),
        )
        self.projection = nn.Linear(self.channels, self.horizon)

    def forward(self, z: torch.Tensor, evidence: S0StateEvidence | torch.Tensor, graph: SparseGraph) -> torch.Tensor:
        _validate_z(z, self.channels)
        if graph.graph_id != "directed_multilag":
            raise ValueError("delay expert requires a directed multilag graph")
        state = _coerce_evidence(evidence, z=z, horizon=self.horizon)
        def expert_path(z_value, trend_deviation, high_frequency_change):
            evidence_last = torch.stack((trend_deviation[..., -1], high_frequency_change[..., -1]), dim=-1)
            context = self.evidence_encoder(evidence_last)
            source = torch.as_tensor(graph.edges[:, 0], dtype=torch.long, device=z_value.device)
            target = torch.as_tensor(graph.edges[:, 1], dtype=torch.long, device=z_value.device)
            lags = torch.as_tensor(graph.lags, dtype=torch.long, device=z_value.device)
            if torch.any(lags <= 0) or torch.any(lags >= z_value.shape[-1]) or torch.any(lags > self.max_lag_steps):
                raise ValueError("delay graph lag is outside the input history")
            history = z_value.permute(0, 2, 3, 1).index_select(1, source)
            time_index = (z_value.shape[-1] - lags - 1).view(1, -1, 1, 1)
            time_index = time_index.expand(z_value.shape[0], -1, 1, z_value.shape[1])
            source_value = history.gather(2, time_index).squeeze(2)
            modulation = 1.0 + torch.tanh(self.lag_embedding(lags))
            transformed = self.message_transform(
                torch.cat((source_value * modulation.unsqueeze(0), context.index_select(1, target)), dim=-1)
            )
            weights = torch.as_tensor(graph.weights, dtype=z_value.dtype, device=z_value.device).view(1, -1, 1)
            message = z_value.new_zeros((z_value.shape[0], z_value.shape[2], z_value.shape[1]))
            message.index_add_(1, target, transformed * weights)
            return _forecast(message + context, self.projection)

        return _checkpoint_expert(
            self,
            expert_path,
            z,
            state.trend_deviation,
            state.high_frequency_change,
        )


# SOURCE: teddcf/v2/s0/experts.py:245
class SelfExpert(nn.Module):
    temporal_encoder_kind = "node_independent_gru"
    spatial_encoder_kind = "none_no_cross_node_propagation"

    def __init__(self, *, channels: int, horizon: int):
        super().__init__()
        self.channels = int(channels)
        self.horizon = int(horizon)
        self.gru = nn.GRU(self.channels + 1, self.channels, batch_first=True)
        self.self_transform = nn.Sequential(
            nn.Linear(self.channels, self.channels), nn.Tanh(),
            nn.Linear(self.channels, self.channels),
        )
        self.projection = nn.Linear(self.channels, self.horizon)

    def forward(self, z: torch.Tensor, evidence: S0StateEvidence | torch.Tensor, graph: SparseGraph) -> torch.Tensor:
        _validate_z(z, self.channels)
        _validate_graph(graph, ("self_identity",), "identity")
        if not np.array_equal(graph.edges[:, 0], graph.edges[:, 1]):
            raise ValueError("identity graph may not propagate across nodes")
        state = _coerce_evidence(evidence, z=z, horizon=self.horizon)
        def expert_path(z_value, node_history):
            batch, channels, nodes, steps = z_value.shape
            features = z_value.permute(0, 2, 3, 1)
            history = node_history.unsqueeze(-1)
            sequence = torch.cat((features, history), dim=-1).reshape(batch * nodes, steps, channels + 1)
            _, hidden = self.gru(sequence)
            encoded = self.self_transform(hidden[-1]).reshape(batch, nodes, channels)
            return _forecast(encoded, self.projection)

        return _checkpoint_expert(self, expert_path, z, state.node_history)


# SOURCE: bgst/graph.py:8
@dataclass
class Candidates:
    source_index: np.ndarray
    neighbor_mask: np.ndarray
    edge_origin: np.ndarray
    selection_stage: np.ndarray
    excluded_edges: np.ndarray | None = None

    @property
    def in_degree(self):
        return self.neighbor_mask.sum(axis=1)

    def arrays(self):
        return dict(source_index=self.source_index, neighbor_mask=self.neighbor_mask,
                    edge_origin=self.edge_origin, selection_stage=self.selection_stage,
                    in_degree=self.in_degree,
                    excluded_edges=np.empty((0,3),dtype=np.int64) if self.excluded_edges is None else self.excluded_edges)

    def save(self, path):
        np.savez_compressed(path, **self.arrays())

    def digest(self):
        digest = hashlib.sha256()
        for name, value in self.arrays().items():
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            obj = cls(*(a[k].copy() for k in ('source_index', 'neighbor_mask', 'edge_origin', 'selection_stage')))
            if 'excluded_edges' in a: obj.excluded_edges=a['excluded_edges'].copy()
            if not np.array_equal(obj.in_degree, a['in_degree']):
                raise ValueError('candidate degree mismatch')
        obj.validate()
        return obj

    def validate(self):
        ix, mask = self.source_index, self.neighbor_mask
        if ix.ndim != 2 or ix.shape[1] != 8 or mask.shape != ix.shape:
            raise ValueError('candidates must be [N,8]')
        if ix.dtype != np.int64 or mask.dtype != np.bool_:
            raise ValueError('candidate dtype differs')
        if np.any(ix < 0) or np.any(ix >= len(ix)):
            raise ValueError('unsafe source index')
        for i in range(len(ix)):
            selected = ix[i, mask[i]]
            if i in selected or len(set(selected.tolist())) != len(selected):
                raise ValueError('self or duplicate source')
        for meta in (self.edge_origin, self.selection_stage):
            if meta.shape != ix.shape:
                raise ValueError('metadata shape differs')


# SOURCE: bgst/model.py:10
class TemporalPatterns(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.projection = nn.Conv2d(channels, dim, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(dim, dim, 3, dilation=d, groups=dim),
                          nn.Conv1d(dim, dim, 1), nn.GELU()) for d in (1, 2, 4)])

    def forward(self, high):
        x = self.projection(high)
        b, d, n, t = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b*n, d, t)
        patterns = [branch(F.pad(x, (2*dilation, 0))).reshape(b, n, d, t).transpose(-1, -2)
                    for dilation, branch in zip((1, 2, 4), self.branches)]
        return torch.stack(patterns, dim=2)  # B,N,S,T,D


# SOURCE: bgst/model.py:27
class BGST(nn.Module):
    def __init__(self, *, channels, candidates: Candidates, dim=32, interval_minutes=5,
                 chunk_size=16, sync=True):
        super().__init__()
        candidates.validate()
        if dim != 32 or interval_minutes != 5 or chunk_size <= 0:
            raise ValueError('frozen T5 BGST configuration differs')
        self.dim, self.interval, self.chunk_size = dim, interval_minutes, chunk_size
        self.sync = bool(sync)
        self.background = nn.Sequential(nn.Linear(channels, dim), nn.LayerNorm(dim))
        self.temporal = TemporalPatterns(channels, dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.film_q = nn.Linear(dim, 2*dim)
        self.film_k = nn.Linear(dim, 2*dim)
        for film in (self.film_q, self.film_k):
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)
        self.score_q = nn.Linear(dim, dim, bias=False)
        self.score_k = nn.Linear(dim, dim, bias=False)
        self.score_product = nn.Linear(dim, dim, bias=False)
        self.score_difference = nn.Linear(dim, dim, bias=False)
        self.lag_embed = nn.Linear(3, dim, bias=False)
        self.horizon_embed = nn.Linear(3, dim, bias=False)
        self.scale_embed = nn.Linear(3, dim, bias=False)
        self.score_bias = nn.Parameter(torch.zeros(dim))
        self.score_out = nn.Linear(dim, 1, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.head = nn.Linear(dim, 1, bias=False)
        self.beta = nn.Parameter(torch.zeros(()))
        self.register_buffer('lag_steps',torch.arange(4))
        self.register_buffer('is_sync',torch.tensor([True,False,False,False]))
        self.register_buffer('source_index', torch.from_numpy(candidates.source_index.copy()))
        self.register_buffer('neighbor_mask', torch.from_numpy(candidates.neighbor_mask.copy()))

    @staticmethod
    def _condition(projection, film, pattern, context):
        gamma, delta = film(context).chunk(2, dim=-1)
        return (1+gamma)*projection(pattern)+delta

    def _minutes(self, values, reference):
        x = reference.new_tensor(values) * self.interval / 60.
        return torch.stack((x, torch.sin(2*math.pi*x), torch.cos(2*math.pi*x)), dim=-1)

    def _block(self, pattern, context, sources, mask, target_indices, diagnostics=False):
        b, _, scales, steps, dim = pattern.shape
        m, neighbors = sources.shape
        lags = self.lag_steps
        target = pattern[:, target_indices, :, -1, :]
        source = pattern[:, sources, :, :, :].index_select(-2, steps-1-lags)
        # source: B,M,K,S,L,D; query: B,M,1,S,1,D
        q = self._condition(self.q, self.film_q, target,
                            context[:, target_indices, None, :])[:, :, None, :, None, :]
        k = self._condition(self.k, self.film_k, source, context[:, sources, None, None, :])
        base = (self.score_q(q)+self.score_k(k)+self.score_product(q*k)
                + self.score_difference(q-k)+self.score_bias)
        lag_e = self.lag_embed(self._minutes([0, 1, 2, 3], pattern))[None, None, None, None, :, :]
        scale_e = self.scale_embed(self._minutes([3, 5, 9], pattern))[None, None, None, :, None, :]
        base = base+lag_e+scale_e
        h_e = self.horizon_embed(self._minutes(list(range(1, 13)), pattern))
        valid = mask[:, :, None, None] & (pattern.new_tensor([3,5,9])[:, None]+lags <= steps)[None, None]
        if not self.sync:
            valid = valid & (~self.is_sync)[None, None, None, :]
        valid = valid.reshape(1, m, -1)
        values = self.value(source).reshape(b, m, -1, dim)
        outputs, all_weights = [], []
        # Reuse values across horizons; avoid B*M*K*S*L*H*D allocation.
        for h in range(12):
            score = self.score_out(F.gelu(base+h_e[h])).reshape(b, m, -1)
            score = score.masked_fill(~valid, -torch.inf)
            weights = torch.softmax(torch.cat((score, score.new_zeros(b, m, 1)), -1), -1)
            message = torch.sum(weights[..., :-1, None]*values, dim=-2)
            outputs.append(self.head(message).squeeze(-1))
            if diagnostics:
                all_weights.append(weights)
        result = torch.stack(outputs, dim=1).unsqueeze(-1)  # B,H,M,1
        if diagnostics:
            return result, torch.stack(all_weights, dim=1)
        return result

    def forward(self, low, high, *, return_weights=False):
        if low.shape != high.shape or low.ndim != 4 or low.shape[-1] != 12:
            raise ValueError('native features must be matching [B,C,N,12]')
        if low.shape[2] != self.source_index.shape[0]:
            raise ValueError('node count differs from frozen candidates')
        context = self.background(low.mean(dim=-1).transpose(1, 2))
        pattern = self.temporal(high)
        blocks, weights = [], []
        for start in range(0, low.shape[2], self.chunk_size):
            stop = min(start+self.chunk_size, low.shape[2])
            target = torch.arange(start, stop, device=low.device)
            args = (pattern, context, self.source_index[start:stop], self.neighbor_mask[start:stop], target)
            if return_weights:
                result, w = self._block(*args, diagnostics=True)
                weights.append(w.detach())
            elif self.training and torch.is_grad_enabled():
                result = checkpoint(self._block, *args, use_reentrant=False, preserve_rng_state=True)
            else:
                result = self._block(*args)
            blocks.append(result)
        raw = torch.cat(blocks, dim=2)
        delta = torch.tanh(self.beta)*raw
        return (delta, torch.cat(weights, dim=2)) if return_weights else delta


# SOURCE: ablation_experiments/joint_full_v3/scripts/model.py:10
class JointNative(MemorySafeNative):
    def __init__(self, device, nodes, channels=96):
        super().__init__(device=device, input_dim=4, num_nodes=nodes,
                         channels=channels, granularity=288, dropout=.1)
        self.regression_layer = nn.Conv2d(channels*2, 24, (1, 12))

    def forward(self, low, high, temporal):
        # Public flow tensors B,N,T,2; auxiliary clock B,N,T,2.
        if low.shape != high.shape or low.shape[1:] != (self.num_nodes,12,2):
            raise ValueError('Expected matching B,N,12,2 flow tensors')
        def embed(x):
            value = torch.cat((x, temporal), -1).permute(0,3,1,2)
            # Existing embedding expects clock at positions 1,2, not 2,3.
            clock = torch.cat((x[..., :1], temporal), -1).permute(0,2,1,3)
            return torch.cat((self.start_conv(value), self.Temb(clock)), 1)
        low, high = embed(low), embed(high)
        low = low.permute(0,3,2,1)
        for attn in self.attn_layers_t:
            low = self.stage(lambda x, layer=attn: layer(x, dim=1), low)
        low = self.stage(self.xl_dgcn, low.permute(0,3,2,1))
        low_features = low
        low = self.stage(self.glu, low) + low
        high = self.stage(self.temporal_conv, high)
        high = self.stage(self.xh_dgcn, high)
        out = self.regression_layer(torch.relu(high+low)).squeeze(-1)
        out = out.reshape(out.shape[0],12,2,self.num_nodes).permute(0,3,1,2)
        return out, low_features, high


# SOURCE: ablation_experiments/joint_full_v3/scripts/model.py:39
class JointBGST(BGST):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = nn.Linear(self.dim, 2, bias=False)

    def _block(self, *args, diagnostics=False):
        result = super()._block(*args, diagnostics=diagnostics)
        # Parent appends a singleton for its single-output head.
        if diagnostics:
            return result[0].squeeze(-1), result[1]
        return result.squeeze(-1)


# SOURCE: ablation_experiments/joint_full_v3/scripts/model.py:52
class JointFull(nn.Module):
    def __init__(self, *, device, nodes, delay_graphs, candidates, channels=96):
        super().__init__()
        self.base_model = JointNative(device, nodes, channels)
        width = channels*2
        self.self_expert = SelfExpert(channels=width, horizon=12)
        self.delay_expert = DelayExpert(channels=width, horizon=12)
        self.identity_graph = build_identity_graph(nodes=nodes)
        self.delay_graphs = delay_graphs
        # Channel-specific evidence conditions shared expert weights.
        self.gate = nn.Sequential(nn.Linear(width*2+4,32), nn.SiLU(), nn.Linear(32,4))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.bgst = JointBGST(channels=width, candidates=candidates, chunk_size=16, sync=True)

    def forward(self, low, high, evidence):
        states, temporal = evidence
        base, zl, zh = self.base_model(low, high, temporal)
        selfs, delays = [], []
        for c, state in enumerate(states):
            selfs.append(self.self_expert(zl+zh, state, self.identity_graph))
            delays.append(torch.cat([self.delay_expert(zh[i:i+1], state.select(slice(i,i+1)),
                self.delay_graphs[c]) for i in range(len(zh))], 0))
        s = torch.cat(selfs,-1).transpose(1,2)
        d = torch.cat(delays,-1).transpose(1,2)
        features = [zl.mean(-1).transpose(1,2), zh.mean(-1).transpose(1,2)]
        features += [torch.stack((e.trend_deviation[..., -1],e.high_frequency_change[..., -1]),-1) for e in states]
        weights = self.gate(torch.cat(features,-1)).reshape(len(low),low.shape[1],2,2).softmax(-1)
        self.last_weights = weights.detach()
        ws, wd = weights[...,0].unsqueeze(2), weights[...,1].unsqueeze(2)
        rs, rd = ws*(s-base), wd*d
        self.last_diagnostics = dict(alpha=float(self.alpha.detach()),
            self_weight=weights[...,0].detach().mean((0,1)).cpu().tolist(),
            delay_weight=weights[...,1].detach().mean((0,1)).cpu().tolist(),
            weighted_self_abs=rs.detach().abs().mean((0,1,2)).cpu().tolist(),
            weighted_delay_abs=rd.detach().abs().mean((0,1,2)).cpu().tolist())
        return base + self.alpha*(rs+rd) + self.bgst(zl,zh).transpose(1,2)


# SOURCE: teddcf/v2/s0/state_evidence.py:62
class S0StateEvidenceProvider:
    """Build per-batch state evidence from train-fit climate and history only."""

    def __init__(
        self,
        *,
        data: np.ndarray,
        climate_baseline: np.ndarray,
        granularity: int,
        channel_index: int,
    ) -> None:
        values = np.asarray(data)
        climate = np.asarray(climate_baseline, dtype=np.float32)
        if values.ndim != 3 or channel_index < 0 or channel_index >= values.shape[2]:
            raise ValueError("S0 evidence timeline/channel shape differs")
        if climate.shape != (2, int(granularity), values.shape[1]):
            raise ValueError("S0 climate baseline shape differs")
        self.data = values
        self.climate = climate
        self.granularity = int(granularity)
        self.channel_index = int(channel_index)

    def build(self, history_indices: torch.Tensor, *, device: torch.device) -> S0StateEvidence:
        indices = torch.as_tensor(history_indices, dtype=torch.int64).detach().cpu().numpy()
        if indices.ndim != 2 or indices.shape[1] < 2:
            raise ValueError("S0 history indices must be [B,T] with T>=2")
        if indices.min() < 0 or indices.max() >= self.data.shape[0]:
            raise ValueError("S0 history index is outside the timeline")
        history_btn = np.asarray(
            self.data[indices, :, self.channel_index], dtype=np.float32
        )
        weekday = (1 + indices // self.granularity) % 7
        weekday_type = (weekday >= 5).astype(np.int64)
        slot = indices % self.granularity
        climate_btn = self.climate[weekday_type, slot]
        history = np.transpose(history_btn, (0, 2, 1)).copy()
        climate = np.transpose(climate_btn, (0, 2, 1)).copy()
        deviation = history - climate
        recent = np.zeros_like(history)
        recent[..., 1:] = history[..., 1:] - history[..., :-1]
        rolling = np.empty_like(history)
        for step in range(history.shape[-1]):
            rolling[..., step] = history[..., max(0, step - 2):step + 1].mean(axis=-1)
        trend = rolling - climate
        high_frequency = history - rolling
        phase_day = slot.astype(np.float32) / np.float32(self.granularity)
        phase_week = weekday.astype(np.float32) / np.float32(7.0)
        periodic = np.stack((
            np.sin(2.0 * np.pi * phase_day), np.cos(2.0 * np.pi * phase_day),
            np.sin(2.0 * np.pi * phase_week), np.cos(2.0 * np.pi * phase_week),
        ), axis=-1).astype(np.float32)

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(value)).to(device=device)

        return S0StateEvidence(
            climate_baseline=tensor(climate),
            climate_deviation=tensor(deviation),
            recent_residual=tensor(recent),
            trend_deviation=tensor(trend),
            high_frequency_change=tensor(high_frequency),
            node_history=tensor(history),
            periodic_time=tensor(periodic),
        )


# SOURCE: ablation_experiments/joint_full_v3/scripts/data.py:7
class ChannelScaler:
    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean), np.asarray(std)
    def transform(self, x):
        if torch.is_tensor(x):
            return (x-x.new_tensor(self.mean))/x.new_tensor(self.std)
        return (x-self.mean)/self.std
    def inverse_transform(self, x):
        if torch.is_tensor(x):
            return x*x.new_tensor(self.std)+x.new_tensor(self.mean)
        return x*self.std+self.mean


# SOURCE: ablation_experiments/joint_full_v3/scripts/data.py:20
class JointData:
    def __init__(self, data, temporal):
        if data.ndim != 3 or data.shape[-1] != 2:
            raise ValueError('Expected raw T,N,2: inflow then outflow')
        self.data, self.temporal = data, temporal
        self.h,self.y,self.rows = indices(len(data))
        train = np.asarray(data[self.h[self.rows['train']]])
        self.mean,self.std = train.mean((0,1,2)),train.std((0,1,2))
        if not np.isfinite(self.std).all() or (self.std<=0).any():
            raise ValueError('Invalid training-only scaler')
        self.scaler = ChannelScaler(self.mean,self.std)
        self.order = {}
        for split, rows in self.rows.items():
            order = rows.copy()
            if split in ('train','val'):
                order = order[torch.randperm(len(order)).numpy()]
            self.order[split] = pad_order(order)

    def batch(self, rows, device):
        history = self.h[rows]
        raw = np.asarray(self.data[history]).transpose(0,2,1,3) # BNTC
        # Decompose time, never mix flow channels.
        coef = pywt.wavedec(raw, 'coif1', level=1, axis=2)
        low = pywt.waverec([coef[0],None], 'coif1', axis=2)
        high = pywt.waverec([None,coef[1]], 'coif1', axis=2)
        target = np.asarray(self.data[self.y[rows]]).transpose(0,2,1,3)
        return (torch.as_tensor(self.scaler.transform(low),device=device),
                torch.as_tensor(self.scaler.transform(high),device=device),
                torch.as_tensor(target,device=device),torch.from_numpy(history))

    def batches(self, split, device):
        for i in range(0,len(self.order[split]),64):
            yield self.batch(self.order[split][i:i+64],device)


# SOURCE: ablation_experiments/joint_full_v3/scripts/data.py:55
class JointProvider:
    def __init__(self, providers, temporal, nodes):
        self.providers,self.temporal,self.nodes = providers,temporal,nodes
    def build(self, history_indices, device):
        ix = history_indices.detach().cpu().numpy()
        clock = torch.as_tensor(self.temporal[ix],device=device,dtype=torch.float32)
        clock = clock[:,None].expand(-1,self.nodes,-1,-1)
        return [p.build(history_indices,device=device) for p in self.providers], clock


# Provenance of original files; definitions above are saved, not read at import.
SOURCE_MANIFEST = {'teddcf/layers.py': {'sha256': '4c56e386ee60b301f9f39bfa65a618c51d4a7f9e7a6fe18c602ba9e0ee4dc28d', 'definitions': ['AttentionLayer', 'SelfAttentionLayer']}, 'baselines/TEDDCF-main/TEDDCF-main/TEDDCF/model.py': {'sha256': '643cda738b737cddb7079001346317e9534c4259d1ce80ee52bef0d0c17dd64f', 'definitions': ['GLU', 'TemporalEmbedding', 'Diffusion_GCN', 'EventGraph_Fusion', 'EventGCN', 'TrendGCN', 'TrendGraph_Fusion', 'Chomp1d', 'TemporalConvNet', 'FeedForward', 'Adaptive_Fusion', 'TEDDCF']}, 'teddcf/v2/model.py': {'sha256': '5a65ec09902812f9bee9f96b0e34137da3881f2064f936a681996baee199df6d', 'definitions': ['DeviceSafeTemporalEmbedding', 'NativeCFModel']}, 'teddcf/scratch_t5.py': {'sha256': '59a1152aceed8d86fecfd0570354c5c9f15b4fee80489318fc5c1f13e565b7d5', 'definitions': ['MemorySafeNative', 'indices', 'pad_order']}, 'teddcf/v2/s0/graphs.py': {'sha256': '95d24d2374ac5577eba1948967795fd7d7c08cd17219ca316a9ce5206d6f9a5a', 'definitions': ['SparseGraph', '_top_k', '_empty_lags', '_graph', 'build_identity_graph']}, 'teddcf/v2/s0/experts.py': {'sha256': '998b6f4b5b5022e20fb49d392023a55a0de04af8d4c2197455b260d2689dfbb3', 'definitions': ['S0StateEvidence', '_resize_series', '_coerce_evidence', '_validate_z', '_validate_graph', '_forecast', '_checkpoint_expert', 'DelayExpert', 'SelfExpert']}, 'bgst/graph.py': {'sha256': '25ce9661cc2613452ae6da3f91a84effef2d70b8a753422a15f763ead8411a89', 'definitions': ['Candidates']}, 'bgst/model.py': {'sha256': '17668408010a83ee1355d85543d434bbb9454ab6b419f35cec271ade71c3e584', 'definitions': ['TemporalPatterns', 'BGST']}, 'ablation_experiments/joint_full_v3/scripts/model.py': {'sha256': 'ae8c2ebe72640befeb249325d4250d9a49a4093e055e36197337bf066dd758e9', 'definitions': ['JointNative', 'JointBGST', 'JointFull']}, 'teddcf/v2/s0/state_evidence.py': {'sha256': '5168b9c283031195aefd28305271d8392e50fcb4ee6999b6d775face1d14c135', 'definitions': ['S0StateEvidenceProvider']}, 'ablation_experiments/joint_full_v3/scripts/data.py': {'sha256': '24026ba8ffdf745b39d3f7094415a69cd53e31e4b3ecfc0678d27afd350d3fae', 'definitions': ['ChannelScaler', 'JointData', 'JointProvider']}}
