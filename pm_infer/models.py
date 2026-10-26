"""PhaseMamba paper-based reimplementation. Requires true mamba_ssm; never substitutes an R50 or a dummy recurrent layer."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def require_mamba():
    try: from mamba_ssm import Mamba
    except Exception as e: raise RuntimeError(f'mamba_ssm is required: {e}. Fix CUDA/Torch ABI; no silent fallback.') from e
    return Mamba

def repair_legacy_gelu(model: nn.Module) -> nn.Module:
    """Restore GELU defaults missing from full-module checkpoints saved by older PyTorch.

    Legacy ``nn.GELU`` instances can lack ``approximate`` after unpickling;
    modern PyTorch expects it in forward().  This changes no weights.
    """
    fixed = 0
    for module in model.modules():
        if isinstance(module, nn.GELU) and not hasattr(module, 'approximate'):
            module.approximate = 'none'
            fixed += 1
    if fixed:
        print(f'[LoViT compatibility] Restored approximate=none on {fixed} legacy GELU modules')
    return model

class LovitBackbone(nn.Module):
    """Official LoViT spatial encoder serialized with torch.save(module). Trusted weights only."""
    def __init__(self, repo, checkpoint, dim=768):
        super().__init__()
        repo=Path(repo).resolve();ckpt=Path(checkpoint).resolve()
        if not (repo/'models').is_dir():raise FileNotFoundError(f'Missing official LoViT models/ at {repo}; clone MRUIL/LoViT')
        if not ckpt.is_file():raise FileNotFoundError(f'Missing official LoViT checkpoint {ckpt}')
        sys.path.insert(0,str(repo))
        # Original LoViT checkpoint is an nn.Module pickle. ONLY use trusted local checkpoint.
        obj=torch.load(ckpt,map_location='cpu',weights_only=False)
        if isinstance(obj,dict):
            raise TypeError('Expected original LoViT nn.Module pickle, not a state_dict; no alternative encoder is substituted.')
        if not isinstance(obj,nn.Module):raise TypeError('LoViT checkpoint is not an nn.Module')
        self.encoder=repair_legacy_gelu(obj);self.dim=dim
    def forward(self,x):
        # expects x [B,3,H,W] -> [B,3,1,H,W], official encoder output [B,768,1,1,1]
        x=self.encoder(x.unsqueeze(2))
        if isinstance(x,(tuple,list)):x=x[0]
        if not torch.is_tensor(x):raise RuntimeError('LoViT encoder did not return a Tensor')
        b=x.shape[0]; x=x.reshape(b,-1)
        if x.shape[1]!=self.dim:raise RuntimeError(f'LoViT feature output dimension {x.shape[1]} != {self.dim}')
        return x

class Stage0(nn.Module):
    """Fine-tune LoViT spatial model with causal context head; discard the head for feature extraction."""
    def __init__(self,encoder,classes,dim,heads=8,layers=1):
        super().__init__();self.encoder=encoder
        layer=nn.TransformerEncoderLayer(dim,heads,dim*2,batch_first=True,dropout=0.1)
        self.context=nn.TransformerEncoder(layer,num_layers=layers,enable_nested_tensor=False)
        self.classifier=nn.Linear(dim,classes)
    def forward(self,images):
        b,t,c,h,w=images.shape
        z=self.encoder(images.reshape(b*t,c,h,w)).reshape(b,t,-1)
        mask=torch.ones(t,t,device=z.device,dtype=torch.bool).triu(1)
        z=self.context(z,mask=mask,is_causal=True)
        return self.classifier(z),F.normalize(z,dim=-1)

class TemporalConv(nn.Module):
    def __init__(self,dim,causal):
        super().__init__();self.causal=causal
        self.dw=nn.Conv1d(dim,dim,3,groups=dim,padding=0 if causal else 1)
        self.pw=nn.Conv1d(dim,dim,1)
        self.norm=nn.LayerNorm(dim)
    def forward(self,x):
        z=x.transpose(1,2)
        if self.causal:z=F.pad(z,(2,0))
        return self.norm(x+F.gelu(self.pw(self.dw(z)).transpose(1,2)))

class BiMambaBlock(nn.Module):
    def __init__(self,d,d_state):
        super().__init__();Mamba=require_mamba()
        self.fw=Mamba(d_model=d,d_state=d_state,d_conv=4,expand=2)
        self.bw=Mamba(d_model=d,d_state=d_state,d_conv=4,expand=2)
        self.norm=nn.LayerNorm(d)
    def forward(self,x):
        a=self.fw(self.norm(x));b=self.bw(self.norm(x).flip(1)).flip(1)
        return x+F.gelu((a+b)*x)

class ForwardBlock(nn.Module):
    def __init__(self,d,d_state):
        super().__init__();Mamba=require_mamba()
        self.norm=nn.LayerNorm(d)
        self.mamba=Mamba(d_model=d,d_state=d_state,d_conv=4,expand=2)
    def forward(self,x):return x+self.mamba(self.norm(x))

class Stage1(nn.Module):
    def __init__(self,dim,hidden,classes,blocks=2,d_state=16):
        super().__init__();self.norm=nn.LayerNorm(dim)
        self.proj=nn.Linear(dim,hidden)
        self.tcm=TemporalConv(hidden,causal=False)
        self.proj2=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden))
        self.blocks=nn.ModuleList([BiMambaBlock(hidden,d_state) for _ in range(blocks)])
        self.phase_head=nn.Linear(hidden,classes)
        self.transition_head=nn.Linear(hidden,1)
    def local(self,x):
        z=self.proj(self.norm(x));z=self.proj2(self.tcm(z))
        for block in self.blocks:z=block(z)
        return z
    def forward(self,x):
        z=self.local(x)
        pooled=z.mean(1) # window-level dominant phase + transition presence
        return self.phase_head(pooled),self.transition_head(pooled).squeeze(-1),z

class Stage2(nn.Module):
    def __init__(self,dim,hidden,classes,blocks=3,d_state=16):
        super().__init__()
        self.spatial=nn.Linear(dim,hidden);self.local=nn.Linear(hidden,hidden)
        self.gate=nn.Linear(hidden*2,hidden)
        self.tcm=TemporalConv(hidden,causal=True)
        self.blocks=nn.ModuleList([ForwardBlock(hidden,d_state) for _ in range(blocks)])
        self.phase_head=nn.Linear(hidden,classes)
        self.activity_head=nn.Linear(hidden,2)
    def forward(self,spatial,local):
        f=self.spatial(spatial);p=self.local(local)
        g=torch.sigmoid(self.gate(torch.cat((f,p),-1)))
        z=self.tcm(F.relu(g*f+(1-g)*p))
        for m in self.blocks:z=m(z)
        return self.phase_head(z),self.activity_head(z)


def supcon_loss(emb,y,temperature=0.07):
    # SupCon positives share the same valid phase, excluding self.
    emb=F.normalize(emb.reshape(-1,emb.shape[-1]),dim=-1)
    y=y.reshape(-1);keep=y>=0;emb=emb[keep];y=y[keep]
    if y.numel()<2:return emb.sum()*0
    dot=emb@emb.T /temperature
    mask=~torch.eye(len(y),device=y.device,dtype=torch.bool)
    pos=(y[:,None]==y[None,:]) & mask
    dot=dot.masked_fill(~mask,float('-inf'))
    logp=dot-torch.logsumexp(dot,dim=1,keepdim=True)
    active=pos.any(1)
    if not active.any():return emb.sum()*0
    return -((logp.masked_fill(~pos,0).sum(1)/pos.sum(1).clamp(min=1))[active]).mean()
