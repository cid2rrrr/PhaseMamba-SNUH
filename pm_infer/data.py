from collections import defaultdict
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import Dataset
from .common import read_csv, LOVIT_IMAGE_SIZE

def load_index(cfg):
    p=Path(cfg['paths']['output_dir'])/'prepared'/'index.csv'
    if not p.exists():raise FileNotFoundError(f'Run prepare first: {p}')
    rows=read_csv(p); d=defaultdict(list)
    for x in rows:
        d[x['video']].append({'sec':int(x['sec']),'image':x['image'],
                            'phase':int(x['phase_id']),'activity':int(x['activity_id']),
                            'raw':x['raw'],'canonical':x['canonical']})
    for v in d:d[v].sort(key=lambda x:x['sec'])
    return dict(d)

def transform(image):
    im=cv2.imread(str(image))
    if im is None:raise FileNotFoundError(f'Missing/corrupt image: {image}')
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im=cv2.resize(im,(LOVIT_IMAGE_SIZE,LOVIT_IMAGE_SIZE))
    x=torch.from_numpy(im.copy()).permute(2,0,1).float()/255.
    mean=x.new_tensor([0.485,0.456,0.406])[:,None,None]
    std=x.new_tensor([0.229,0.224,0.225])[:,None,None]
    return (x-mean)/std

class Stage0Clips(Dataset):
    def __init__(self,index,videos,clip=8,max_per_class=200):
        self.index=index;self.clip=clip;self.items=[]
        for name in videos:
            rows=index[name];counts=defaultdict(int)
            for i,r in enumerate(rows):
                if r['phase']<0:continue
                if counts[r['phase']]>=max_per_class:continue
                # Only adjacent 1s frames belong to clips; no jumps across missing frames.
                start=i
                while start>0 and i-start+1<clip and rows[start]['sec']==rows[start-1]['sec']+1:
                    start-=1
                if i-start+1<clip:continue
                self.items.append((name,start,i+1));counts[r['phase']]+=1
        if not self.items:raise ValueError('No Stage0 training clips after filtering')
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        name,a,b=self.items[i];rows=self.index[name][a:b]
        return torch.stack([transform(r['image']) for r in rows]),torch.tensor([r['phase'] for r in rows],dtype=torch.long)

class Stage1Windows(Dataset):
    def __init__(self,features,index,videos,window=64,stride=32):
        self.entries=[]
        for v in videos:
            arr=features[v];rows=index[v]
            if len(arr)!=len(rows):raise RuntimeError(f'feature count mismatch {v}')
            for a in range(0,len(rows),stride):
                b=min(a+window,len(rows))
                if b-a<2:continue
                self.entries.append((v,a,b))
            if len(rows)>window:self.entries.append((v,len(rows)-window,len(rows)))
        self.features=features;self.index=index
    def __len__(self):return len(self.entries)
    def __getitem__(self,i):
        v,a,b=self.entries[i];rows=self.index[v][a:b]
        labels=[r['phase'] for r in rows];valid=[k for k in labels if k>=0]
        if not valid:phase=-100
        else:phase=max(set(valid),key=lambda k:(valid.count(k),-k))
        transition=float(any(labels[j]!=labels[j-1] and min(labels[j],labels[j-1])>=0 for j in range(1,len(labels))))
        return self.features[v][a:b],torch.tensor(phase),torch.tensor(transition)

def stage1_collate(batch):
    """Batch size 1 avoids padding and accidentally including pad tokens in pooling."""
    assert len(batch)==1,'Stage1 dataloader batch_size must be 1'
    a,b,c=batch[0];return a.unsqueeze(0),b.unsqueeze(0),c.unsqueeze(0)
