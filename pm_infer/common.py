import csv, hashlib, json, os, random
from pathlib import Path
import numpy as np
import torch
import yaml

# Original pretrained LoViT ViT patch embedding is fixed at 224 x 224.
LOVIT_IMAGE_SIZE = 224

def config(path):
    p = Path(path).resolve()
    cfg = yaml.safe_load(p.read_text(encoding='utf-8'))
    shape = cfg['preprocess'].get('resize')
    if shape != [LOVIT_IMAGE_SIZE, LOVIT_IMAGE_SIZE]:
        raise ValueError(
            f'LoViT checkpoint requires preprocess.resize: [{LOVIT_IMAGE_SIZE}, {LOVIT_IMAGE_SIZE}], got {shape}. '
            'Update config.yaml and rerun prepare before training.'
        )
    for name in ('labels', 'lovit_repo','lovit_checkpoint','output_dir','external_video_dir'):
        val = cfg['paths'].get(name)
        if val and not Path(val).is_absolute():
            cfg['paths'][name] = str((p.parent.parent / val).resolve())
    return cfg

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def json_write(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:20]

def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields=None):
    rows=list(rows); path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None: fields=list(rows[0]) if rows else []
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LabelBook:
    def __init__(self, path):
        self.path=str(path)
        self.data=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        self.mapper={str(k).strip():str(v) for k,v in self.data['raw_to_canonical'].items()}
        self.names=list(self.data['phase_order'])
        if len(set(self.names))!=len(self.names) or not self.names:
            raise ValueError('phase_order must contain unique, nonempty class names')
        self.to_id={v:i for i,v in enumerate(self.names)}
        self.inactive=set(map(str,self.data.get('inactive_raw',[])))
        self.unlabeled=set(map(str,self.data.get('unlabeled_raw',[])))
        self.exclude=set(self.data.get('excluded_phase',[]))
        self.unknown=self.data.get('unknown_policy','skip_video')
        assert self.unknown in ('error','skip_frame','skip_video')
        for raw, canonical in self.mapper.items():
            if canonical not in self.to_id and canonical not in self.exclude:
                raise ValueError(f'{raw} -> {canonical}: not in phase_order or excluded_phase')
        self.signature=sha(self.data)
    @property
    def n(self): return len(self.names)
    def map(self,raw):
        raw=str(raw).strip()
        if raw in self.unlabeled:
            return {'raw':raw, 'canonical':'unlabeled', 'phase_id':-100, 'activity_id':-100}
        if raw not in self.mapper:
            if self.unknown=='error': raise ValueError(f'Unknown raw label {raw}')
            return None
        name=self.mapper[raw]
        return {'raw':raw, 'canonical':name, 'phase_id':self.to_id.get(name,-100),
                'activity_id':0 if raw in self.inactive else 1}
    def validate_checkpoint(self, obj):
        if obj.get('labels_signature')!=self.signature: raise ValueError('Label configuration differs from checkpoint; retrain to avoid incorrect class mapping')
