"""Conda/RTX3090-only PhaseMamba offline inference; no training routines.

Checkpoint semantics are identical to vitphase.models from 2026-10-23.
The model contains a learned 2-way activity head [Inactive,Active], never a hand-made idle detector.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

from .common import LabelBook, json_write
from .models import LovitBackbone, Stage0, Stage1, Stage2, require_mamba
from .data import transform
from .preprocess import _extract_external_ffmpeg, VIDEO_EXT
from .visualize import export_predictions


def read_weights(path, module, book, dev):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Trained checkpoint missing: {path}')
    # These are user-generated local/trusted checkpoints; never unpickle untrusted files.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or 'model' not in payload:
        raise ValueError(f'{path}: expected vitphase checkpoint with model and labels_signature')
    book.validate_checkpoint(payload)
    module.load_state_dict(payload['model'], strict=True)
    return module.to(dev).eval()


def choose_checkpoint(run, stage, mode):
    directory = run/stage
    chosen = directory/f'{mode}.pt'
    if not chosen.is_file():
        raise FileNotFoundError(f'{chosen} not found; use --weights best/last according to the run directory')
    return chosen


def load_config(run):
    path = run/'run_config.json'
    if not path.exists():
        raise FileNotFoundError(f'Missing run_config.json inside {run}. This bundle requires the vitphase run directory, not an arbitrary PyTorch .pt')
    obj = json.loads(path.read_text(encoding='utf-8'))
    if 'config' not in obj or 'labels_signature' not in obj:
        raise ValueError('run_config.json lacks config and/or labels_signature')
    cfg = obj['config']
    if cfg['preprocess'].get('resize') != [224,224] or cfg['preprocess'].get('fps') != 1:
        raise ValueError('Original LoViT inference expects 224x224, 1 frame per second')
    if cfg['model'].get('feature_dim') != 768:
        raise ValueError('This deployment supports LoViT 768-D only')
    return obj,cfg


def get_encoder(run, cfg, book, args, dev):
    m, t = cfg['model'],cfg['train']
    ckpt=Path(args.lovit_checkpoint).expanduser().resolve()
    repo=Path(args.lovit_repo).expanduser().resolve()
    encoder=LovitBackbone(repo,ckpt,m['feature_dim'])
    if not t.get('skip_stage0',False):
        s0=Stage0(encoder,book.n,m['feature_dim'],t['stage0_temporal_heads'],t['stage0_temporal_layers'])
        # Stage0 is trained with LoViT, and the encoder is updated by loading this exact checkpoint.
        s0=read_weights(choose_checkpoint(run,'stage0',args.weights),s0,book,dev)
        encoder=s0.encoder
    elif t.get('stage0_checkpoint'):
        source=Path(args.stage0_checkpoint or t['stage0_checkpoint']).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f'Run used externally fine-tuned Stage0: pass --stage0-checkpoint PATH; missing {source}')
        s0=Stage0(encoder,book.n,m['feature_dim'],t['stage0_temporal_heads'],t['stage0_temporal_layers'])
        s0=read_weights(source,s0,book,dev)
        encoder=s0.encoder
    encoder.eval().to(dev)
    return encoder


def images_to_features(rows,encoder,dev,batch_size):
    features=[]
    with torch.inference_mode():
        for i in tqdm(range(0,len(rows),batch_size),desc='LoViT 1-FPS',leave=False):
            image_batch=torch.stack([transform(r['image']) for r in rows[i:i+batch_size]]).to(dev)
            features.append(encoder(image_batch).float().cpu())
    x=torch.cat(features,dim=0)
    if x.shape!=(len(rows),768):
        raise RuntimeError(f'LoViT feature mismatch: {tuple(x.shape)}')
    return x


def features_from_pt(path):
    """Accept tensors [T,768] or dict(features, secs). User must ensure identical LoViT weights/preprocess."""
    obj=torch.load(path,map_location='cpu',weights_only=False)
    sec=None
    if isinstance(obj,dict):
        for name in ('sec','secs','seconds','timestamps'):
            if name in obj:
                sec=obj[name];break
        for name in ('features','feature','embeddings','x'):
            if name in obj:
                obj=obj[name];break
        else:
            raise ValueError(f'{path}: no features/feature/embeddings/x in mapping')
    if not isinstance(obj,torch.Tensor):
        obj=torch.as_tensor(obj)
    if obj.ndim!=2 or obj.shape[1]!=768:
        raise ValueError(f'{path}: expected [T,768] LoViT features; got {list(obj.shape)}')
    if not torch.isfinite(obj).all():
        raise ValueError('Non-finite precomputed feature values')
    x=obj.float().cpu().contiguous()
    secs=list(range(len(x))) if sec is None else torch.as_tensor(sec).flatten().tolist()
    secs=[int(s) for s in secs]
    if len(secs)!=len(x) or secs!=sorted(set(secs)):
        raise ValueError('Feature secs must be unique ascending integers, one per feature')
    rows=[{'video':path.stem,'sec':s,'image':'','phase':-100,'activity':-100,'raw':'','canonical':''} for s in secs]
    return x, rows


def local_features(x,stage1,window,dev,batch_size):
    """Identical to vitphase.engine.local_feature_cache, including variable-length early frames."""
    window=int(window)
    xs=x.to(dev);seq=[]
    with torch.inference_mode():
        early=min(len(xs),window-1)
        for i in tqdm(range(early),desc='Stage1 warmup',leave=False):
            seq.append(stage1.local(xs[:i+1].unsqueeze(0))[:,-1,:].cpu())
        for start in tqdm(range(early,len(xs),batch_size),desc='Stage1 windows',leave=False):
            stop=min(start+batch_size,len(xs))
            batch=torch.stack([xs[i-window+1:i+1] for i in range(start,stop)])
            seq.append(stage1.local(batch)[:,-1,:].cpu())
    return torch.cat(seq,0)


def infer_one(run,cfg,book,stage1,stage2,encoder,pri,dev,args,path):
    output=Path(args.output)/Path(path).stem
    output.mkdir(parents=True,exist_ok=True)
    if args.features_pt:
        x,rows=features_from_pt(path)
        source=None
    else:
        frames,info=_extract_external_ffmpeg(path,Path(args.output)/'_frames'/path.stem,
            [224,224],cfg['preprocess'].get('frame_index_start',0),not args.no_reuse_frames,args.min_coverage)
        rows=[{'video':path.stem,'sec':sec,'image':str(fp),'phase':-100,'activity':-100,
               'raw':'','canonical':''} for sec,fp in frames]
        json_write(output/'decode_info.json',info)
        x=images_to_features(rows,encoder,dev,args.batch_size)
        source=path
    local=local_features(x,stage1,cfg['model']['window'],dev,args.local_batch_size)
    # This is full-procedure causal Mamba; no resets/chunking that alter temporal context.
    with torch.inference_mode():
        logits,activity=stage2(x.to(dev)[None],local.to(dev)[None])
    probs=logits[0].softmax(-1).float().cpu().numpy()
    ap=activity[0].softmax(-1).float().cpu().numpy()
    if not np.isfinite(probs).all() or not np.isfinite(ap).all():
        raise RuntimeError('Non-finite prediction probabilities')
    if pri is None and cfg['postprocess']['refinement']:
        raise ValueError('Refinement requested, but no transition_prior.npy exists')
    export_predictions(output,rows,probs,ap,book,cfg,pri,source_video=source)
    # Extra machine-readable probabilities, independent of phase gating.
    np.savez_compressed(output/'probabilities.npz',phase=probs,activity=ap,
                        seconds=np.asarray([r['sec'] for r in rows]))
    json_write(output/'provenance.json',{'run':str(run),'source':str(path),'weights':args.weights,
        'labels_signature':book.signature,'stage2_activity_head':True,'samples':len(x),
        'features_pt':bool(args.features_pt),'phase_order':book.names})
    print(f'[DONE] {path.name}: {len(rows)} timestamps -> {output / "predictions.csv"}',flush=True)


def verify_environment():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check PyTorch CUDA build, NVIDIA driver and active Conda environment')
    if tuple(torch.cuda.get_device_capability(0)) != (8,6):
        warnings.warn(f'Target was RTX 3090 (SM 8.6); detected {torch.cuda.get_device_name(0)}')
    print(f'Torch={torch.__version__}, CUDA runtime={torch.version.cuda}, GPU={torch.cuda.get_device_name(0)}')
    Mamba=require_mamba()
    with torch.inference_mode():
        y=Mamba(d_model=256,d_state=16,d_conv=4,expand=2).to('cuda')(
            torch.randn(1,8,256,device='cuda'))
    if y.shape!=(1,8,256):
        raise RuntimeError('Mamba kernel smoke check produced wrong shape')
    print('Mamba CUDA kernel: PASS')


def main(argv=None):
    parser=argparse.ArgumentParser(description='PhaseMamba LoViT + Stage2 Active/Inactive inference on Conda / RTX 3090')
    parser.add_argument('--check',action='store_true',help='Check CUDA+real Mamba kernel, no checkpoint needed')
    parser.add_argument('--check-lovit',action='store_true',help='With --check, also test official LoViT 224px -> 768D')
    parser.add_argument('--run-dir',type=Path,help='Trained vitphase full/ or cv/fold_XX/ folder with run_config.json')
    parser.add_argument('--input',type=Path,help='Video file OR directory (.mp4,.mov,.avi,.mkv)')
    parser.add_argument('--features-pt',type=Path,help='Already extracted LoViT feature .pt file OR directory; mutually exclusive with --input')
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    parser.add_argument('--labels',type=Path,default=Path('configs/labels.yaml'))
    parser.add_argument('--weights',choices=['last','best'],default='last')
    parser.add_argument('--lovit-repo',type=Path,default=Path('third_party/LoViT'))
    parser.add_argument('--lovit-checkpoint',type=Path,default=Path('weights/Trained_VIT_Cholec80.pth'))
    parser.add_argument('--stage0-checkpoint',type=Path,help='Override originally used external Stage0 checkpoint')
    parser.add_argument('--batch-size',type=int,default=4,help='LoViT frame mini-batch (RTX3090 default 4)')
    parser.add_argument('--local-batch-size',type=int,default=16)
    parser.add_argument('--min-coverage',type=float,default=0.8)
    parser.add_argument('--no-reuse-frames',action='store_true')
    parser.add_argument('--activity-gate',action='store_true',help='Set final phase to -1 if activity head predicts Inactive')
    parser.add_argument('--no-activity-gate',action='store_true',help='Disable phase gating even if trained config enables it')
    parser.add_argument('--no-refinement',action='store_true',help='Use raw Stage2 phase predictions; keep activity head')
    parser.add_argument('--no-overlay',action='store_true',help='Skip overlay MP4 generation; CSV/PNG remain')
    parser.add_argument('--overlay-speed',type=float,default=None,help='Display speed multiplier, 2=2x')
    parser.add_argument('--overlay-resolution',choices=['original','model'],default=None)
    parser.add_argument('--strict',action='store_true',help='Stop batch on first bad file (default: write failures report and continue)')
    args=parser.parse_args(argv)
    if args.check:
        verify_environment()
        if args.check_lovit:
            encoder=LovitBackbone(args.lovit_repo,args.lovit_checkpoint,768).cuda().eval()
            with torch.inference_mode():
                feat=encoder(torch.randn(1,3,224,224,device='cuda'))
            if tuple(feat.shape)!=(1,768):raise RuntimeError('LoViT 768-D feature check FAILED')
            print('LoViT original checkpoint: PASS')
        return 0
    if args.run_dir is None:
        parser.error('--run-dir is required')
    if (args.input is None)==(args.features_pt is None):
        parser.error('Provide exactly one of --input or --features-pt')
    if args.batch_size<1 or args.local_batch_size<1:
        parser.error('batch sizes must be positive')
    if args.activity_gate and args.no_activity_gate:
        parser.error('Choose only one activity gate option')
    if args.features_pt and args.overlay_resolution=='original' and not args.no_overlay:
        parser.error('Feature-only .pt has no source video; use --no-overlay or --overlay-resolution model')
    verify_environment()
    run=args.run_dir.expanduser().resolve()
    meta,cfg=load_config(run)
    book=LabelBook(args.labels.expanduser().resolve())
    if meta['labels_signature']!=book.signature:
        raise ValueError('Label signature mismatch: restore the *original training* labels.yaml, do not edit IDs or phase_order')
    if args.activity_gate:cfg['postprocess']['activity_gate']=True
    if args.no_activity_gate:cfg['postprocess']['activity_gate']=False
    if args.no_refinement:cfg['postprocess']['refinement']=False
    if args.no_overlay or args.features_pt:cfg['report']['overlay_mp4']=False
    if args.overlay_speed is not None:cfg['report']['overlay_speed']=args.overlay_speed
    if args.overlay_resolution:cfg['report']['overlay_resolution']=args.overlay_resolution
    cfg['postprocess']['egtp']=False   # inference package has no StableSPR dependency.
    dev=torch.device('cuda:0')
    m=cfg['model']
    stage1=read_weights(choose_checkpoint(run,'stage1',args.weights),
        Stage1(m['feature_dim'],m['hidden_dim'],book.n,m['stage1_blocks'],m['d_state']),book,dev)
    stage2=read_weights(choose_checkpoint(run,'stage2',args.weights),
        Stage2(m['feature_dim'],m['hidden_dim'],book.n,m['stage2_blocks'],m['d_state']),book,dev)
    if stage2.activity_head.out_features!=2:
        raise AssertionError('Stage2 checkpoint does not include the required 2-class activity head')
    encoder=None
    if args.input:
        encoder=get_encoder(run,cfg,book,args,dev)
    pri_path=run/'transition_prior.npy'
    pri=np.load(pri_path) if pri_path.is_file() else None
    if pri is not None and pri.shape!=(book.n,book.n):
        raise ValueError(f'Invalid transition prior {pri.shape}; expected {(book.n,book.n)}')
    root=(args.features_pt or args.input).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    ext={'.pt'} if args.features_pt else VIDEO_EXT
    paths=([root] if root.is_file() else sorted(p for p in root.iterdir() if p.suffix.lower() in ext))
    if not paths or any(p.suffix.lower() not in ext for p in paths):
        raise ValueError(f'No suitable inputs ({ext}) found in {root}')
    if len({p.stem for p in paths})!=len(paths):
        raise ValueError('Duplicate input stems: output paths would collide')
    args.output.mkdir(parents=True,exist_ok=True)
    failed=[]
    for path in tqdm(paths,desc='Videos'):
        try:
            infer_one(run,cfg,book,stage1,stage2,encoder,pri,dev,args,path)
        except (Exception,KeyboardInterrupt) as ex:
            if isinstance(ex,KeyboardInterrupt):raise
            print(f'[FAILED] {path.name}: {type(ex).__name__}: {ex}',file=sys.stderr,flush=True)
            failed.append({'video':str(path),'error':f'{type(ex).__name__}: {ex}'})
            if args.strict:raise
    json_write(args.output/'summary.json',{'ok':len(paths)-len(failed),'failed':failed,'phase_order':book.names})
    if len(failed)==len(paths):
        raise RuntimeError('All videos failed. See errors in output/summary.json')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
