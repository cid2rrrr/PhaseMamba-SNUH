from pathlib import Path
import shutil,subprocess,tempfile,warnings
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import write_csv,read_csv,json_write

def export_predictions(out,rows,probs,activity_prob,books,cfg,prior,source_video=None):
    from .postprocess import refine,egtp
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=probs.argmax(1);ref=refine(probs,prior,cfg['postprocess']['refine_confidence'],cfg['postprocess']['refine_alpha']) if cfg['postprocess']['refinement'] else raw.copy()
    eg=egtp(probs,cfg['postprocess']['egtp_threshold']) if cfg['postprocess']['egtp'] else None
    activity=activity_prob.argmax(1)
    final=eg if eg is not None else ref
    if cfg['postprocess']['activity_gate']:
        final=np.where(activity==0,-1,final)
    cols=[]
    for i,r in enumerate(rows):
        cv=lambda q:'Inactive' if int(q)<0 else books.names[int(q)]
        cols.append({'video':r.get('video','external'),'sec':r['sec'],
          'gt_phase':r['phase'],'gt_activity':r['activity'],
          'raw':int(raw[i]),'raw_name':cv(raw[i]),'refined':int(ref[i]),'refined_name':cv(ref[i]),
          'egtp':int(eg[i]) if eg is not None else '',
          'egtp_name':cv(eg[i]) if eg is not None else '',
          'activity':int(activity[i]),'activity_confidence':float(activity_prob[i].max()),
          'p_inactive':float(activity_prob[i,0]),'p_active':float(activity_prob[i,1]),
          'final':int(final[i]),'final_name':cv(final[i]),
          'gt_name':cv(r['phase']) if r['phase']>=0 else '',
          'confidence':float(probs[i].max()),'image':r['image']})
    write_csv(out/'predictions.csv',cols)
    timeline(out/'timeline.png',cols,books)
    if cfg['report'].get('overlay_mp4',True):
        rep=cfg['report']
        try:
            overlay(out/'overlay.mp4',cols,rep.get('overlay_fps',1),
                    source_video=source_video,
                    resolution=rep.get('overlay_resolution','original'),
                    speed=rep.get('overlay_speed',2.0))
        except (OSError,RuntimeError,ValueError) as exc:
            # Inference results and timeline are still valid if visualization fails.
            message=f'Overlay rendering failed ({source_video}): {type(exc).__name__}: {exc}'
            (out/'overlay_error.txt').write_text(message+'\n',encoding='utf-8')
            print('[Overlay] '+message,flush=True)
    return cols
    return cols

def timeline(path,rows,books):
    cols=['gt_phase','raw','refined','egtp','final','gt_activity','activity']
    cols=[c for c in cols if any(r.get(c,'')!='' and (c!='gt_phase' or int(r[c])>=0) for r in rows)]
    names={'gt_phase':'Ground truth','raw':'Raw PhaseMamba','refined':'Phase-order refinement',
           'egtp':'EGTP','final':'Final output','activity':'Predicted activity','gt_activity':'GT activity'}
    fig,axes=plt.subplots(len(cols),1,figsize=(16,max(4,1.35*len(cols))),sharex=True,squeeze=False)
    for ax,c in zip(axes[:,0],cols):
        vals=np.array([int(r[c]) if str(r.get(c,''))!='' else -1 for r in rows]);sec=np.array([int(r['sec']) for r in rows]);
        ax.step(sec,vals,where='post',linewidth=1);ax.set_ylabel(names[c],fontsize=8)
        ax.grid(alpha=.2)
        if c not in ('activity','gt_activity'):
            ax.set_yticks(list(range(books.n)),books.names,fontsize=6)
        else:ax.set_yticks([0,1],['Inactive','Active'],fontsize=7)
    axes[-1,0].set_xlabel('Second');fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)

def _original_frames(source_video, out):
    """Return (sec, jpeg) using sequential FFmpeg extraction at ORIGINAL resolution.

    Identical FPS/PTS policy to external model frames, but never scales to 224.
    This is presentation only; network input remains 224x224.
    """
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('FFmpeg is required for original-resolution overlay')
    video=Path(source_video)
    if not video.is_file():
        raise FileNotFoundError(f'Original video does not exist: {video}')
    cmd=[ffmpeg,'-hide_banner','-nostdin','-loglevel','warning',
         '-fflags','+discardcorrupt','-err_detect','ignore_err',
         '-i',str(video),'-map','0:v:0','-an','-sn',
         '-vf','fps=1:round=down','-frame_pts','1','-q:v','3',
         '-f','image2',str(Path(out)/'%06d.jpg')]
    with (Path(out)/'ffmpeg_stderr.log').open('w',encoding='utf-8') as err:
        p=subprocess.run(cmd,stderr=err,stdout=subprocess.DEVNULL,check=False)
    frames=[]
    for fp in Path(out).glob('*.jpg'):
        try: sec=int(fp.stem)
        except ValueError: continue
        if sec>=0:frames.append((sec,fp))
    if not frames:
        log=(Path(out)/'ffmpeg_stderr.log').read_text(encoding='utf-8',errors='replace')
        raise RuntimeError(f'No frames decoded from {video}; FFmpeg status={p.returncode}: {log[-500:]}')
    if p.returncode:
        print(f'[Overlay] FFmpeg exited {p.returncode} on {video.name}; salvaged {len(frames)} frames',flush=True)
    return sorted(frames)


def _draw_overlay(im,row):
    h,w=im.shape[:2]
    # Scale typography with original image while retaining legibility at 224px.
    scale=max(0.48,min(1.8,w/1000.0))
    thickness=max(1,round(scale*1.7))
    line_height=round(32*scale)
    margin=round(12*scale)
    bar_h=min(h,max(44,round(2.8*line_height)))
    # Semi-transparent overlay keeps the underlying surgical image visible.
    layer=im.copy()
    cv2.rectangle(layer,(0,0),(w,bar_h),(0,0,0),-1)
    cv2.addWeighted(layer,0.62,im,0.38,0,im)
    texts=[f"{row['sec']}s  Phase: {row['final_name']}",
           f"Activity: {'ACTIVE' if int(row['activity'])==1 else 'INACTIVE'}  Conf: {float(row['confidence']):.3f}"]
    for i,txt in enumerate(texts):
        yy=min(h-5,margin+line_height*(i+1))
        cv2.putText(im,txt,(margin,yy),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),thickness,cv2.LINE_AA)
    return im


def overlay(path,rows,fps=1,source_video=None,resolution='original',speed=2.0):
    """Create overlay using either original video resolution or model input images.

    Prediction rate is 1/sec. 2x means encode the resulting sampled sequence
    at 2 FPS, not extrapolate predictions or alter inference timestamps.
    """
    if not rows:return
    if resolution not in ('original','model'):
        raise ValueError('overlay_resolution must be original or model')
    speed=float(speed)
    if not (0.1<=speed<=32):raise ValueError('overlay_speed must be in [0.1,32]')
    fps=float(fps)
    if fps<=0:raise ValueError('overlay_fps must be positive')
    out=Path(path);out.parent.mkdir(parents=True,exist_ok=True)
    frame_list=[]
    if resolution=='original':
        if not source_video:
            raise RuntimeError('Original-resolution overlay needs source_video path')
        with tempfile.TemporaryDirectory(prefix='phase_overlay_',dir=out.parent) as td:
            frame_list=_original_frames(source_video,td)
            err_file=Path(td)/'ffmpeg_stderr.log'
            user_log=out.parent/'overlay_decode.log'
            if err_file.is_file() and err_file.stat().st_size:
                shutil.copy2(err_file,user_log)
                print(f'[Overlay] FFmpeg decode warnings: {user_log}',flush=True)
            else:
                user_log.unlink(missing_ok=True)
            # Preserve alignment if some seconds were missing/corrupt.
            labels={int(r['sec']):r for r in rows}
            pairs=[(fp,labels[sec]) for sec,fp in frame_list if sec in labels]
            return _write_overlay(out,pairs,fps*speed)
    else:
        return _write_overlay(out,[(Path(r['image']),r) for r in rows],fps*speed)


def _write_overlay(path,source_rows,output_fps):
    # Never leave an older low-resolution overlay behind after a failed render.
    path.unlink(missing_ok=True)
    writer=None
    rendered=0
    try:
        for fp,row in source_rows:
            im=cv2.imread(str(fp))
            if im is None:continue
            h,w=im.shape[:2]
            if writer is None:
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),output_fps,(w,h))
                if not writer.isOpened():raise RuntimeError(f'Failed to open MP4 writer: {path}')
            elif (h,w)!=(height,width):
                raise RuntimeError(f'Inconsistent source resolution for overlay: {(w,h)} != {(width,height)}')
            if rendered==0:height,width=h,w
            writer.write(_draw_overlay(im,row))
            rendered+=1
    finally:
        if writer is not None:writer.release()
    if not rendered:
        path.unlink(missing_ok=True)
        raise RuntimeError('No matching decodable original-resolution frames for video overlay')
    print(f'[Overlay] Saved {path}: {width}x{height}, {output_fps:g} fps, {rendered} frames',flush=True)
    return {'width':width,'height':height,'fps':output_fps,'frames':rendered}


def rerender_external(cfg,external=None):
    """Rebuild overlay MP4s from saved prediction CSVs. No LoViT/GPU/retraining.

    Useful when only visualization resolution or playback speed is changed.
    """
    from .preprocess import VIDEO_EXT
    p=cfg['paths'];root=Path(p['output_dir'])/'full'/'external_predictions'
    if not root.is_dir():
        raise FileNotFoundError(f'Missing external predictions: {root}; run predict first')
    src=Path(external or p['external_video_dir'])
    videos=[src] if src.is_file() else sorted(v for v in src.iterdir() if v.suffix.lower() in VIDEO_EXT) if src.is_dir() else []
    if not videos:raise FileNotFoundError(f'No input MP4s at {src}')
    by_stem={}
    for v in videos:
        if v.stem in by_stem:raise RuntimeError(f'Duplicate source filename stem: {v.stem}')
        by_stem[v.stem]=v
    processed=[];skipped=[]
    for out in sorted(x for x in root.iterdir() if x.is_dir()):
        if src.is_file() and out.name!=src.stem:continue
        csv_path=out/'predictions.csv'
        if not csv_path.is_file():continue
        source=by_stem.get(out.name)
        if not source:
            skipped.append({'video':out.name,'reason':'original MP4 not supplied'})
            continue
        try:
            rows=read_csv(csv_path)
            if not rows:raise ValueError('Empty prediction CSV')
            rep=cfg['report']
            info=overlay(out/'overlay.mp4',rows,rep.get('overlay_fps',1),
                         source_video=source,resolution=rep.get('overlay_resolution','original'),
                         speed=rep.get('overlay_speed',2.0))
            (out/'overlay_error.txt').unlink(missing_ok=True)
            processed.append({'video':out.name,**info})
        except (OSError,RuntimeError,ValueError) as exc:
            skipped.append({'video':out.name,'reason':str(exc)})
            print(f'[Render] SKIPPED {out.name}: {exc}',flush=True)
    json_write(root/'render_report.json',{'rendered':processed,'skipped':skipped})
    write_csv(root/'render_skipped.csv',skipped,fields=['video','reason'])
    if not processed:
        raise RuntimeError(f'No overlays were produced. Check {root / "render_report.json"}')
    return root



def _cv_sources(cfg):
    """Resolve an OOF video to its original MP4, never to 224x224 cached frames.

    Prefer the saved prepare manifest. If a source directory was moved, use an
    unambiguous case-insensitive filename-stem match in configured video_dir.
    """
    from .preprocess import VIDEO_EXT
    p=cfg['paths']
    manifest_path=Path(p['output_dir'])/'prepared'/'manifest.json'
    mapping={}
    directory_mapping={}
    if manifest_path.is_file():
        manifest=__import__('json').loads(manifest_path.read_text(encoding='utf-8'))
        for item in manifest.get('videos',[]):
            vid=str(item['video'])
            src=item.get('video_path')
            if src and Path(src).is_file():
                mapping[vid.casefold()]=Path(src)
    source_dir=Path(p['video_dir'])
    if source_dir.is_dir():
        for src in sorted(source_dir.iterdir()):
            if src.is_file() and src.suffix.lower() in VIDEO_EXT:
                key=src.stem.casefold()
                if key in mapping: continue
                if key in directory_mapping:
                    raise ValueError(f'Duplicate source video stem (case-insensitive): {src.stem}')
                directory_mapping[key]=src
    mapping.update({key:src for key,src in directory_mapping.items() if key not in mapping})
    return mapping


def rerender_cv(cfg,fold=None):
    """Rerender already evaluated CV/OOF predictions without any GPU or training.

    Reads cv/fold_XX/predictions/<video>/predictions.csv and original MP4s.
    Preserves prediction CSVs, existing metrics and model checkpoints exactly.
    A failed/missing source video is reported while other videos proceed.
    """
    cv_root=Path(cfg['paths']['output_dir'])/'cv'
    if not cv_root.is_dir():
        raise FileNotFoundError(f'CV outputs not found: {cv_root}; run cv first')
    if fold is not None:
        # fold 0, fold 00, fold_00 all refer to the same run.
        fold_text=str(fold).removeprefix('fold_')
        if not fold_text.isdigit():
            raise ValueError(f'Invalid fold {fold!r}; expected 0, 00, or fold_00')
        folders=[cv_root/f'fold_{int(fold_text):02d}']
        if not folders[0].is_dir():
            raise FileNotFoundError(f'CV fold does not exist: {folders[0]}')
    else:
        folders=sorted(x for x in cv_root.glob('fold_*') if x.is_dir())
    if not folders:
        raise FileNotFoundError(f'No fold_XX directories found in {cv_root}')
    sources=_cv_sources(cfg)
    rep=cfg['report']
    processed=[];skipped=[]
    for folder in folders:
        pred_dir=folder/'predictions'
        if not pred_dir.is_dir():
            skipped.append({'fold':folder.name,'video':'*','reason':'missing predictions directory'})
            continue
        for out in sorted(x for x in pred_dir.iterdir() if x.is_dir()):
            csv_path=out/'predictions.csv'
            if not csv_path.is_file():
                skipped.append({'fold':folder.name,'video':out.name,'reason':'missing predictions.csv'})
                continue
            source=sources.get(out.name.casefold())
            if rep.get('overlay_resolution','original')=='original' and source is None:
                skipped.append({'fold':folder.name,'video':out.name,'reason':'original MP4 not found in prepared manifest or video_dir'})
                continue
            try:
                rows=read_csv(csv_path)
                if not rows:raise ValueError('Empty prediction CSV')
                info=overlay(out/'overlay.mp4',rows,rep.get('overlay_fps',1),
                             source_video=source,resolution=rep.get('overlay_resolution','original'),
                             speed=rep.get('overlay_speed',2.0))
                (out/'overlay_error.txt').unlink(missing_ok=True)
                processed.append({'fold':folder.name,'video':out.name,**info})
                print(f'[CV render] {folder.name}/{out.name} OK',flush=True)
            except (OSError,RuntimeError,ValueError) as exc:
                (out/'overlay_error.txt').write_text(f'{type(exc).__name__}: {exc}\n',encoding='utf-8')
                skipped.append({'fold':folder.name,'video':out.name,'reason':str(exc)})
                print(f'[CV render] SKIPPED {folder.name}/{out.name}: {exc}',flush=True)
    report={'rendered':processed,'skipped':skipped,
            'resolution':rep.get('overlay_resolution','original'),
            'speed':rep.get('overlay_speed',2.0)}
    json_write(cv_root/'render_cv_report.json',report)
    write_csv(cv_root/'render_cv_skipped.csv',skipped,fields=['fold','video','reason'])
    if not processed:
        raise RuntimeError(f'No CV overlays were rendered. Check {cv_root / "render_cv_report.json"}')
    print(f'[CV render] Complete: {len(processed)} rendered, {len(skipped)} skipped; report={cv_root / "render_cv_report.json"}',flush=True)
    return cv_root
