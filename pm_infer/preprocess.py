"""Paired CSV/MP4 discovery and reproducible 1-Hz image extraction."""
from pathlib import Path
from collections import Counter
import json, math, re, shutil, subprocess, tempfile
import cv2
from .common import LabelBook, json_write, read_csv, write_csv, sha

VIDEO_EXT={'.mp4','.avi','.mov','.mkv'}
TIME_COLS=['sec','second','seconds','time','timestamp','elapsed','frame','frame_id','frame_index']
LABEL_COLS=['phase_or_event','phase','phase_id','label','label_id','class','class_id','phase_name','annotation']
START_COLS=['start','start_sec','start_time','onset','begin']
END_COLS=['end','end_sec','end_time','offset','stop']

def _column(headers, override, choices):
    if override:
        if override not in headers: raise ValueError(f'Expected CSV column {override}; got {headers}')
        return override
    norm={re.sub('[^a-z0-9]','',h.lower()):h for h in headers}
    for c in choices:
        if re.sub('[^a-z0-9]','',c.lower()) in norm:return norm[re.sub('[^a-z0-9]','',c.lower())]
    return None

def _time(x, unit='second', fps=1):
    s=str(x).strip()
    if ':' in s:
        p=list(map(float,s.split(':')))
        v=sum(v*60**i for i,v in enumerate(p[::-1]))
    else:v=float(s)
    return v/fps if unit=='frame' else v

def load_annotation(csv_path, pp, n_seconds):
    rows=read_csv(csv_path)
    if not rows: raise ValueError('empty annotation CSV')
    hdr=list(rows[0]); label=_column(hdr,pp.get('label_column'),LABEL_COLS)
    # SNUH Vitrectomy CSV includes one type=case row describing the entire surgery.
    # It is metadata, not a training label. Keep phase rows only.
    row_type=_column(hdr,pp.get('type_column'),['type'])
    if row_type and any(str(row.get(row_type,'')).strip().casefold()=='phase' for row in rows):
        rows=[row for row in rows if str(row.get(row_type,'')).strip().casefold()=='phase']
    if not rows: raise ValueError('CSV has no phase annotation rows')
    if not label:
        if len(hdr)==2: label=hdr[-1]
        else:raise ValueError(f'Cannot identify label column from {hdr}')
    sc=_column(hdr,pp.get('start_column'),START_COLS)
    ec=_column(hdr,pp.get('end_column'),END_COLS)
    tc=_column(hdr,pp.get('time_column'),TIME_COLS)
    if not tc and len(hdr)==2: tc=hdr[0]
    ans={};offset=float(pp.get('annotation_offset_seconds',0))
    if sc and ec:
        for row in rows:
            a=int(math.ceil(_time(row[sc],pp['time_unit'],pp['annotation_fps'])+offset))
            b=int(math.ceil(_time(row[ec],pp['time_unit'],pp['annotation_fps'])+offset))
            for sec in range(max(0,a),min(n_seconds,b)):
                if sec in ans and ans[sec]!=row[label]:raise ValueError(f'Overlapping labels at {sec}s')
                ans[sec]=row[label]
    elif tc:
        pts=sorted((int(round(_time(row[tc], 'frame' if 'frame' in tc.lower() else pp['time_unit'],pp['annotation_fps'])+offset)),str(row[label]).strip()) for row in rows)
        if pp.get('frame_label_default','missing')=='forward_fill':
            for i,(start,v) in enumerate(pts):
                stop=pts[i+1][0] if i+1<len(pts) else n_seconds
                for sec in range(max(0,start),min(n_seconds,stop)):ans[sec]=v
        else:
            for sec,v in pts:ans[sec]=v
    else: raise ValueError('No time/start/end column')
    return ans

def pair_sources(cfg):
    """Case-insensitive filename stem pairing; report all unmatched inputs."""
    p=cfg['paths']; pp=cfg['preprocess']; vd=Path(p['video_dir']); cd=Path(p['csv_dir'])
    if not vd.is_dir(): raise FileNotFoundError(f'video_dir does not exist: {vd}')
    if not cd.is_dir(): raise FileNotFoundError(f'csv_dir does not exist: {cd}')
    videos={}; csvs={}
    for x in sorted(vd.iterdir()):
        if x.is_file() and x.suffix.casefold() in VIDEO_EXT:
            key=x.stem.casefold()
            if key in videos: raise ValueError(f'Duplicate video stem: {videos[key]}, {x}')
            videos[key]=x
    for x in sorted(cd.iterdir()):
        if x.is_file() and x.suffix.casefold()=='.csv':
            key=x.stem.casefold()
            if key in csvs: raise ValueError(f'Duplicate CSV stem: {csvs[key]}, {x}')
            csvs[key]=x
    all_ids=sorted(set(videos)|set(csvs)); pairs=[]; skipped=[]
    for key in all_ids:
        if key not in videos or key not in csvs:
            present=videos.get(key) or csvs.get(key)
            row={'video':present.stem, 'reason':'missing_mp4' if key not in videos else 'missing_csv', 'detail':str(present)}
            skipped.append(row)
            if pp['pair_policy']=='strict':raise FileNotFoundError(str(row))
            continue
        pairs.append((videos[key].stem,videos[key],csvs[key]))
    return pairs,skipped

def _video_seconds(video):
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError(f'Cannot open video: {video}')
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not math.isfinite(fps) or fps<=0 or total<=0:
        raise ValueError(f'Bad video metadata: FPS={fps}, frames={total}, path={video}')
    return int(math.ceil(total/fps))

def _extract_frames(video, out, resize, fps_target, reuse, index_start):
    """Decode sequentially, selecting exact source-frame indices for each integer second."""
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('unreadable video')
    fps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or total<=0:cap.release();raise ValueError('invalid FPS/frame count')
    duration=total/fps; n_seconds=int(math.ceil(duration))
    out.mkdir(parents=True,exist_ok=True)
    source={'source':str(video.resolve()),'size':video.stat().st_size,'mtime_ns':video.stat().st_mtime_ns,
            'fps_target':fps_target,'resize':resize,'index_start':index_start}
    meta_path=out/'_source.json'
    if meta_path.exists():
        try:
            old=json.loads(meta_path.read_text())
            if old!=source:reuse=False
        except (OSError,ValueError):reuse=False
    frames=[]
    for sec in range(n_seconds):
        src_idx=int(round(sec*fps))
        if src_idx>=total:break
        name=out/f'{sec+index_start:06d}.jpg'
        if reuse and name.is_file() and cv2.imread(str(name)) is not None:
            frames.append((sec,name));continue
        # Random-access seek to target index so reused frames don't alter alignment.
        cap.set(cv2.CAP_PROP_POS_FRAMES,src_idx)
        ok,frame=cap.read()
        if not ok: continue
        if resize:
            frame=cv2.resize(frame,(int(resize[0]),int(resize[1])),interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(name),frame):raise IOError(f'Cannot write {name}')
        frames.append((sec,name))
    cap.release()
    if not frames:raise ValueError('no extracted frames')
    meta_path.write_text(json.dumps(source,ensure_ascii=False,indent=2))
    return frames,n_seconds


def _extract_external_ffmpeg(video, out, resize, index_start=0, reuse=True, min_coverage=0.80):
    """Extract unlabeled-video frames in one FFmpeg decode pass, preserving integer-second PTS.

    Do not use OpenCV CAP_PROP_POS_FRAMES seek for every second: some long H.264
    streams are not reliably random-seekable.  The original MP4 is never modified.
    All stderr (including recoverable NAL warnings) is stored as _ffmpeg_decode.log.
    """
    video=Path(video);out=Path(out)
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg not installed; install apt-get install ffmpeg')
    if resize is None:
        raise ValueError('External LoViT inference requires preprocess.resize [224,224]')
    min_coverage=float(min_coverage)
    if not 0 <= min_coverage <= 1:
        raise ValueError('external_min_coverage must be in [0,1]')
    source={
        'source':str(video.resolve()),'size':video.stat().st_size,
        'mtime_ns':video.stat().st_mtime_ns,'fps_target':1,
        'resize':list(map(int,resize)),'index_start':int(index_start),
        'decoder':'ffmpeg-fps1-frame_pts-v1'
    }
    meta=out/'_source.json'
    if reuse and meta.exists():
        try:
            if json.loads(meta.read_text(encoding='utf-8'))==source:
                files=sorted(out.glob('*.jpg'))
                if files and all(cv2.imread(str(fp)) is not None for fp in files):
                    return [(int(fp.stem)-index_start,fp) for fp in files], {'reused':True,'frames':len(files),'log':str(out/'_ffmpeg_decode.log')}
        except (ValueError,OSError):
            pass
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix=out.name+'_extract_',dir=out.parent))
    try:
        scale=f'scale={int(resize[0])}:{int(resize[1])}:flags=area'
        # fps=1 assigns a 1-second PTS time base; frame_pts names files by true
        # output timestamps (second) rather than renumbering after missing frames.
        vf=f'fps=1:round=down,{scale}'
        cmd=[ffmpeg,'-hide_banner','-nostdin','-loglevel','warning',
             '-fflags','+discardcorrupt','-err_detect','ignore_err',
             '-i',str(video),'-map','0:v:0','-an','-sn',
             '-vf',vf,'-frame_pts','1','-q:v','3',
             '-f','image2',str(tmp/'%06d.jpg')]
        with (tmp/'_ffmpeg_decode.log').open('w',encoding='utf-8',errors='replace') as log:
            proc=subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=log,check=False)
        images=sorted(tmp.glob('*.jpg'))
        # Capture only images that are readable, and keep absolute second indices.
        valid=[]
        for fp in images:
            try: sec=int(fp.stem)
            except ValueError:continue
            if sec<0 or cv2.imread(str(fp)) is None:continue
            valid.append((sec,fp))
        if not valid:
            raise RuntimeError(f'FFmpeg could not decode any usable frames (exit={proc.returncode}); log: {tmp / "_ffmpeg_decode.log"}')
        if len(valid)!=len(images):
            print(f'[External decoder] {video.name}: ignored {len(images)-len(valid)} unreadable/invalid frames',flush=True)
        # Estimate coverage from stream metadata if reliable; do not rely on FPS
        # to create timestamps because FFmpeg has already done so from PTS.
        expected=None
        try:
            expected=_video_seconds(video)
        except (ValueError,cv2.error):
            pass
        actual_secs={sec for sec,_ in valid}
        coverage=(len(actual_secs)/expected) if expected else None
        if coverage is not None and coverage<min_coverage:
            raise RuntimeError(f'Only {len(actual_secs)}/{expected} seconds decoded ({coverage:.1%}); '
                               f'required {min_coverage:.0%}. Input may be truncated/corrupt; see FFmpeg log')
        # Report hidden recoverable decoder errors; they are not an exception if
        # sufficient timestamp-aligned frames can be obtained.
        error_log=(tmp/'_ffmpeg_decode.log').read_text(encoding='utf-8',errors='replace')
        had_warnings=bool(error_log.strip())
        if had_warnings:
            print(f'[External decoder] {video.name}: FFmpeg emitted warnings; saved to {out / "_ffmpeg_decode.log"}',flush=True)
        if proc.returncode!=0:
            print(f'[External decoder] {video.name}: FFmpeg exited {proc.returncode}; recovered {len(valid)} frames',flush=True)
        for sec,fp in sorted(valid,key=lambda x:x[0],reverse=index_start>0):
            dest=tmp/f'{sec+index_start:06d}.jpg'
            if fp!=dest: fp.rename(dest)
        info={'decoder':'ffmpeg','ffmpeg_exit_code':proc.returncode,
              'reused':False,'frames':len(valid),'duration_seconds':expected,
              'coverage':coverage,'warnings':had_warnings,'log':str(out/'_ffmpeg_decode.log')}
        (tmp/'_source.json').write_text(json.dumps(source,ensure_ascii=False,indent=2),encoding='utf-8')
        (tmp/'_decode_report.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        # Replace the entire frame directory only after decoding succeeded,
        # preventing stale images from contaminating an updated video.
        if out.exists():shutil.rmtree(out)
        tmp.rename(out)
        return [(sec,out/f'{sec+index_start:06d}.jpg') for sec,_ in valid],info
    except Exception as exc:
        # Preserve the FFmpeg diagnostics even if this video must be skipped.
        out.mkdir(parents=True,exist_ok=True)
        logfile=tmp/'_ffmpeg_decode.log'
        if logfile.exists():
            shutil.copy2(logfile,out/'_ffmpeg_decode_failed.log')
        raise RuntimeError(f'{video.name}: external FFmpeg extraction failed: {exc}. '
                           f'Log: {out / "_ffmpeg_decode_failed.log"}') from exc
    finally:
        if tmp.exists():shutil.rmtree(tmp)


def prepare(cfg):
    """Validate annotations before costly image extraction, always write a skip report."""
    label=LabelBook(cfg['paths']['labels']); pp=cfg['preprocess']; paths=cfg['paths']
    out=Path(paths['output_dir'])/'prepared'; out.mkdir(parents=True,exist_ok=True)
    if pp['fps']!=1:
        raise ValueError('This pipeline requires preprocess.fps=1 for annotation alignment')
    pairs,skipped=pair_sources(cfg)
    all_rows=[]; done=[]
    for vid,mp4,annotation in pairs:
        try:
            # First validate annotation columns, time intervals and label values.
            # The old code extracted thousands of frames before detecting an invalid CSV.
            n_seconds=_video_seconds(mp4)
            ann=load_annotation(annotation,pp,n_seconds)
            if not ann: raise ValueError('no integer-second annotations in video duration')
            valid={}
            for sec,raw in ann.items():
                if not 0 <= sec < n_seconds:continue
                mapped=label.map(raw)
                if mapped is None:
                    if label.unknown=='skip_video':
                        raise ValueError(f'Unknown annotation {raw!r} at {sec}s: add to configs/labels.yaml')
                    continue
                valid[sec]=mapped
            if not valid:raise ValueError('No usable labels after annotation mapping; inspect labels.yaml')
            frames,actual_n=_extract_frames(mp4,Path(paths['frame_dir'])/vid,pp.get('resize'),pp['fps'],pp['reuse_frames'],pp.get('frame_index_start',0))
            if n_seconds!=actual_n: raise ValueError(f'Video duration changed during processing: {n_seconds} vs {actual_n}')
            rows=[]
            for sec,fp in frames:
                if sec in valid:
                    rows.append({'video':vid,'sec':sec,'image':str(fp),**valid[sec]})
            if not rows:raise ValueError('No usable paired frames (video decoding or annotation alignment failed)')
            all_rows.extend(rows)
            done.append({'video':vid,'frames':len(rows),'duration_seconds':n_seconds,
                         'video_path':str(mp4),'annotation_path':str(annotation)})
        except Exception as e:
            if pp['bad_video_policy']=='strict':raise
            skipped.append({'video':vid,'reason':type(e).__name__, 'detail':str(e),
                            'video_path':str(mp4), 'annotation_path':str(annotation)})

    # Do this even when nothing succeeded: otherwise the promised skipped.csv never exists.
    write_csv(out/'skipped.csv',skipped,fields=['video','reason','detail','video_path','annotation_path'])
    summary={'paired':len(pairs),'successful_videos':len(done),'frame_rows':len(all_rows),
             'skipped_count':len(skipped), 'skipped_reasons':dict(Counter(x['reason'] for x in skipped)),
             'video_dir':str(paths['video_dir']), 'csv_dir':str(paths['csv_dir']),
             'skipped_report':str(out/'skipped.csv')}
    json_write(out/'preprocess_report.json',summary)
    if not all_rows:
        examples=[f"{x['video']}: {x['reason']} ({x.get('detail','')})" for x in skipped[:5]]
        details='; '.join(examples) if examples else 'No CSV or supported video files found at specified paths'
        raise RuntimeError('No usable paired videos. '
                           f'Found {len(pairs)} pairs; {len(skipped)} skipped. '
                           f'Report: {out / "skipped.csv"}. First failures: {details}')
    index=out/'index.csv';write_csv(index,all_rows)
    manifest={'labels_signature':label.signature,'preprocess_signature':sha(pp),'fps':pp['fps'],
              'videos':done,'skipped':skipped,'index':str(index)}
    json_write(out/'manifest.json',manifest)
    return index
