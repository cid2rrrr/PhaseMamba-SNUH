"""CPU mock tests shape/checkpoint/CSV logic only. This is NOT a GPU/Mamba test."""
import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

# A stable fake to exercise our surrounding inference code without pretending Mamba is tested.
class FakeMamba(nn.Module):
    def __init__(self,d_model,d_state,d_conv,expand):
        super().__init__()
        self.scale=nn.Parameter(torch.ones(d_model))
    def forward(self,x):return x*self.scale

sys.modules['mamba_ssm']=types.SimpleNamespace(Mamba=FakeMamba)
from pm_infer.models import Stage1,Stage2
from pm_infer.common import LabelBook
from pm_infer.run import read_weights,local_features,infer_one,load_config

class TestPipeline(unittest.TestCase):
    def test_checkpoint_activity_and_export(self):
        base=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);run=out/'full'; run.mkdir()
            book=LabelBook(base/'configs'/'labels.yaml')
            cfg={'model':{'feature_dim':768,'hidden_dim':16,'window':4,'d_state':4,'stage1_blocks':2,'stage2_blocks':3},
                 'preprocess':{'resize':[224,224],'fps':1},
                 'train':{'skip_stage0':True},
                 'postprocess':{'refinement':False,'refine_confidence':0.7,'refine_alpha':0.6,'activity_gate':True,'egtp':False},
                 'report':{'overlay_mp4':False}}
            (run/'run_config.json').write_text(json.dumps({'config':cfg,'labels_signature':book.signature}))
            assert load_config(run)[1]==cfg
            s1=Stage1(768,16,book.n,2,4);s2=Stage2(768,16,book.n,3,4)
            assert s2.activity_head.out_features==2
            for name,model in [('stage1',s1),('stage2',s2)]:
                folder=run/name;folder.mkdir()
                torch.save({'model':model.state_dict(),'labels_signature':book.signature},folder/'last.pt')
            s1=read_weights(run/'stage1/last.pt',Stage1(768,16,book.n,2,4),book,'cpu')
            s2=read_weights(run/'stage2/last.pt',Stage2(768,16,book.n,3,4),book,'cpu')
            x=torch.randn(6,768)
            loc=local_features(x,s1,4,torch.device('cpu'),2)
            self.assertEqual(tuple(loc.shape),(6,16))
            f=out/'video.pt';torch.save({'features':x,'sec':torch.arange(6)},f)
            args=types.SimpleNamespace(output=out/'preds',features_pt=True,weights='last',local_batch_size=2)
            infer_one(run,cfg,book,s1,s2,None,None,torch.device('cpu'),args,f)
            p=out/'preds/video/predictions.csv'
            self.assertTrue(p.exists())
            with p.open() as fh:rows=list(csv.DictReader(fh))
            self.assertEqual(len(rows),6)
            for row in rows:
                self.assertIn(row['activity'],('0','1'))
                self.assertAlmostEqual(float(row['p_active'])+float(row['p_inactive']),1.0,places=4)
            obj=np.load(out/'preds/video/probabilities.npz')
            self.assertEqual(tuple(obj['phase'].shape),(6,book.n))
            self.assertEqual(tuple(obj['activity'].shape),(6,2))
    def test_model_rejects_invalid_state(self):
        base=Path(__file__).resolve().parents[1]
        book=LabelBook(base/'configs'/'labels.yaml')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'wrong.pt'
            model=Stage2(768,16,book.n,3,4)
            state=model.state_dict();del state['activity_head.weight']
            torch.save({'model':state,'labels_signature':book.signature},p)
            with self.assertRaises(RuntimeError):
                read_weights(p,model,book,'cpu')

if __name__=='__main__':unittest.main()

class TestSyntheticVideo(unittest.TestCase):
    def test_video_decode_and_original_overlay(self):
        import shutil
        import subprocess
        if not shutil.which('ffmpeg'):self.skipTest('No FFmpeg')
        from pm_infer.preprocess import _extract_external_ffmpeg
        from pm_infer.visualize import export_predictions
        base=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            video=root/'sample.mp4'
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi',
                            '-i','testsrc=size=320x240:rate=5:duration=3',
                            '-pix_fmt','yuv420p','-c:v','mpeg4',str(video)],check=True)
            frames,info=_extract_external_ffmpeg(video,root/'frames',[224,224],0,False,.8)
            self.assertEqual(len(frames),3)
            self.assertEqual([i for i,_ in frames],[0,1,2])
            book=LabelBook(base/'configs'/'labels.yaml')
            probs=np.zeros((3,book.n),dtype='float64');probs[:,0]=1
            activities=np.array([[.1,.9],[.7,.3],[.1,.9]],dtype='float64')
            cfg={'postprocess':{'refinement':False,'egtp':False,'activity_gate':True},
                 'report':{'overlay_mp4':True,'overlay_fps':1,'overlay_speed':2.0,'overlay_resolution':'original'}}
            rows=[{'video':'sample','sec':sec,'phase':-100,'activity':-100,'image':str(img)} for sec,img in frames]
            res=export_predictions(root/'render',rows,probs,activities,book,cfg,prior=None,source_video=video)
            self.assertEqual([x['final'] for x in res],[0,-1,0])
            self.assertTrue((root/'render/overlay.mp4').exists())
            self.assertGreater((root/'render/overlay.mp4').stat().st_size,1000)
