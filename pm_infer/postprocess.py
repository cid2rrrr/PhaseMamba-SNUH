import numpy as np

def transition_prior(index,videos,num_classes,laplace=1.):
    # Only adjacent observed seconds and valid phase labels are counted.
    counts=np.full((num_classes,num_classes),float(laplace),dtype=np.float64)
    for video in videos:
        rows=index[video]
        for a,b in zip(rows[:-1],rows[1:]):
            if b['sec']==a['sec']+1 and a['phase']>=0 and b['phase']>=0:
                counts[a['phase'],b['phase']]+=1
    return counts/counts.sum(1,keepdims=True)

def refine(probs,prior,confidence=.7,alpha=.6):
    """Conditional CAUSAL PhaseMamba Eq. 8-9, no retrospective Viterbi."""
    probs=np.asarray(probs);out=[];prev=None
    for p in probs:
        raw=int(p.argmax());j=raw
        if prev is not None and raw!=prev and p.max()<confidence:
            j=int((np.log(np.maximum(p,1e-10))+alpha*np.log(np.maximum(prior[prev],1e-10))).argmax())
        out.append(j);prev=j
    return np.asarray(out,dtype=np.int64)

def egtp(probs,threshold):
    try:from stablespr import EvidenceGatedTransitionPredictor
    except ImportError as e:raise ImportError('EGTP requested: install official MRUIL/StableSPR (pip install -e third_party/StableSPR)') from e
    model=EvidenceGatedTransitionPredictor(threshold=threshold)
    return np.asarray(model.predict(np.asarray(probs)),dtype=np.int64)

def tfi(pred,gt):
    try:from stablespr import temporal_fragmentation_index
    except ImportError:return None
    return float(temporal_fragmentation_index(pred,gt))
