#!/usr/bin/env python3
"""改进版枕头推荐算法 — KNN多维度加权 + 舒适度加权 + ✓/✗匹配评估"""

import json, numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent

def load_data():
    data_dir = Path(__file__).parent / 'pwa' / 'data'
    with open(data_dir / 'users.json') as f:   users = json.load(f)
    with open(data_dir / 'pillows.json') as f: pillows = json.load(f)
    with open(data_dir / 'records.json') as f: records = json.load(f)
    return users, pillows, records

def parse_neck_curve(s):
    if isinstance(s, list): return np.array(s, dtype=float)
    return np.array(json.loads(s), dtype=float)

def to_num(v): return 0.0 if v == '' or v is None else float(v)

FEATURE_KEYS = ['gender','age','height','weight','bmi','shoulder_width',
    'neck_max','neck_max_pos','neck_avg','neck_sum','neck_std',
    'neck_front_sum','neck_mid_sum','neck_back_sum','neck_max_diff','neck_rise']

def extract_neck_features(curve):
    c = np.array(curve, dtype=float)
    return {
        'neck_max': float(np.max(c)), 'neck_max_pos': float(np.argmax(c)),
        'neck_avg': float(np.mean(c)), 'neck_sum': float(np.sum(c)),
        'neck_std': float(np.std(c)),
        'neck_front_sum': float(np.sum(c[:7])), 'neck_mid_sum': float(np.sum(c[7:14])),
        'neck_back_sum': float(np.sum(c[14:])),
        'neck_max_diff': float(np.max(np.abs(np.diff(c)))) if len(c)>1 else 0,
        'neck_rise': float(np.max(np.diff(c))) if len(c)>1 else 0}

def user_to_vector(user):
    nf = extract_neck_features(parse_neck_curve(user['neck_curve']))
    d = {'gender': 1.0 if user.get('gender')=='男' else 0.0,
        'age': float(user['age']), 'height': float(user['height']),
        'weight': float(user['weight']),
        'bmi': float(user['weight'])/(float(user['height'])/100)**2,
        'shoulder_width': float(user['shoulder_width']), **nf}
    return np.array([d[k] for k in FEATURE_KEYS], dtype=float)

def cosine_sim(a,b):
    d=np.dot(a,b); return d/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8)

def minmax_norm(X):
    mn,mx=X.min(axis=0),X.max(axis=0); d=mx-mn; d[d==0]=1
    return (X-mn)/d, mn, mx

def softmax(arr, T=0.15):
    a=np.array(arr); a-=a.max(); e=np.exp(a/T); return e/(e.sum()+1e-6)

def comfort_score(r):
    return (r['overall']*0.45+r['supine']*0.25+r['side']*0.25+r['neck']*0.03+r['head']*0.02)

def weighted_ideal(recs, key):
    vals = np.array([to_num(r.get(key,0)) for r in recs if to_num(r.get(key,0))>0], dtype=float)
    scores = np.array([r['_cs'] for r in recs if to_num(r.get(key,0))>0], dtype=float)
    if len(vals)==0: return None
    s=scores-scores.min()+1; w=np.exp(s/(max(1,s.std())+1e-6)); w/=w.sum()
    return float(np.dot(w, vals))

class ImprovedRecommender:
    def __init__(self):
        users, pillows, records = load_data()
        self.users, self.pillows, self.records = users, pillows, records
        self.ids = [u['id'] for u in users]
        self.feats = np.array([user_to_vector(u) for u in users], dtype=float)
        self.Xn, self.Xmin, self.Xmax = minmax_norm(self.feats)
        self.ur = defaultdict(list)
        for r in records:
            r['_cs'] = comfort_score({'overall':to_num(r['comfort']['comfort_overall']),
                'supine':to_num(r['comfort']['comfort_supine']),'side':to_num(r['comfort']['comfort_side']),
                'neck':to_num(r['comfort']['comfort_neck']),'head':to_num(r['comfort']['comfort_head'])})
            self.ur[r['user_id']].append(r)
        self._compute_ideal()

    def _compute_ideal(self):
        self.ideal = {}
        for uid in self.ids:
            recs = self.ur.get(uid,[]); s=[r for r in recs if r.get('sleep_pos')=='仰睡']
            sid=[r for r in recs if r.get('sleep_pos')=='侧睡']
            ih=weighted_ideal(s, 'pillow_head_height'); inn=weighted_ideal(s, 'pillow_neck_height')
            ish=weighted_ideal(sid, 'pillow_side_height'); ihd=weighted_ideal(recs, 'pillow_center_hardness')
            if any(v is not None for v in [ih, inn, ish, ihd]):
                self.ideal[uid] = {'h':ih or 0,'n':inn or 0,'s':ish or 0,'ch':ihd or 0}

    def recommend(self, user_info, k=3):
        v = user_to_vector(user_info); rng = self.Xmax-self.Xmin; rng[rng==0]=1
        vn = np.clip((v-self.Xmin)/rng, 0, 1)
        valid = [i for i,uid in enumerate(self.ids) if uid in self.ideal]
        if not valid: return self._empty()
        sims = np.array([cosine_sim(vn, self.Xn[i]) for i in valid])
        top = np.argsort(sims)[::-1][:k]; tids = [self.ids[valid[i]] for i in top]
        ts = sims[top]; w = softmax(ts, 0.15)
        hv,nv,sv,dv,ws=[],[],[],[]
        for uid,wt in zip(tids, w):
            ide = self.ideal.get(uid,{})
            for arr,k in [(hv,'h'),(nv,'n'),(sv,'s'),(dv,'ch')]:
                if ide.get(k) is not None: arr.append(ide[k]); ws.append(wt)
        def wa(a): a=np.array(a); ww=np.array(ws[:len(a)]); return float(np.dot(a,ww/ww.sum())) if len(a) else 0
        return {'head_height':round(wa(hv),1),'neck_height':round(wa(nv),1),
            'side_height':round(wa(sv),1),'hardness':round(wa(dv),1),
            'similar_users':tids,'similarities':[float(s) for s in ts],
            'matched_pillows':self._match(wa(hv),wa(nv),wa(sv),wa(dv))}

    def _empty(self): return {'head_height':0,'neck_height':0,'side_height':0,'hardness':0,
        'similar_users':[],'similarities':[],'matched_pillows':[]}

    def _match(self, hh, nh, sh, hd):
        tol = {'h':10, 'n':10, 's':10, 'd':3}
        scored = []
        for p in self.pillows:
            ph=to_num(p.get('head_height',0)); pn=to_num(p.get('neck_height',0))
            ps=to_num(p.get('side_height',0)); pd=to_num(p.get('center_hardness',0))
            dh,dn,ds,dd = abs(ph-hh),abs(pn-nh),abs(ps-sh),abs(pd-hd)
            dist = np.sqrt(dh**2+dn**2+ds**2+(dd*3)**2)
            checks = {'head':'✓' if dh<=tol['h'] else '✗','neck':'✓' if dn<=tol['n'] else '✗',
                'side':'✓' if ds<=tol['s'] else '✗','hard':'✓' if dd<=tol['d'] else '✗'}
            mpct = sum(1 for v in checks.values() if v=='✓')/4*100
            scored.append((dist,p,checks,mpct,dh,dn,ds,dd))
        scored.sort(key=lambda x: x[0])
        return [{'id':p['id'],'brand':p.get('brand','?'),'material':p.get('material','?'),
            'head_h':ph,'neck_h':pn,'side_h':ps,'hardness':pd,
            'match_score':round(max(0,100-dist),1),'param_match':f'{mpct:.0f}%',
            'diffs':{'head':f'{dh:+.0f}','neck':f'{dn:+.0f}','side':f'{ds:+.0f}','hard':f'{dd:+.1f}'}}
            for dist,p,checks,mpct,dh,dn,ds,dd in scored[:3]]

if __name__ == '__main__':
    r = ImprovedRecommender()
    u = {'gender':'男','age':28,'height':175,'weight':70,'shoulder_width':16.5,
        'neck_curve':'[0,0,0,0.2,0.8,1.5,2.2,2.8,3.2,3.4,3.0,2.5,1.8,1.0,0.5,0.1,0,0,0,0,0]'}
    res = r.recommend(u)
    print(f"推荐: 后脑勺{res['head_height']:.0f}mm 颈椎{res['neck_height']:.0f}mm 侧睡{res['side_height']:.0f}mm 硬度{res['hardness']:.1f}")
    for i,p in enumerate(res['matched_pillows']):
        print(f"  {i+1}. {p['brand']}({p['id']}) 匹配{p['param_match']} {p['diffs']}")
