import argparse,random,heapq
from common import *
set_path()
parser = argparse.ArgumentParser()
parser.add_argument('-acq', type=str,  default=random.choice(['poi','ei']))  #ei samples boundary
parser.add_argument("--trunc", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument('-init', type=int, default=1) 
parser.add_argument('-iter', type=int, default=2)
args = parser.parse_args()
level=1
musthave=[]
mustnothave=[]
mustretain=[]
maxncols=300
patience = 20
largestwait=15 #number of largest entries in p to determine for n_rounds and n_wait, which when becomes too large reset the featimp
bestn = 10   #making sure these best rows has all their features, and keep the entry with smallest ncols among them
resetrows=10 #keep this many rows after reset, don't use them for feat selection, but use for BO

picklefile='lgb1.dat'
from bayes_opt import BayesianOptimization
import lightgbm as lgb
cols0,train,extra_y,test=cols_train_y_test(1,loadtest=True,coltype=not os.path.isfile(picklefile) and 'ext')
data_type=test.data_type
train=pd.concat((train,test[data_type=='validation']))#############################
extra_y=pd.concat((extra_y,test[data_type=='validation'][extra_y.columns]))#############################
test=test[data_type!='validation']#############################

if not os.path.isfile(picklefile):
  featimp=pd.DataFrame(columns=cols0)
  with open(picklefile,'wb') as outfile:
    pickle.dump((featimp,{},0,0),outfile, pickle.HIGHEST_PROTOCOL)
  time.sleep(5)
while True:
  try:
    with open(picklefile,'rb') as infile:    (featimp,p,n_rounds_without_improve,n_wait)=pickle.load(infile)
    break
  except: time.sleep(5)
  
lgbparams ={'learning_rate': 0.002,'feature_fraction_seed':random.randint(0,100),
            'objective': 'binary','metric': 'binary_logloss','silent': 1}
discreteP=['ncols']
p_range={
       'sub_feature': (0.1,1),
               'lmb': (4,12),
               'lmd': (12,14.5),
               'lnl': (5,15),
  'bagging_fraction': (.7,1),
             'ncols': (5,min(maxncols,featimp.shape[1])),
        }
      
erastarts=np.where(extra_y['era']!=extra_y['era'].shift())[0]
erastarts=erastarts[0::len(erastarts)/12]
nfold=4
folds=[(range(erastarts[-3-nfold+i],erastarts[-nfold+i]),range(erastarts[-nfold+i],erastarts[1-nfold+i] if 1-nfold+i<0 else train.shape[0])) for i in range(nfold)]
      
def score(**params):
      global featimp
      for k in params.keys():
        params[k]=p_range[k][0]*(1-params[k])+p_range[k][1]*params[k]
        if k in discreteP: params[k]=int(round(params[k]))
      featimpmean=gen_featimpmean(featimp)
      if random.random()<.3 or featimp.shape[0]<5:
        chosen=[]
        while len(chosen)<min(params['ncols'],featimp.shape[1]):
          candfeat=weighted_featimp(featimp.iloc[:-resetrows],chosen).fillna(featimpmean.fillna(1.))
          candfeat=candfeat.fillna(1./featimp.shape[1])
          candfeat=candfeat.replace(0,candfeat[candfeat>0].min())
          theone = np.random.choice(candfeat.index,p=candfeat.values/np.sum(candfeat.values))
          chosen.append( theone )
        chosen=list(set(chosen+musthave))
      else:
        cols={k:[vk for vk,imp in v.iteritems() if imp>0] for k,v in featimp.T.to_dict().iteritems()}
        estimatedfeatimp=featimp_from_cols(cols)
        chosen=list(np.random.choice(estimatedfeatimp.keys(),min(params['ncols'],len(filter(None,estimatedfeatimp.values()))),replace=False,p=estimatedfeatimp.values()))
              
      params['sub_feature']=max(2./len(chosen),params['sub_feature'])
      params['num_leaves']=int(2**params['lnl'])
      params['min_data']=int(2**params['lmd'])
      params['max_bin']=int(2**params['lmb'])
      
      lgbp=lgbparams.copy()
      lgbp.update(params)
      fscores=dict((el,0) for el in chosen)
      if args.verbose: print 'generate_train_x',len(chosen)
      train_x = generate_train_x(chosen,train,extra_y[target],test)
      train_y = extra_y[target]
      if args.verbose: print train_x.shape,lgbp
      if train_x.dropna(1,how='all').shape[1]==0: return np.mean(featimp.index)
      
      d_tr = lgb.Dataset(train_x, label=train_y, silent=True)
      cv = lgb.cv(lgbp,d_tr,folds=folds,
                  num_boost_round=100000,early_stopping_rounds=patience,
                  verbose_eval=args.verbose and 50, show_stdv=False)
      cv = pd.DataFrame.from_dict(cv)
      s = cv.iloc[-1,0] #cv columns: ['test-rmse-mean', 'test-rmse-std', 'train-rmse-mean','train-rmse-std']
      
      model = lgb.train(lgbp,d_tr,num_boost_round=cv.shape[0],verbose_eval=args.verbose and 50) #solely for feature importances
      fi = model.feature_importance(importance_type='gain')
      fscore = {chosen[i]:fi[i] for i in xrange(len(chosen))}
      featimpmean=featimpmean.fillna(1./featimp.shape[1])
      normalization = featimpmean[chosen].sum()/featimpmean.sum()/np.sum(fscore.values())
      for k,v in fscore.iteritems():fscores[k]+=normalization*v
      
      idx=round(1000*(np.log(2)-s),scoredp)
      featimp = featimp.append(pd.Series(fscores,name=idx))
      return idx

while True:
  init_points=args.init
  n_iter=args.iter
  scaledrange={k:(0,1) for k in p_range.keys()}
  bo = BayesianOptimization(score, scaledrange)
  if p: bo.initialize({k:{pk:(pv-p_range[pk][0])/(p_range[pk][1]-p_range[pk][0]) for pk,pv in param.iteritems()} for k,param in p.iteritems()})
  else: init_points,n_iter=5,0
  if not args.trunc:
    bo.maximize(init_points=init_points, n_iter=n_iter, acq=args.acq)
    featimp_cur=featimp
    p_new = {}
    for i in xrange(len(bo.Y)):
      if bo.Y[i] not in bo.y_init:
        p_new[bo.Y[i].round(scoredp)]={bo.keys[j]:p_range[bo.keys[j]][0]*(1-bo.X[i,j])+p_range[bo.keys[j]][1]*bo.X[i,j] for j in xrange(len(bo.keys))}

    if not os.path.isfile(picklefile): break
    with open(picklefile,'rb') as infile:
        try:
          (featimp,p_now,n_rounds_without_improve,n_wait)=pickle.load(infile)
          p.update(p_now)
        except: pass
    if featimp.shape[1]!=featimp_cur.shape[1]: break
    i1=featimp.index
    i2=featimp_cur.index
    oldi1=[i for i in xrange(len(i1)) if i1[i] not in i2]
    featimp=pd.concat((featimp.iloc[oldi1],featimp_cur)).sort_index()

    if p_new and len(p)>largestwait:
      if max(p_new.keys())<=sorted(p.keys())[-largestwait]: 
        # n_nonboundary=sum([all([v2 not in scaledrange[k2] for k2,v2 in v.iteritems()]) for _,v in p_new.iteritems()])
        # if args.acq!="ei":n_rounds_without_improve+=n_nonboundary-init_points
        if args.acq!="ei":n_rounds_without_improve+=n_iter
      else: 
        n_wait,n_rounds_without_improve = max(n_wait,n_rounds_without_improve),0
    p.update(p_new)

  if len(p)>600:  # too large makes bayes opt slow
    p = {k:p[k] for k in heapq.nlargest(600, p)}
  if featimp.shape[0]>1000:  # filesize, time generating chosen
    featimp = featimp.tail(1000)
    
  if args.trunc or (n_rounds_without_improve>100 and featimp.shape[0]>=1000) and 'ncols' in p_range:
    cols = list(featimp.tail(bestn).dropna(1,'all').columns)
    smallest = pd.DataFrame.from_dict(p).T.tail(bestn).ncols.argmin() #keep the entry with smallest ncols
    cols_smallest_ncols = list(featimp.loc[smallest].dropna().index) 
    cols = list(set(cols+mustretain+cols_smallest_ncols))
    if len(cols)!=featimp.shape[1]:
      i=bestn+1
      while len(cols)<featimp.shape[1]*0.8 and i<featimp.shape[0]:
        r = list(featimp.iloc[-i].replace(0,np.nan).dropna().index)
        cols = list(set(r+cols))
        i+=1
        print i,len(cols)
      featimpnew = featimp.tail(resetrows).ix[:,cols]
      if smallest not in featimpnew.index:
        featimpnew.loc[smallest] = featimp.loc[smallest]
      featimp = featimpnew.sort_index()
      p = {k:p[k] for k in heapq.nlargest(resetrows, p)+[smallest]}
      if featimp.shape[1]<maxncols:
        for k in p:
          p[k]['ncols']=float(featimp.loc[k,:].dropna().shape[0]-p_range['ncols'][0])/(featimp.shape[1]-p_range['ncols'][0])
      n_rounds_without_improve = 0
      n_wait = 0

  with open(picklefile,'wb') as outfile:
    pickle.dump((featimp,p,n_rounds_without_improve,n_wait),outfile, pickle.HIGHEST_PROTOCOL)
  print (featimp.shape,len(p),n_rounds_without_improve,n_wait)
  
  if args.trunc: sys.exit()
  else: break
  
# os.execv(sys.executable,[sys.executable]+sys.argv)
os.execv(sys.executable,[sys.executable]+[e.replace('genlgb','genlr') for e in sys.argv])
