# with early-stopping, testthis data is averaged. trainthis n_estimators is average of those found in generating testthis data
import argparse
from common import *
set_path()
parser = argparse.ArgumentParser()
parser.add_argument('-np',  type=int,  default=-1)
parser.add_argument('-n',   type=int,  default=3)
parser.add_argument('-nbags',   type=int,  default=10)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--verbose", help="increase output verbosity",action="store_true")
parser.add_argument("--second", action="store_true")
parser.add_argument('-level',  type=int,  default=level)
args = parser.parse_args()
level=args.level
nextleakfile='%d_leak'%(level+1)
cols0,train,extra_y,test=cols_train_y_test(level,loadtest=True,coltype='ext')
test1=pd.read_pickle('test1')
data_type=test1.data_type
trainN=train.shape[0]
train=pd.concat((train,test[data_type=='validation']))#############################
test=test[data_type!='validation']#############################
extra_y=pd.concat((extra_y,test1[data_type=='validation'][extra_y.columns]))#############################
file='lgb%d.dat'%level
base='lgb' #for colname

import lightgbm as lgb,sys
patience=100  #for early stopping
args = parser.parse_args()
nthread = args.np
nbags=args.nbags
maxepoch=1000 if not args.debug else 5

while True:
  try:
    with open(file,'rb') as infile: (featimp,p,n_rounds_without_improve,n_wait)=pickle.load(infile)
    break
  except: time.sleep(5)

lgbparams ={'learning_rate': 0.002,'feature_fraction_seed':random.randint(0,100),
            'objective': 'binary','metric': 'binary_logloss','silent': 1}
discreteP =['ncols']

erastarts=np.where(extra_y['era']!=extra_y['era'].shift())[0]
erastarts=erastarts[0::len(erastarts)/12]
nfold=4
folds=[(range(erastarts[i],erastarts[-nfold+i]),range(erastarts[-nfold+i],erastarts[1-nfold+i] if 1-nfold+i<0 else train.shape[0])) for i in range(nfold)]

trainnext,testnext = nextlevel(level)
# trainnext_leak,testnext_leak = nextlevel(nextleakfile)
for i in random.sample(range(1,args.n+1),args.n):
    r=featimp.iloc[-i]
    params = p[r.name]
    chosen = list(r.dropna().index)
    colname = base+'_'+str(round(abs(r.name),scoredp))
    if colname in testnext:
      if any(colname+'_'+str(j) in testnext.columns.tolist()+trainnext.columns.tolist() for j in range(nbags)):
        nextlevel(level,delcols=[colname+'_'+str(j) for j in range(nbags)])
      continue

    for k in params.keys():
      if k in discreteP: params[k]=int(round(params[k]))
    params['sub_feature']=max(5./len(chosen),params['sub_feature'])
    params['num_leaves']=int(2**params['lnl'])
    params['min_data']=int(2**params['lmd'])
    params['max_bin']=int(2**params['lmb'])

    print r.name,params
    if any(colname+'_'+str(j) not in trainnext.columns for j in range(nbags)):
      train_x,test_x = generate_train_x(chosen,train,extra_y[target],test,returntest=True)
      train_y = extra_y[target]

    for j in random.sample(range(nbags),nbags):
      colname_sub=colname+'_'+str(j)
      if (colname_sub in trainnext or colname in trainnext) and\
         (colname_sub in testnext or colname in testnext) and\
         ((not args.second) or any(testnext[colname_sub])):
          continue
      
      testcolnext =0
      traincolnext =0*train_y
      lgbp=lgbparams.copy()
      lgbp.update(params)
      lgbp.update({'feature_fraction_seed':j+100})

      d_tra = lgb.Dataset(train_x, label=train_y, silent=True)
      cv = lgb.cv(lgbp,d_tra,folds=folds,
                  num_boost_round=100000,early_stopping_rounds=patience,
                  verbose_eval=args.verbose and 50)
      cv = pd.DataFrame.from_dict(cv)

      for i in range(nfold+1):
        tra_start=erastarts[i]
        val_start=erastarts[-nfold+i] if -nfold+i<0 else train_x.shape[0]
        val_end=erastarts[1-nfold+i] if 1-nfold+i<0 else train_x.shape[0]
        X_tra, y_tra = train_x.iloc[tra_start:val_start].values, train_y.iloc[tra_start:val_start].values
        X_val, y_val = train_x.iloc[val_start:val_end].values,   train_y.iloc[val_start:val_end].values
        d_tra = lgb.Dataset(X_tra, label=y_tra, silent=True)
        d_val = lgb.Dataset(X_val, label=y_val, silent=True)
        model = lgb.train(lgbp,d_tra,
                          num_boost_round=cv.shape[0],
                          verbose_eval=args.verbose and 50)
        if X_val.shape[0]:
          traincolnext.iloc[val_start:val_end]=model.predict(X_val)
          print i,log_loss(y_val,traincolnext.iloc[val_start:val_end])
        testcolnext+=model.predict(test_x)

      testcolnext/=nfold+1
      print log_loss(train_y.iloc[erastarts[-nfold]:],traincolnext.iloc[erastarts[-nfold]:])

      testval=traincolnext[trainN:]
      traincolnext=traincolnext[:trainN]
      testcolnext=np.concatenate((testval,testcolnext))
      # trainnext_leak,testnext_leak=nextlevel(nextleakfile,traincolnext,testcolnext,colname_sub)
      trainnext,testnext=nextlevel(level,traincolnext,testcolnext,colname_sub)

    bagcols = [colname+'_'+str(j) for j in xrange(nbags)]
    if all([e in trainnext.columns for e in bagcols]):
      if trainnext.ix[:,bagcols].mean().min()>0:
        from scipy.stats.mstats import gmean
        gmeanscore=log_loss(extra_y[target][:trainN],gmean(trainnext.ix[:,bagcols],1))
        ameanscore=log_loss(extra_y[target][:trainN],trainnext.ix[:,bagcols].mean(1))
        print gmeanscore,ameanscore
        if gmeanscore<ameanscore:
          # traincolnext_leak=gmean(trainnext_leak.ix[:,bagcols],1)
          # testcolnext_leak=gmean(testnext_leak.ix[:,bagcols],1)
          traincolnext=gmean(trainnext.ix[:,bagcols],1)
          testcolnext=gmean(testnext.ix[:,bagcols],1)
        else:
          # traincolnext_leak=trainnext_leak.ix[:,bagcols].mean(1)
          # testcolnext_leak=testnext_leak.ix[:,bagcols].mean(1)
          traincolnext=trainnext.ix[:,bagcols].mean(1)
          testcolnext=testnext.ix[:,bagcols].mean(1)
        # nextlevel(nextleakfile,traincolnext_leak,testcolnext_leak,colname,delcols=bagcols)
        nextlevel(level,traincolnext,testcolnext,colname,delcols=bagcols)
        os.execv(sys.executable,[sys.executable]+sys.argv)

if not args.second: os.execv(sys.executable,[sys.executable]+sys.argv+['--second'])
