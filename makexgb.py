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
file='xgb%d.dat'%level
base='xgb' #for colname

import xgboost as xgb,sys
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

xgbparams ={'silent': 1, 'objective': 'binary:logistic',
            'eta': 0.02,'eval_metric':'logloss',
            'booster': 'gbtree'}
discreteP=['max_depth','ncols']

erastarts=np.where(extra_y['era']!=extra_y['era'].shift())[0].tolist()
erastarts=erastarts[0::len(erastarts)/8]
erastarts_=erastarts+[train.shape[0]]
folds=[(i,range(erastarts[i],erastarts_[i+1])) for i in range(len(erastarts))]
      
trainnext,testnext = nextlevel(level)
# trainnext_leak,testnext_leak = nextlevel(nextleakfile)
for i in random.sample(range(1,args.n+1),args.n):
    r=featimp.iloc[-i]
    params = p[r.name]
    chosen_feat = list(r.dropna().index)
    colname = base+'_'+str(round(abs(r.name),scoredp))
    if colname in testnext.columns:
      if any(colname+'_'+str(j) in testnext.columns.tolist()+trainnext.columns.tolist() for j in range(nbags)):
        nextlevel(level,delcols=[colname+'_'+str(j) for j in range(nbags)])
      continue

    for k in params.keys():
      if k in discreteP: params[k]=int(round(params[k]))
    params['colsample_bytree']=max(2./len(chosen_feat),params['colsample_bytree'])

    print r.name,params
    if any(colname+'_'+str(j) not in trainnext for j in range(nbags)):
      train_x,test_x = generate_train_x(chosen_feat,train,extra_y[target],test,returntest=True)
      train_y = extra_y[target]

    for j in random.sample(range(nbags),nbags):
      colname_sub=colname+'_'+str(j)
      if (colname_sub in trainnext or colname in trainnext) and\
         (colname_sub in testnext or colname in testnext) and\
         ((not args.second) or any(testnext[colname_sub])):
          continue
      
      testcolnext =0
      traincolnext =0*train_y
      xgbp=xgbparams.copy()
      xgbp.update(params)
      xgbp.update({'seed':j+100})

      d_tr = xgb.DMatrix(train_x, label=train_y)
      cv = xgb.cv(xgbp,d_tr,folds=folds,
                  num_boost_round=100000,early_stopping_rounds=patience,
                  verbose_eval=args.verbose and 50, show_stdv=False)

      d_test = xgb.DMatrix(test_x)
      for vali in range(len(folds)):
        tra_idx=[i for fold in folds for i in fold[1] if fold[0]!=vali]
        val_idx=folds[vali][1]
        X_tra,y_tra = train_x.iloc[tra_idx], train_y.iloc[tra_idx]
        X_val,y_val = train_x.iloc[val_idx], train_y.iloc[val_idx]
        d_tra = xgb.DMatrix(X_tra, label=y_tra)
        d_val = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(xgbp,d_tra,
                          num_boost_round=cv.shape[0],
                          evals=[(d_val, 'eval')],
                          verbose_eval=args.verbose and 50)
        traincolnext.iloc[val_idx]=model.predict(d_val)
        print vali,log_loss(y_val,traincolnext.iloc[val_idx])
        testcolnext+=model.predict(d_test)

      testcolnext/=len(folds)
      print j,log_loss(train_y,traincolnext)

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
