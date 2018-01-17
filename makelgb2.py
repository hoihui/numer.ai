# with early-stopping, testthis data is averaged. trainthis n_estimators is average of those found in generating testthis data
import argparse
from common import *
set_path()
parser = argparse.ArgumentParser()
parser.add_argument('-np',  type=int,  default=-1)
parser.add_argument('-n',   type=int,  default=1)
parser.add_argument('-nbags',   type=int,  default=3)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--verbose", help="increase output verbosity",action="store_true")
parser.add_argument("--second", action="store_true")
args = parser.parse_args()
level=2
nextleakfile='%d_leak'%(level+1)
cols0,train,extra_y,test=cols_train_y_test(level,loadtest=True,coltype='ext')
test1=pd.read_pickle('test1')
data_type=test1.data_type
testvalN=(data_type=='validation').sum()
train=pd.concat((train,test[data_type=='validation']))#############################
extra_y=pd.concat((extra_y,test1[data_type=='validation'][extra_y.columns])).tail(train.shape[0])##############################
test=test[data_type!='validation']#############################
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

trainnext,testnext = nextlevel(level)
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
    
    lgbp=lgbparams.copy()
    lgbp.update(params)
    d_trval = lgb.Dataset(train_x, label=train_y, silent=True)
    d_tr = lgb.Dataset(train_x.iloc[:-testvalN], label=train_y.iloc[:-testvalN], silent=True)
    cvtrval = lgb.cv(lgbp,d_trval,nfold=8,
                num_boost_round=100000,early_stopping_rounds=patience,
                verbose_eval=args.verbose and 50)
    cvtrval = pd.DataFrame.from_dict(cvtrval)
    cvtr = lgb.cv(lgbp,d_tr,nfold=8,
                num_boost_round=100000,early_stopping_rounds=patience,
                verbose_eval=args.verbose and 50)
    cvtr = pd.DataFrame.from_dict(cvtrval)

    for j in random.sample(range(nbags),nbags):
      colname_sub=colname+'_'+str(j)
      if (colname_sub in trainnext or colname in trainnext) and\
         (colname_sub in testnext or colname in testnext) and\
         ((not args.second) or any(testnext[colname_sub])):
          continue
      
      testcolnext =0
      traincolnext =0*extra_y[target]
      lgbp.update({'feature_fraction_seed':j+100})

      model = lgb.train(lgbp,d_trval,num_boost_round=cvtrval.shape[0],
                        verbose_eval=args.verbose and 50)
      traincolnext=model.predict(train_x)
      testcolnext=model.predict(test_x)
      model = lgb.train(lgbp,d_tr,num_boost_round=cvtr.shape[0],
                        verbose_eval=args.verbose and 50)
      traincolnext[-testvalN:]=model.predict(train_x.iloc[-testvalN:])
      print j,log_loss(train_y[-testvalN:],traincolnext[-testvalN:])

      testval=traincolnext[-testvalN:]
      traincolnext=traincolnext[:-testvalN]
      testcolnext=np.concatenate((testval,testcolnext))
      trainnext,testnext=nextlevel(level,traincolnext,testcolnext,colname_sub)

    bagcols = [colname+'_'+str(j) for j in xrange(nbags)]
    if all([e in trainnext.columns for e in bagcols]):
      if testnext.ix[:,bagcols].mean().min()>0:
        from scipy.stats.mstats import gmean
        traincolnext=gmean(trainnext.ix[:,bagcols],1)
        testcolnext=gmean(testnext.ix[:,bagcols],1)
        nextlevel(level,traincolnext,testcolnext,colname,delcols=bagcols)
        os.execv(sys.executable,[sys.executable]+sys.argv)

if not args.second: os.execv(sys.executable,[sys.executable]+sys.argv+['--second'])
