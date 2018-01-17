import argparse 
from common import *
set_path()
parser = argparse.ArgumentParser()
parser.add_argument('-np',  type=int,  default=-1)
parser.add_argument('-n',   type=int,  default=5)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--verbose", help="increase output verbosity",action="store_true")
parser.add_argument("--second", action="store_true")
parser.add_argument('-level',  type=int,  default=level)
args = parser.parse_args()
level=args.level
nextleakfile='%d_leak'%(level+1)
cols0,train,extra_y,test=cols_train_y_test(level,loadtest=True,coltype='poly')
test1=pd.read_pickle('test1')
data_type=test1.data_type
trainN=train.shape[0]
testN=test.shape[0]
train=pd.concat((train,test[data_type=='validation']))#############################
test=test[data_type!='validation']#############################
extra_y=pd.concat((extra_y,test1[data_type=='validation'][extra_y.columns]))#############################
file="lr%d.dat" %(level)
base='lr' #for colname

while True:
  try:
    with open(file,'rb') as infile: (cols,p_prev)=pickle.load(infile)
    break
  except: time.sleep(5)

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,KFold
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
kftrain= KFold(n_splits=8,shuffle=False)
discreteP=['ncols']
p_range={'ncols': (2,400)}

erastarts=np.where(extra_y['era']!=extra_y['era'].shift())[0]
erastarts=erastarts[0::len(erastarts)/12]
nfold=4
cv=[(range(erastarts[i],erastarts[-nfold+i]),range(erastarts[-nfold+i],erastarts[1-nfold+i] if 1-nfold+i<0 else train.shape[0])) for i in range(nfold)]

# trainnext_leak,testnext_leak = nextlevel(nextleakfile)
trainnext,testnext = nextlevel(level)
for score in random.sample(sorted(p_prev.keys())[-args.n:],args.n):
    chosen=cols[score]
    colname = base+'_'+str(round(abs(score),scoredp))
    if colname in trainnext and\
      ((not args.second) or any(testnext[colname])):
      continue
    print colname,len(chosen)
    nextlevel(level,np.zeros((trainN,)),np.zeros((testN,)),colname)
    
    train_x,test_x = generate_train_x(chosen,train,extra_y[target],test,returntest=True)
    train_x = np.clip(train_x.fillna(train_x.mean()),-999,999)
    test_x  = np.clip(test_x.fillna(train_x.mean()),-999,999)
    train_y = extra_y[target]

    testcolnext =0
    traincolnext =0*train_y

    Cs=10.**np.array(np.linspace(-4,5,10))
    LR=LogisticRegressionCV(Cs=Cs,fit_intercept=True,cv=cv,scoring='neg_log_loss')
    LR.fit(train_x,extra_y[target])

    for i in range(nfold+1):
        tra_start=erastarts[i]
        val_start=erastarts[-nfold+i] if -nfold+i<0 else train_x.shape[0]
        val_end=erastarts[1-nfold+i] if 1-nfold+i<0 else train_x.shape[0]
        X_tra, y_tra = train_x.iloc[tra_start:val_start].values, train_y.iloc[tra_start:val_start].values
        X_val, y_val = train_x.iloc[val_start:val_end].values,   train_y.iloc[val_start:val_end].values
        model = LogisticRegression(C=LR.C_[0],solver='lbfgs')
        model.fit(X_tra,y_tra)
        if X_val.shape[0]:
          traincolnext.iloc[val_start:val_end]=model.predict_proba(X_val)[:,list(model.classes_).index(1)]
          print i,log_loss(y_val,traincolnext.iloc[val_start:val_end])
        testcolnext+=model.predict_proba(test_x)[:,list(model.classes_).index(1)]

    testcolnext/=nfold+1
    print log_loss(train_y.iloc[erastarts[-nfold]:],traincolnext.iloc[erastarts[-nfold]:])

    testval=traincolnext[trainN:]
    traincolnext=traincolnext[:trainN]
    testcolnext=np.concatenate((testval,testcolnext))
    # nextlevel(nextleakfile,traincolnext,testcolnext,colname)
    trainnext,testnext=nextlevel(level,traincolnext,testcolnext,colname)
      
      
if not args.second: os.execv(sys.executable,[sys.executable]+sys.argv+['--second'])
