from common import *
set_path()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-fs',  type=str,  default='sfs')
parser.add_argument('-cv',  type=int,  default=0)
parser.add_argument('-max',  type=int,  default=4)
parser.add_argument("--out", action="store_true")
args = parser.parse_args()

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import make_scorer
train=pd.read_pickle('train2')
test=pd.read_pickle('test2')
extra_y=pd.read_pickle('train0')[['era',target]]
test1=pd.read_pickle('test1')
data_type=test1.data_type
train=pd.concat((train,test[data_type=='validation']))#############################
extra_y=pd.concat((extra_y,test1[data_type=='validation'][extra_y.columns]))#############################

traindrop=train.replace(0.0,np.nan).dropna(1,how='all').dropna(0,how='any')
trainN=traindrop.shape[0]
traincols=[c for c in traindrop if c.count('_')<2]
train=train[traincols].tail(trainN)
train_y=extra_y[target].tail(trainN)

nval = test[data_type=='validation'].shape[0]
split = trainN - nval
cv = args.cv or [(range(split),range(split,trainN))]

print 'original train_err:\n', pd.concat((train.apply(lambda x:log_loss(train_y,x)),pd.Series(range(train.shape[1]),name='#',index=train.columns)),1)
print 'simple av validation error:', log_loss(train_y.iloc[split:],train.iloc[split:].mean(1))
Cs=10.**np.array(np.linspace(-4,5,50))
k_features=(1,min(train.shape[1],args.max))
  
lr=LogisticRegressionCV(Cs=Cs,fit_intercept=True)
# lr.fit(train,train_y)
if args.fs=='sfs': fs=SFS(lr,k_features=k_features,forward=True,floating=True,scoring='neg_log_loss',cv=cv,verbose=2)
else:              fs=EFS(lr,min_features=1,max_features=min(train.shape[1],8),scoring='neg_log_loss',cv=cv)
fs.fit(train.values,train_y.values)
print
print pd.DataFrame.from_dict(fs.get_metric_dict()).T
if args.fs=='sfs':
  print 'SFS best score:', fs.k_score_
  print len(fs.k_feature_idx_),'features:',fs.k_feature_idx_
else:
  print 'EFS best score:', fs.best_score_
  print len(fs.best_idx_),'features:',fs.best_idx_
  
lr.fit(fs.transform(train.iloc[:split].values),train_y.iloc[:split])
print
print 'Regularization C:', lr.C_
print 'validation error fitting on train:', log_loss(train_y.iloc[split:],lr.predict_proba(fs.transform(train.iloc[split:].values))[:,1])

lr.fit(fs.transform(train.iloc[split:].values),train_y.iloc[split:])
print 'validation error fitting on val:', log_loss(train_y.iloc[split:],lr.predict_proba(fs.transform(train.iloc[split:].values))[:,1])
if args.out:
  test1['probability']=lr.predict_proba(fs.transform(test[train.columns].values))[:,1]
  test1.to_csv('~/val.csv',index=True,columns=['probability'],float_format='%.6f')

lr.fit(fs.transform(train.values),train_y)
print 'validation error fitting on trainval:', log_loss(train_y.iloc[split:],lr.predict_proba(fs.transform(train.iloc[split:].values))[:,1])
if args.out:
  test1['probability']=lr.predict_proba(fs.transform(test[train.columns].values))[:,1]
  test1.to_csv('~/trainval.csv',index=True,columns=['probability'],float_format='%.6f')
