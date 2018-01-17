level=1
import os,sys,random,time,itertools
from sklearn.metrics import log_loss
import cPickle as pickle, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
kf = KFold(n_splits=8,shuffle=False)
tmpfile = '/tmp/' + str(time.time()).replace(".","") + str(random.random())
train_raw = 'raw/numerai_training_data.csv'
test_raw = 'raw/numerai_tournament_data.csv'
# polyfile = 'chebyshev.dat'
# polycache = 'chebyshev'
# polyeval = np.polynomial.chebyshev.chebval
polyfile = 'poly.dat'
polycache = 'poly'
polyeval = np.polynomial.legendre.legval
npolyfeat=100000
index_col=0
scoredp=7  #score's decimal places
target='target'
dropfeat=['data_type','era','testp',target]
def set_path():
  try:
    if os.uname()[1]=='sun.phys.vt.edu': path = "/Users/hyhui/Downloads/numerai/" 
    elif os.uname()[1]=="tbird.phys.vt.edu" or "-" in os.uname()[1]: path ="/home/hyhui/numerai/"
    elif os.uname()[1][:2]=='hs': path ="/home/hyhui/numerai/"
    else: path=random.choice(["/work/cascades/hyhui/numerai/","/work/cascades/hyhui/numerai/"])
  except: path = "D:/numerai/"
  os.chdir(path)

embed_p=[#('isomap',5),('isomap',50),('lda',),
         ('tsne',2,10),('tsne',2,30),('tsne',2,100),
         ('lv',2,10),('lv',2,30),('lv',2,100)]
# embed_p=[]
if True:
  def embed_largevis(train,test,dim=2,pp=50,nproc=-1):
      if nproc==-1:
        import psutil
        nproc=psutil.cpu_count()
      traintest=pd.DataFrame(np.concatenate((train,test)))
      traintest.to_csv("LargeVis_in",header=[str(traintest.shape[0]),str(traintest.shape[1])]+["" for i in xrange(traintest.shape[1]-2)],sep=' ',index=False)
      os.system('python LargeVis_run.py -input LargeVis_in -output LargeVis_out -outdim {} -perp {} -neigh {} -threads {}'.format(dim,pp,3*pp,nproc))
      X2d = pd.read_csv('LargeVis_out',skiprows=1,sep=' ',header=None).values
      X2d = MinMaxScaler().fit_transform(X2d)
      return X2d[:train.shape[0]], X2d[train.shape[0]:]
  def embed_lle(train,test,nn=10,method='standard'):
      traintest=np.concatenate((train,test))
      from sklearn.manifold import LocallyLinearEmbedding
      lle=LocallyLinearEmbedding(n_neighbors=nn,n_components=2,method=method)
      lle.fit(traintest)
      X2d=lle.transform(traintest)
      X2d=MinMaxScaler().fit_transform(X2d)
      return X2d[:train.shape[0]], X2d[train.shape[0]:]
  def embed_mds(train,test):
      traintest=np.concatenate((train,test))
      from sklearn.manifold import MDS
      mds=MDS(n_components=2, n_init=1, max_iter=100,verbose=5)
      X2d=mds.fit_transform(traintest)
      X2d=MinMaxScaler().fit_transform(X2d)
      return X2d[:train.shape[0]], X2d[train.shape[0]:]
  def embed_spectral(train,test):
      traintest=np.concatenate((train,test))
      from sklearn.manifold import SpectralEmbedding
      se=SpectralEmbedding(n_components=2,eigen_solver="arpack")
      X2d=se.fit_transform(traintest)
      X2d=MinMaxScaler().fit_transform(X2d)
      return X2d[:train.shape[0]], X2d[train.shape[0]:]
  def embed_isomap(train,test,nn):
      traintest=np.concatenate((train,test))
      from sklearn.manifold import Isomap
      iso=Isomap(n_neighbors=nn)
      # iso.fit(traintest)
      X2d=iso.fit_transform(traintest)
      X2d=MinMaxScaler().fit_transform(X2d)
      return X2d[:train.shape[0]], X2d[train.shape[0]:]
  def embed_lda(train,test,train_y):
      from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
      mms=MinMaxScaler()
      X2 = np.array(train).copy()
      X2.flat[::X2.shape[1] + 1] += 0.001  # Make X invertible
      lda=LinearDiscriminantAnalysis(n_components=2)
      lda.fit(X2,np.array(train_y))
      train2D = lda.transform(X2)
      train2D = mms.fit_transform(train2D)
      X2 = np.array(test).copy()
      X2.flat[::X2.shape[1] + 1] += 0.001  # Make X invertible
      test2D = lda.transform(X2)
      test2D = mms.transform(test2D)
      return train2D,test2D
  def embed_tsne(train,test,dim,pp):
      traintest=np.concatenate((train,test))
      if dim==2:
        from MulticoreTSNE import MulticoreTSNE as TSNE
        tsne=TSNE(n_components=2,perplexity=float(pp),n_iter=1000,n_jobs=-1)
        X2d =tsne.fit_transform(traintest.astype(float).copy())
      else:
        try: 
          from tsne import bh_sne
          X2d = bh_sne(traintest.astype(float),d=3,perplexity=float(pp),copy_data=True)
        except:
          from sklearn.manifold import TSNE
          X2d = TSNE(n_components=3,perplexity=float(pp),verbose=5,n_iter=1000).fit_transform(traintest.copy())
      X2d=MinMaxScaler().fit_transform(X2d)
      return X2d[:train.shape[0]],X2d[train.shape[0]:]

def cols_train_y_test(level,loadtest=False,coltype=False):
  trainfile='train%d'%level
  testfile='test%d'%level
  if level==0 or (level==1 and not os.path.isfile(trainfile)):  # reorder rows according to testp
    if not os.path.isfile('train0'):
      train=pd.read_csv(train_raw,index_col=index_col)
      test=pd.read_csv(test_raw,index_col=index_col)

      # traintest0['is_test']=1
      # traintest0.iloc[:train0.shape[0],-1]=0
      # import xgboost
      # from sklearn.model_selection import cross_val_predict
      # model=xgboost.XGBClassifier(n_estimators=100)
      # pred=cross_val_predict(model,traintest0[cols0],traintest0.is_test,cv=kftrain,method='predict_proba')[:,1]
      # traintest0['testp']=pred/np.mean(pred)
      # test = traintest0[train0.shape[0]:].drop(['is_test'],axis = 1)
      # traintest0 = traintest0.iloc[pred.argsort()]
      # train = traintest0[traintest0.is_test==0].drop(['is_test'],axis = 1)

      train.to_pickle('train0')
      test.to_pickle('test0')
    elif level=='0':
      train=pd.read_pickle('train0')
      if loadtest: test=pd.read_pickle('test0')
  train0=pd.read_pickle("train0")
  if loadtest: test0=pd.read_pickle('test0')
  if level==1: #feature engineering: legendre, embedding      
    if not os.path.isfile(trainfile):
      train=pd.read_pickle("train0")
      test=pd.read_pickle("test0")
      train_=pd.concat((train,test[test.data_type=='validation']))
      test_=test[test.data_type!='validation']
      
      if coltype in ['poly','ext']:
        if not os.path.isfile(polyfile):
          with open(polyfile,'wb') as outfile: pickle.dump({'ev':[],'data':{}},outfile, pickle.HIGHEST_PROTOCOL)
        polydata=pickle.load(open(polyfile,'rb'))
        cols=[c for c in train.columns if c not in dropfeat]
        lfeats=[]
        for n in [1,2,3,4]: lfeats=lfeats+list(['|'+'|'.join(c) for c in itertools.combinations(cols,n)])
        print 'total:',len(lfeats)
        remain=[e for e in lfeats if e not in polydata['ev']]
        random.shuffle(remain)
        if remain:
          print 'remain:',len(remain)
          i=0
          for count,f in enumerate(remain):
            if f not in polydata['ev']:
              i+=1;print i,f
              newd,traincol,_=poly_exp(train_[f[1:].split('|')],train_[target],test_[f[1:].split('|')])###########################
              polydata['ev'].append(f)
              polydata['data'].update({f:newd})
              if i%3==0 or count==len(remain)-1:
                try: ondisk=pickle.load(open(polyfile,'rb'))
                except: ondisk={'ev':[],'data':{}}
                polydata['ev']=list(set(polydata['ev']).union(ondisk['ev']))
                polydata['data'].update(ondisk['data'])
                if len(polydata['ev'])!=len(ondisk['ev']): 
                  d=polydata['data']
                  if len(d)>npolyfeat: polydata['data']=dict(sorted(d.iteritems(), key=lambda x:x[1]['score'], reverse=True)[:npolyfeat])
                  with open(polyfile,'wb') as outfile:
                    pickle.dump(polydata,outfile, pickle.HIGHEST_PROTOCOL)
      if coltype == 'ext':
        cols=[c for c in train.columns if c not in dropfeat]
        for p in random.sample(embed_p,len(embed_p)):
          npfile='_'.join(map(str,p))+'.npz'
          if not os.path.isfile(npfile):
            print 'running: ', npfile
            try:
              if p[0]=='isomap': train2D,test2D=embed_isomap(train[cols],test[cols],p[1])
              elif p[0]=='lda':  train2D,test2D=embed_lda(train[cols],test[cols],train[target])
              elif p[0]=='tsne': train2D,test2D=embed_tsne(train[cols],test[cols],p[1],p[2])
              elif p[0]=='lv': train2D,test2D=embed_largevis(train[cols],test[cols],p[1],p[2])
              np.savez(npfile, train=train2D,test=test2D)
            except MemoryError:
              print 'insufficient memory'
        for p in embed_p:
          print p
          npfile='_'.join(map(str,p))+'.npz'
          d=np.load(npfile)
          columns=[npfile.split('.')[0]+'_%d'%i for i in range(d['train'].shape[1])]
          train=pd.concat((train,pd.DataFrame(d['train'],columns=columns,index=train.index)),axis=1)
          test=pd.concat((test,pd.DataFrame(d['test'],columns=columns,index=test.index)),axis=1)

      train.to_pickle(trainfile)
      test.to_pickle(testfile)
    else:
      while True:
        try:
          train=pd.read_pickle(trainfile)
          if loadtest: test=pd.read_pickle(testfile)
          break
        except:time.sleep(5)
  elif level>1:
    cols0=['feature%d'%i for i in range(1,22)]
    train=pd.read_pickle(trainfile)
    traindrop=train.replace(0.0,np.nan).dropna(1,how='all').dropna(0,how='any')
    trainN=traindrop.shape[0]
    traincols=[c for c in traindrop if c.count('_')<2]
    train=train[traincols]
    train=pd.concat((train,train0[cols0]),1).tail(trainN)
    if loadtest: 
      test=pd.read_pickle(testfile)[traincols]
      test=pd.concat((test,test0[cols0]),1)
  extra_y=train0[['era',target]]
  cols=[c for c in train.columns if c not in dropfeat]
  if level==1:
    if coltype=='raw': cols=['feature%d'%i for i in range(1,22)]
    elif coltype in ['poly','ext']:
      cols=[]
      with open(polyfile,'rb') as infile: polydata=pickle.load(infile)
      for ncomp in [1,2,3,4]:
        compositecols=[k for k in polydata['data'] if k.count('|')==ncomp]
        compositecols.sort(key=lambda k: polydata['data'][k]['score'],reverse=True)
        elemcount=[]
        for k in compositecols:
          klist = filter(None,k.split('|'))
          if all(elemcount.count(ke)<ncomp*ncomp for ke in klist) and all(ke in train for ke in klist):
            cols.append(k)
            elemcount.extend(klist)
      if coltype=='poly': cols.extend(['feature%d'%i for i in range(1,22)])
      else: cols.extend([c for c in train.columns if c not in dropfeat])
  if loadtest: return cols,train,extra_y,test
  else: return cols,train,extra_y



def poly_exp(traincols,train_y,testcols,config={}):
    nc=traincols.shape[1]
    traincols=2*traincols-1
    testcols=2*testcols-1
    if not config or config['uniformized']:
        traintest=np.concatenate((traincols,testcols),0)
        traintest=2*(pd.DataFrame(traintest).rank().values)/float(traintest.shape[0])-1
        traincolsU=traintest[:traincols.shape[0]]
        testcolsU=traintest[traincols.shape[0]:]

    bestconfig={'score':-np.inf}
    # bestscore0=-np.inf
    besttraincol=besttestcol=0
    if not config:
      for train_,test_,uniformized in [(traincols,testcols,False),(traincolsU,testcolsU,True)]:
        colid=[tuple([0]*nc)] #weight matrix for entanglement, colid=indices except first col
        trainleg=np.ones(traincols.shape)
        testleg=np.ones(testcols.shape)
        traintmp=np.ones((traincols.shape[0],1))
        testtmp=np.ones((testcols.shape[0],1))
        for orderi in range(1,7):
          # order=orderi
          order=(-1)**((orderi-1)%2) * ((orderi+1)/2)
          newtrainleg=train_**order
          newtestleg=test_**order
          newtrainleg=pd.DataFrame(newtrainleg).replace([np.inf,-np.inf],np.nan)
          newtrainleg=(newtrainleg/newtrainleg.max()).fillna(newtrainleg.mean()).values
          newtestleg=pd.DataFrame(newtestleg).replace([np.inf,-np.inf],np.nan).fillna(np.mean(newtestleg))
          newtestleg=(newtestleg/newtestleg.max()).fillna(newtestleg.mean()).values
          trainleg=np.concatenate((trainleg,newtrainleg),1)
          testleg=np.concatenate((testleg,newtestleg),1)
          for tup in itertools.product(*[range(orderi+1) for ii in range(nc)]):
            if orderi in tup:
              colid.append(tup)
              tup=traincols.shape[1]*np.array(tup)+np.array(range(nc))
              traintmp=np.concatenate((traintmp,trainleg[:,tup].prod(1).reshape(-1,1)),1)
              testtmp=np.concatenate((testtmp,testleg[:,tup].prod(1).reshape(-1,1)),1)
          lr = LogisticRegressionCV(Cs=10**np.linspace(-4,4,9),fit_intercept=False,scoring='neg_log_loss',n_jobs=1)
          lr.fit(np.clip(traintmp,-10,10),train_y)
          # lr0 = LogisticRegressionCV(Cs=10**np.linspace(-5,5,11),fit_intercept=False,scoring='neg_log_loss',n_jobs=1)
          # lr0.fit(np.nan_to_num(traintmp[:,np.where(np.logical_not(map(all,colid)))[0]]),train_y)
          score = np.log(2)+lr.scores_[1].mean(0).max()
          # score0 = np.log(2)+lr0.scores_[1].mean(0).max()
          # bestscore0=max(bestscore0,score0)
          print uniformized,order,score,lr.C_[0]
          if score>bestconfig['score']:
            bestconfig={'C':lr.C_[0],'w':lr.coef_.T,'score':score,'orderi':orderi,'uniformized':uniformized}
            w=lr.coef_.T
            besttraincol=traintmp.dot(w/np.linalg.norm(w))
            besttestcol=testtmp.dot(w/np.linalg.norm(w))
          if orderi>bestconfig['orderi']+3:break
    else:
      w=config['w']
      if config['uniformized']: 
        traincols=traincolsU
        testcols=testcolsU
      trainleg=np.ones(traincols.shape)
      testleg=np.ones(testcols.shape)
      traintmp=np.ones((traincols.shape[0],1))
      if testcols is not None: testtmp=np.ones((testcols.shape[0],1))
      for orderi in range(1,config['orderi']+1):
        # order=orderi
        order=(-1)**((orderi-1)%2) * ((orderi+1)/2)
        newtrainleg=traincols**order
        newtestleg=testcols**order
        newtrainleg=pd.DataFrame(newtrainleg).replace([np.inf,-np.inf],np.nan)
        newtrainleg=(newtrainleg/newtrainleg.max()).fillna(newtrainleg.mean()).values
        newtestleg=pd.DataFrame(newtestleg).replace([np.inf,-np.inf],np.nan).fillna(np.mean(newtestleg))
        newtestleg=(newtestleg/newtestleg.max()).fillna(newtestleg.mean()).values
        trainleg=np.concatenate((trainleg,newtrainleg),1)
        testleg=np.concatenate((testleg,newtestleg),1)
        for tup in itertools.product(*[range(orderi+1) for ii in range(nc)]):
          if orderi in tup:
            tup=traincols.shape[1]*np.array(tup)+np.array(range(nc))
            traintmp=np.concatenate((traintmp,trainleg[:,tup].prod(1).reshape(-1,1)),1)
            testtmp=np.concatenate((testtmp,testleg[:,tup].prod(1).reshape(-1,1)),1) 
      besttraincol=traintmp.dot(w/np.linalg.norm(w))
      besttestcol=testtmp.dot(w/np.linalg.norm(w))

    return bestconfig,besttraincol,besttestcol
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
class FractionalLogisticRegression(BaseEstimator, TransformerMixin):
    def __init__(self,C=1.0,tol=1E-4):
        self.C=C
        self.tol=tol
    def loglossf(self, w, X, y, alpha):
        N, m = X.shape
        grad = np.empty_like(w)
        # z = X.dot(w)
        Xw =  X.dot(w[:m])                  # (N,)
        Xv = X[:,0]+X[:,1:].dot(w[1-m:])    # (N,)
        z = Xw / Xv                         # (N,)
        yz = y*z                            # (N,)
        yzpos = yz[yz>0]
        yzneg = yz[yz<0]
        cost = .5 * alpha * np.dot(w, w)  #scaler
        if yzpos.shape[0]: cost += np.sum(np.log(1 + np.exp(-yzpos)))
        if yzneg.shape[0]: cost += -np.sum(yzneg-np.log(1 + np.exp(yzneg)))
        z = expit(yz)  #1/(1+exp(-yz))            # (N,)
        z0 = y * (z - 1.)  # -y_n /(1+exp(yz))    # (N,)
        grad[:m] = X.T.dot(z0/Xv)  # (m,)
        grad[1-m:] = -X[:,1:].T.dot(z0*Xw/Xv/Xv)
        grad += alpha * w
        # print 'cost:',cost
        # print 'grad:',grad
        return cost, grad
    def fit(self, X, y=None):
        w0 = np.random.random(2*X.shape[1]-1).ravel()
        from scipy import optimize
        w, loss, info = optimize.fmin_l_bfgs_b(
                        self.loglossf, w0, fprime=None,
                        args=(X, 2*y-1, 1. / self.C),
                        iprint=0,pgtol=self.tol, maxiter=100)
        self.coef_=w
        return self
    def predict(self,X):
        m=X.shape[1]
        class1=expit(X.dot(self.coef_[:m])/(X[:,0]+X[:,1:].dot(self.coef_[1-m:])))
        return np.round(class1).astype(int)
    def predict_proba(self,X):
        m=X.shape[1]
        class1=expit(X.dot(self.coef_[:m])/(X[:,0]+X[:,1:].dot(self.coef_[1-m:])))
        return np.c_[1-class1,class1]
from sklearn.model_selection import cross_val_predict
class FractionalLogisticRegressionCV(BaseEstimator, TransformerMixin):
    def __init__(self,Cs=[.1,1,10],tol=1E-4):
        self.Cs=Cs
        self.tol=tol
    def fit(self, X, y=None):
        self.score_=np.inf
        for C in self.Cs:
            model = FractionalLogisticRegression(C=C,tol=self.tol)
            pred = cross_val_predict(model,X,y,method='predict_proba')[:,1]
            score = log_loss(y,pred)
            if score<self.score_:
              self.score_ = score
              self.C_ = C
        self.model_ = FractionalLogisticRegression(C=self.C_,tol=self.tol)
        self.model_.fit(X,y)
        self.coef_ = self.model_.coef_        
        return self
    def predict(self,X): return self.model_.predict(X)
    def predict_proba(self,X): return self.model_.predict_proba(X)
def fracpoly_exp(traincols,train_y,testcols,config={}):
    nc=traincols.shape[1]
    traincols=2*traincols-1
    testcols=2*testcols-1
    if not config or config['uniformized']:
        traintest=np.concatenate((traincols,testcols),0)
        traintest=2*(pd.DataFrame(traintest).rank().values-1)/float(traintest.shape[0]-1)-1
        traincolsU=traintest[:traincols.shape[0]]
        testcolsU=traintest[traincols.shape[0]:]

    bestconfig={'score':-np.inf}
    # bestscore0=-np.inf
    besttraincol=besttestcol=0
    if not config:
      for train_,test_,uniformized in [(traincols,testcols,False),(traincolsU,testcolsU,True)]:
        colid=[tuple([0]*nc)] #weight matrix for entanglement, colid=indices except first col
        trainleg=np.ones(traincols.shape)
        testleg=np.ones(testcols.shape)
        traintmp=np.ones((traincols.shape[0],1))
        testtmp=np.ones((testcols.shape[0],1))
        for order in range(1,10):
          legvec=[0]*(order+1); legvec[order]=1
          newtrainleg=polyeval(train_,legvec)
          newtestleg=polyeval(test_,legvec)
          trainleg=np.concatenate((trainleg,newtrainleg),1)
          testleg=np.concatenate((testleg,newtestleg),1)
          for tup in itertools.product(*[range(order+1) for ii in range(nc)]):
            if order in tup:
              colid.append(tup)
              tup=traincols.shape[1]*np.array(tup)+np.array(range(nc))
              traintmp=np.concatenate((traintmp,trainleg[:,tup].prod(1).reshape(-1,1)),1)
              testtmp=np.concatenate((testtmp,testleg[:,tup].prod(1).reshape(-1,1)),1)
          lr = FractionalLogisticRegressionCV(Cs=10**np.linspace(-4,4,9))
          lr.fit(traintmp,train_y)
          score = np.log(2)-lr.score_
          print uniformized,order,score,lr.C_
          if score>bestconfig['score']:
            bestconfig={'C':lr.C_,'w':lr.coef_,'score':score,'order':order,'uniformized':uniformized}
            w=lr.coef_; m=traintmp.shape[1]
            besttraincol=traintmp.dot(w[:m])/(traintmp[:,0]+traintmp[:,1:].dot(w[1-m:]))
            besttestcol=testtmp.dot(w[:m])/(testtmp[:,0]+testtmp[:,1:].dot(w[1-m:]))
          if order>bestconfig['order']+1:break
    else:
      w=config['w']
      if config['uniformized']: 
        traincols=traincolsU
        testcols=testcolsU
      trainleg=np.ones(traincols.shape)
      testleg=np.ones(testcols.shape)
      traintmp=np.ones((traincols.shape[0],1))
      if testcols is not None: testtmp=np.ones((testcols.shape[0],1))
      for order in range(1,config['order']+1):
        legvec=[0]*(order+1); legvec[order]=1
        newtrainleg=polyeval(traincols,legvec)
        newtestleg=polyeval(testcols,legvec)
        trainleg=np.concatenate((trainleg,newtrainleg),1)
        testleg=np.concatenate((testleg,newtestleg),1)
        for tup in itertools.product(*[range(order+1) for ii in range(nc)]):
          if order in tup:
            tup=traincols.shape[1]*np.array(tup)+np.array(range(nc))
            traintmp=np.concatenate((traintmp,trainleg[:,tup].prod(1).reshape(-1,1)),1)
            testtmp=np.concatenate((testtmp,testleg[:,tup].prod(1).reshape(-1,1)),1) 
      m=traintmp.shape[1]
      besttraincol=traintmp.dot(w[:m])/(traintmp[:,0]+traintmp[:,1:].dot(w[1-m:]))
      besttestcol=testtmp.dot(w[:m])/(testtmp[:,0]+testtmp[:,1:].dot(w[1-m:]))

    return bestconfig,besttraincol,besttestcol
    
def generate_train_x(chosen,train,train_y,test,cache=True,returntest=False):
  newcols = [c for c in chosen if c not in train.columns]
  updatepolydata=False
  updatetraincache=False
  updatetestcache=False
  if any('|' in e for e in newcols):
    if not os.path.isfile(polyfile):
      with open(polyfile,'wb') as outfile:
        pickle.dump({'ev':[],'data':{}},outfile, pickle.HIGHEST_PROTOCOL)
    with open(polyfile,'rb') as infile: polydata=pickle.load(infile)
    if os.path.isfile(polycache+'cache'):
      while True:
        try: 
          traincache=pd.read_pickle(polycache+'cache')
          break
        except: time.sleep(5)
    else:
      traincache=pd.DataFrame(index=train.index)
    if returntest: 
      if os.path.isfile(polycache+'cachetest'):
        while True:
          try: 
            testcache=pd.read_pickle(polycache+'cachetest')
            break
          except: time.sleep(5)
      else:
        testcache=pd.DataFrame(index=test.index)
  for e in newcols:
    if '|' in e:
      if e in traincache.columns and traincache.shape[0]==train.shape[0] and (not returntest or e in testcache.columns):
        traincol=traincache[e]
        if returntest: testcol=testcache[e]
      else:
        start=time.time()
        elist=filter(None,e.split('|'))
        if e not in polydata['data']:
          newd,traincol,testcol=poly_exp(train[elist],train_y,test[elist])
          polydata['data'].update({e: newd})
          polydata['ev'].append(e)
          updatepolydata=True
        else:
          _,traincol,testcol=poly_exp(train[elist],train_y,test[elist],polydata['data'][e])
        if cache and time.time()-start>2.5 and (e not in traincache or e not in testcache):
          print e,polydata['data'][e]['orderi']
          if cache and e not in traincache: 
            traincache[e]=traincol
            updatetraincache=True
          if cache and returntest and e not in testcache:
            testcache[e]=testcol
            updatetestcache=True
      train[e]=traincol
      if returntest: test[e]=testcol
    elif '+' in e:
      lr = RidgeCV(alphas=10**np.linspace(-2,2,5),fit_intercept=True,store_cv_values=True)
      w=lr.fit(train[e.split('+')].fillna(0),train_y).coef_.T
      train[e]=train[e.split('+')].dot(w/np.linalg.norm(w))
      test[e]=test[e.split('+')].dot(w/np.linalg.norm(w))
    elif '*' in e:
      train[e]=train[e.split('*')[0]]*train[e.split('*')[1]]
      test[e]=test[e.split('*')[0]]*test[e.split('*')[1]]
    elif '/' in e:
      train[e]=train[e.split('/')[0]]/train[e.split('/')[1]]
      test[e]=test[e.split('/')[0]]/test[e.split('/')[1]]
    elif '^' in e:
      train[e]=train[e.split('^')[0]]**train[e.split('^')[1]]
      test[e]=test[e.split('^')[0]]**test[e.split('^')[1]]
  if updatetraincache: 
    try: 
      ondisk=pd.read_pickle(polycache+'cache')
      for c in ondisk.columns: traincache[c]=ondisk[c].values
    except: pass
    traincache.to_pickle(polycache+'cache')
  if updatetestcache:
    try:
      ondisk=pd.read_pickle(polycache+'cachetest')
      for c in ondisk.columns: testcache[c]=ondisk[c].values
    except: pass
    testcache.to_pickle(polycache+'cachetest')
  if updatepolydata:
    try: ondisk=pickle.load(open(polyfile,'rb'))
    except: ondisk={'ev':[],'data':{}}
    polydata['ev']=list(set(polydata['ev']).union(ondisk['ev']))
    polydata['data'].update(ondisk['data'])
    with open(polyfile,'wb') as outfile:
      pickle.dump(polydata,outfile, pickle.HIGHEST_PROTOCOL) 
  train = train.ix[:,chosen]
  if returntest: test = test.ix[:,chosen]
  if returntest: return train,test
  else: return train


def gen_featimpmean(featimp):
  if featimp.shape[0]<5:return featimp.mean()
  weights = .01+.99*(featimp.index-featimp.index.min())/(featimp.index.max()-featimp.index.min())
  weights = pd.Series(weights.values,index=featimp.index)
  featimpmean = featimp.mul(weights,axis=0).sum()/featimp.notnull().mul(weights,axis=0).sum()
  scoreimp = featimp.replace(0,np.nan).notnull().mul(weights,axis=0).sum()/featimp.replace(0,np.nan).notnull().sum()  #if chosen but fscore=0, ignore that
  featimpmean /= featimpmean.sum()
  scoreimp /= scoreimp.sum()
  result = featimpmean
  return result/result.sum()
def weighted_featimp(featimp,chosen):
  if featimp.shape[0]<5 or not chosen:return featimp.drop(chosen,1).mean()
  weights = featimp[chosen].count(1)
  weights = .01+.99*weights/weights.max()
  unchosen = featimp.drop(chosen,1)
  weightedmean = unchosen.mul(weights,axis=0).sum()/unchosen.notnull().mul(weights,axis=0).sum()
  weightedmean /= weightedmean.sum()
  return weightedmean
def featimp_from_cols(cols):  #feature importance based on score (from various models) cols' keys are scores, values are list of feature names
  keys = cols.keys()
  from collections import defaultdict
  featimp = defaultdict(float)
  for score,flist in cols.iteritems():
    for f in flist:
      featimp[f]+=score-min(keys)
  total=np.sum(featimp.values())
  featimp={k:v/total for k,v in featimp.iteritems()}
  return featimp
  
  
def nextlevel(levelorsuf,traincol=None,testcol=None,colname=None,delcols=[]):
    try:
      trainpath='train%d' %(levelorsuf+1)
      testpath='test%d'%(levelorsuf+1)
    except TypeError:
      trainpath='train'+levelorsuf
      testpath='test'+levelorsuf
    if os.path.isfile(trainpath):
      train_=pd.read_pickle(trainpath)
      test_=pd.read_pickle(testpath)
    else:
      train_ = pd.DataFrame(index=pd.read_csv(train_raw,usecols=[0],index_col=0).index)
      test_ =  pd.DataFrame(index=pd.read_csv(test_raw,usecols=[0],index_col=0).index)
    if delcols:
      train_.drop([c for c in delcols if c in train_.columns],1,inplace=True)
      test_.drop([c for c in delcols if c in test_.columns],1,inplace=True)      
    if traincol is not None and testcol is not None and colname is not None:
      print 'nextlevel',colname
      train_[colname]=0.
      train_[colname].iloc[-traincol.shape[0]:]=traincol
      test_[colname]=0.
      test_[colname].iloc[-testcol.shape[0]:]=testcol
    if traincol is not None or testcol is not None or delcols:
      if all(c in test_.columns for c in train_.columns): test_=test_[train_.columns]
      train_.to_pickle(trainpath)
      test_.to_pickle(testpath)
    return train_,test_