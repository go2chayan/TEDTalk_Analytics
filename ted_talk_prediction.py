from list_of_talks import all_valid_talks
from ted_talk_sentiment import Sentiment_Comparator, read_bluemix
import ted_talk_cluster_analysis as tca
import numpy as np
import scipy as sp
import sklearn as sl
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

kwlist = ['beautiful', 'ingenious', 'fascinating',
            'obnoxious', 'confusing', 'funny', 'inspiring',
             'courageous', 'ok', 'persuasive', 'longwinded', 
             'informative', 'jaw-dropping', 'unconvincing','Totalviews']

def loaddata(indexfile='./index.csv'):
    csv_,vid = tca.read_index(indexfile)
    m = len(all_valid_talks)
    dict_input = {'group_1':all_valid_talks[:m/2],
                  'group_2':all_valid_talks[m/2:]}
    # Load into sentiment comparator for all the pre-comps
    comp = Sentiment_Comparator(dict_input,read_bluemix)
    scores=[]
    Y=[]
    for atalk in comp.alltalks:
        scores.append(comp.sentiments_interp[atalk])
        temp = []
        for akw in kwlist:
            if akw == 'Totalviews':
                temp.append(int(csv_[akw][vid[atalk]]))
            else:
                temp.append(float(csv_[akw][vid[atalk]])/\
                    float(csv_['total_count'][vid[atalk]])*100.)
        Y.append(temp)
    return np.array(scores),np.array(Y),kwlist

def feat_sumstat(scores):
    '''
    Calculate the summary statistics of the scores such as min, max, 
    average, standard deviation etc.    
    Take minimum of the scores
    '''
    X = np.min(scores,axis=1)
    nkw = [akw+'_min' for akw in kwlist]
    # Concat maxmimum of the scores
    X = np.concatenate((X,np.max(scores,axis=1)),axis=1)
    nkw+= [akw+'_max' for akw in kwlist]
    # Concat average of the scores 
    X = np.concatenate((X,np.mean(scores,axis=1)),axis=1)
    nkw+= [akw+'_avg' for akw in kwlist]
    # Concat standard deviation of the scores 
    X = np.concatenate((X,np.std(scores,axis=1)),axis=1)
    nkw+= [akw+'_std' for akw in kwlist]
    return X, nkw

def traintest_idx(N,testsize=0.3):
    '''
    Get the index of training and test split. 
    N is the length of the dataset (sample size)
    '''
    testidx = np.random.rand(int(N*testsize))*N
    testidx = testidx.astype(int).tolist()
    trainidx = [i for i in xrange(N) if not i in testidx]
    return trainidx,testidx

def discretizeY(Y,col):
    '''
    Discretize and returns and specific column of Y. The strategy is:
    to keep the data with score <=33rd percentile be the "low" group,
    score >=66th percentile be the "high" group, and the middle be the
    "medium" group.
    '''
    y = Y[:,col]
    lowthresh = sp.percentile(y,33.3333)
    hithresh = sp.percentile(y,66.6666)
    y[y<=lowthresh] = -1    # Low group
    y[y>=hithresh] = 1      # High group
    y[(y>lowthresh)*(y<hithresh)] = 0   # Medium group
    return y

def binarize(X,y):
    '''
    Keeps only the good and bad parts in the data. Drops the medium part.
    '''
    idxmed = y!=0
    return X[idxmed,:],y[idxmed]

def classifier_eval(clf_trained,X_test,y_test,use_proba=True,
        ROCTitle=None,outfilename='./plots/'):
    y_pred = clf_trained.predict(X_test)
    print 'Report on Test Data'
    print '-------------------'
    print sl.metrics.classification_report(y_test,y_pred)
    print 'Accuracy:',sl.metrics.accuracy_score(y_test,y_pred)
    if use_proba:
        try:
            # trying to get the confidence scores
            y_score = clf_trained.decision_function(X_test)
        except AttributeError:
            print 'model does not have any method named decision function'
            print 'Trying predict_proba:'
            try:
                y_score = clf_trained.predict_proba(X_test)
            except:
                raise
        auc = sl.metrics.roc_auc_score(y_test,y_score)
        print 'AUC:',auc
        fpr,tpr,_ = sl.metrics.roc_curve(y_test,y_score,pos_label=1)        
        plt.figure()
        plt.plot(fpr,tpr,color='darkorange',label='ROC Curve (AUC={0:0.2f})'.\
            format(auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if ROCTitle:
            plt.title(ROCTitle)
        plt.legend()
        if not outfilename:
            plt.show()
        else:
            plt.savefig(outfilename+ROCTitle+'.png')
        

def regressor_eval(y_true,y_pred):
    pass

def train_with_CV(X,y,clf,cvparams=None,
        score_func=roc_auc_score,Nfold=3,nb_iter=10,
        showCV_report=False,use_proba=True,datname=''):
    '''
    Trains the estimator with N fold cross validation. The number of fold
    is given by the parameter Nfold. cvparams is a dictionary specifying
    the hyperparameters of the classifier that needs to be tuned. Scorefunc
    is the metric to evaluate the classifier. 
    '''
    scorer = sl.metrics.make_scorer(score_func)
    randcv = sl.model_selection.RandomizedSearchCV(clf,cvparams,
        n_iter=nb_iter,scoring=scorer,cv=Nfold)
    randcv.fit(X,y)
    y_pred = randcv.best_estimator_.predict(X)
    print 'Report on Training Data'
    print '-----------------------'
    print 'Best parameters:',randcv.best_params_
    print 'Best Score:',randcv.best_score_
    classifier_eval(randcv.best_estimator_,X,y,use_proba,
        'ROC on Training Data '+datname)
    if showCV_report:
        print 'CV Results:'
        print randcv.cv_results_
    return randcv.best_estimator_,randcv.best_score_
    
