import numpy as np
import time
import sklearn
from qiskit import *
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import style
from amplification import amplification_updation 
from learner import weaklearner

class qrealboost(amplification_updation, weaklearner):

  classifiers = [] 
  Beta_j = []
  size = 0
  num_iterations = 25
  no_of_Q = 4

  def binary_predictions(self, X, y_mod, Dti, betas):
    '''
    used for calculating the value of H(x) from the beta values
    '''
    Hx = np.sum(betas, axis = 0)
    
    final_bin = []

    for j in range(len(Hx)):
        if np.sign(Hx[j]) == -1:
            final_bin.append(-1)
        if np.sign(Hx[j]) == 1:
            final_bin.append(1)

    acc = metrics.accuracy_score(final_bin, y_mod)
    
    return final_bin, acc

  def fit(self, X,y, num_iterations = 25,  no_of_Q = 4):
  
    '''
    The function which puts all the above functions together in order to obtain the accuracy and classifiers for the testing 
    
    '''
    self.num_iterations = num_iterations
    self.no_of_Q = no_of_Q

    # clipping the input data
    self.size = self.data_size(X)

    X = X[:2**self.size]
    y = y[:2**self.size]

    # all the arrays
    Dti = np.full(len(X),1/len(X))
    
    #change labels from 0,1 to -1,1
    y_mod = []
    for i in range(len(y)):
        if y[i]==0:
            y_mod.append(-1)
        else:
            y_mod.append(1)

    dti = []
    beta = []
    accuracy_final = []
    Z_all = []
    self.Beta_j = []
    self.classifiers = []

    for itr in range(num_iterations):

        time_prtitr_start = time.time()
        
        print('ITERATION - ', itr+1, '\n')

        # get weights, beta, predictions, classifier and confidence rating for current iteration
        dti_itr,beta_itr,final_beta_itr,preds_itr, classifier_itr, Zt_itr = self.update_dti(X, Dti, y)

        print("->  Updated Dti")
        print(dti_itr)
        dti.append(dti_itr)
        beta.append(final_beta_itr)
        self.Beta_j.append(beta_itr)
        self.classifiers.append(classifier_itr)
        Z_all.append(Zt_itr)

        # get binary labels and accuracy for current iteration
        final_bin, acc = self.binary_predictions(X,y_mod,Dti,beta)
        print("->  Training Accuracy : ", acc)
        accuracy_final.append(acc)
        
        Dti = dti_itr
        
        time_prtitr_end = time.time()
        
        print("->  Total time per iteration", time_prtitr_end - time_prtitr_start)
        print('-------------------------------------------------------------------------------------------------')

    plt.style.use('seaborn')
    plt.plot(list(range(num_iterations)), accuracy_final)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.show()
    
    return

  def predict(self, X, y):
    '''
    here beta_j is the function that contains the betas in the form - 
    [[beta0,beta1,beta2]itr=1,[beta0,beta1,beta2]itr=2,[beta0,beta1,beta2]itr=3....]
    all the clustering models are stored in classifiers
    '''
    T = self.num_iterations

    X = X[:2**self.size]
    y = y[:2**self.size]

    # change labels from 0,1 to -1,1
    y_mod = []
    for i in range(len(y)):
        if y[i]==0:
            y_mod.append(-1)
        else:
            y_mod.append(1)
    
    # partitioning to produce domains
    Dti = np.full(len(X),1/len(X))
    
    Beta_js = [] # the array which will store all the betas
    accuracy_final_test= []

    for i in range(T):

        beta_j = self.Beta_j[i]
        preds_itr = self.classifiers[i].predict(X)
        # print(preds_itr)
        final_beta =[]

        # updation of dti and distribution of betas
        for i in range(len(X)):
            if(preds_itr[i]==0):
                final_beta.append(beta_j[0])

            elif(preds_itr[i]==1):
                final_beta.append(beta_j[1])

            else:
                final_beta.append(beta_j[2])
            
        Beta_js.append(final_beta)
            
        # summing up all the betas from previous iterations to get H(x)
        final_bin, acc = self.binary_predictions(X,y_mod,Dti,Beta_js)
        accuracy_final_test.append(acc)
    
    print('Testing Accuracy : ' ,accuracy_final_test[-1])

    plt.style.use('seaborn')
    plt.plot(list(range(T)), accuracy_final_test)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Testing Accuracy")
    plt.show()
        
    return


class qadaboost(amplification_updation, weaklearner):

  classifiers = []
  preds_mat = []
  alpha = []
  size = 0
  num_iterations = 25
  no_of_Q = 4

  def final_bin_predictions(self, X, y_mod, predt):
    
    '''
    used for calculating the value of H(x) from the beta values
    '''
    
    final_bin = []
    
    for i in range(len(X)):
        Hx = 0
        for t in range(len(self.alpha)):
            Hx = Hx + self.alpha[t]*self.preds_mat[t][i]
            
        if(Hx>0):
            final_bin.append(1)
        else:
            final_bin.append(-1)
            
    inst_acc = metrics.accuracy_score(y_mod,predt)
    
    # print("Instantenous accuracy",inst_acc )
    
    acc = metrics.accuracy_score(final_bin, y_mod)

    return final_bin, acc

  def fit(self, X, y, num_iterations = 25,  no_of_Q = 4):
    
    '''
    The function which puts all the above functions together in order to obtain the accuracy and classifiers for the testing 
    
    '''  

    self.num_iterations = num_iterations
    self.no_of_Q = no_of_Q

    # clipping the input data
    self.size = self.data_size(X)

    X = X[:2**self.size]
    y = y[:2**self.size]

    
    # all the arrays
    Dti = np.full(len(X),1/len(X))
    
    #change labels from 0,1 to -1,1
    y_mod = []
    for i in range(len(y)):
        if y[i]==0:
            y_mod.append(-1)
        else:
            y_mod.append(1)

    dti = []
    # alpha = self.alpha
    accuracy_final = []
    self.alpha = []
    self.classifiers = []
    classifiers_all = self.classifiers
    self.preds_mat = []
    
    for itr in range(num_iterations):

        print('-------------------------------------------------------------------------------------------------')
        print('ITERATION - ', itr+1, '\n')

        time_prtitr_start = time.time()
        dti_itr,alpha_itr,preds_itr, classifier_itr = self.update_dti_qada(X, Dti, y, num_iterations, no_of_Q)
        
        # print(dti_itr)
        dti.append(dti_itr)
        self.alpha.append(alpha_itr)
        
        ## taking the predictions (ht) from 0,1 to -1,1 
        preds_mod = []
        for i in range(len(y)):
            if preds_itr[i]==0:
                preds_mod.append(-1)
            else:
                preds_mod.append(1)
        
        self.preds_mat.append(preds_mod)
        
        self.classifiers.append(classifier_itr)
        print('Alpha values -> ', self.alpha)
        final_bin, acc = self.final_bin_predictions(X, y_mod,preds_mod)
        print("New Binary labels : ", final_bin)
        print("New Accuracy : ", acc)
        accuracy_final.append(acc)
        
        Dti = dti_itr

        time_prtitr_end = time.time()
        
        print("->  Total time per iteration", time_prtitr_end - time_prtitr_start)

    
    plt.style.use('seaborn')
    plt.plot(list(range(num_iterations)), accuracy_final)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.show()
    
    return

  def predict(self, X ,y):
    
    # Predictions are made with the trained classifiers
    ht = []
    accuracy_final_test = []
    pred=[]
    
    T = self.num_iterations

    X = X[:2**self.size]
    y = y[:2**self.size]

    #change labels from 0,1 to -1,1
    h_mod = []
    
    y_mod = []
    for i in range(len(y)):
        if y[i]==0:
            y_mod.append(-1)
        else:
            y_mod.append(1)
            
            
    for t in range(T):

        d = self.classifiers[t].predict(X)
        from scipy.stats import mode 
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y, d)
        cm_argmax = cm.argmax(axis=0)
        if (cm_argmax[0] == cm_argmax[1]):
            correct_d = d
        else:
            correct_d = np.array([cm_argmax[i] for i in d])
        
        # print('-> Corrected predictions', correct_d)
        ht.append(correct_d)
        
        for i in range(len(ht[0])):
            if ht[t][i]==0:
                ht[t][i] = -1
            else:
                ht[t][i] = 1
    
    # now ht has all the particular values

    for t in range(T):
        pred.append(self.alpha[t]*ht[t])
    # print(pred)
    
    
    for t in range(T):  
        final_pred=[]
        # adding up all the alpha*ht
        predsum = np.sum(pred[0:t+1], axis = 0)
        # print(predsum)
        
        for i in range(len(X)):
            if predsum[i]>0:
                final_pred.append(1)
            else:
                final_pred.append(-1)

        
        accuracy_final_test.append(metrics.accuracy_score(y_mod,final_pred))
    print('-> Final Predictions: ',final_pred)
    
    plt.style.use('seaborn')
    plt.plot(list(range(T)), accuracy_final_test)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Testing Accuracy")
    plt.show()
    
    print('----------------------------------------------------------')
    print('Final Testing Accuracy : ', accuracy_final_test[-1])    
    print('----------------------------------------------------------')
    return
 


