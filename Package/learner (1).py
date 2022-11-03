# -*- coding: utf-8 -*-
"""learner.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KphTDZXom_mcvC7_SYiD9ll8xXi9792y
"""

from sklearn.cluster import KMeans
import sklearn
from sklearn import metrics
from helpers import helper_functions

class weaklearner:

  def weak_hypothesis(self, X, Dti, no_of_Q):
      '''
      This function is used for returning the partitioning of the X's
      no_of_Q : represents the number of samples that must be choosen from all the M samples
      '''
      # we will extract the Q values with top Dti's
      Dti = np.array(Dti)
      ind_max = np.argpartition(Dti, -no_of_Q)[-no_of_Q:]

      no_of_partitions = 3
      km = KMeans(
          n_clusters = no_of_partitions, init='random',
          n_init=10, max_iter=300,
          tol=1e-04, random_state=0)

      # now we will pass the corresponding X and parts with the Q samples to train the model 
      fitted_km = km.fit(X[ind_max])
      # prediction will be obtained for all the samples
      prediction = fitted_km.predict(X)
      d = prediction


      return  d, fitted_km


  def original_acc(self, X,y, no_of_Q):
    '''
    Function for checking the accuracy of the classifier in the first iteration
    '''
    Dti = np.full(len(X),1/len(X))
    dti = Dti
    
    preds, cls = self.weak_hypothesis(X, X, Dti, no_of_Q)
    
    dti0_0 = []
    dti0_1 = []
    dti1_0 = []
    dti1_1= []
    dti2_0 = []
    dti2_1 = []


    for i in range(len(preds)):
        if preds[i] == 0 and y[i] == 0:
            dti0_0.append(dti[i])
        if preds[i] == 0 and y[i] == 1:
            dti0_1.append(dti[i])
        if preds[i] == 1 and y[i] == 0:
            dti1_0.append(dti[i])
        if preds[i] == 1 and y[i] == 1:
            dti1_1.append(dti[i])
        if preds[i] == 2 and y[i] == 0:
            dti2_0.append(dti[i])
        if preds[i] == 2 and y[i] == 1:
            dti2_1.append(dti[i])
            
            
    if sum(dti0_0) >= sum(dti0_1):
        y0 = 0
    else:
        y0 = 1
    if sum(dti1_0) >= sum(dti1_1):
        y1 = 0
    else:
        y1 = 1
    if sum(dti2_0) >= sum(dti2_1):
        y2 = 0
    else:
        y2 = 1
        
        
    # Binary labels obtained from weak classifier
    final_y = []
    
    for i in range(len(preds)):
        if preds[i] == 0:
            final_y.append(y0)
        if preds[i] == 1:
            final_y.append(y1)
        if preds[i] == 2:
            final_y.append(y2)
            
    acc = metrics.accuracy_score(final_y, y)

    return acc


  def original_distribution(self, parts, y, dti):
      '''
      This function will tell us about the original distribution of the classification data
      '''
      dti0_0 = []
      dti0_1 = []
      dti1_0 = []
      dti1_1= []
      dti2_0 = []
      dti2_1 = []


      for i in range(len(parts)):
          if parts[i] == 0 and y[i] == 0:
              dti0_0.append(dti[i])
          if parts[i] == 0 and y[i] == 1:
              dti0_1.append(dti[i])
          if parts[i] == 1 and y[i] == 0:
              dti1_0.append(dti[i])
          if parts[i] == 1 and y[i] == 1:
              dti1_1.append(dti[i])
          if parts[i] == 2 and y[i] == 0:
              dti2_0.append(dti[i])
          if parts[i] == 2 and y[i] == 1:
              dti2_1.append(dti[i])


      # Classically calculated Dti for cross checking
      print("0,0 -" , len(dti0_0), "sum - ", sum(dti0_0))
      print("0,1 -" , len(dti0_1), "sum - ", sum(dti0_1))
      print("1,0 -" , len(dti1_0), "sum - ", sum(dti1_0))
      print("1,1 -" , len(dti1_1), "sum - ", sum(dti1_1))
      print("2,0 -" , len(dti2_0), "sum - ", sum(dti2_0))
      print("2,1 -" , len(dti2_1), "sum - ", sum(dti2_1))