import numpy as np
import time
import pandas as pd
import math

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit import *
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.tools.jupyter import *
from qiskit.visualization import *

class helper_functions:

  def flip_string(self, x):
    return x[::-1]

  def numConcat(self, num1, num2):
    
      digits = len(str(num2))
      num2 = str(num2)
      num1 = str(num1)
      num1 += num2
    
      return num1

  def binaryToDecimal(self, binary):
      
      binary1 = binary
      decimal, i, n = 0, 0, 0
      while(binary != 0):
          dec = binary % 10
          decimal = decimal + dec * pow(2, i)
          binary = binary//10
          i += 1
      return decimal 


  def decimalToBinary(self,n,no_of_places):

      num = no_of_places 
      binary = bin(n).replace("0b", "")
      if (len(binary) != num):
          i = num - len(binary)
          for j in range(0,i):
              binary = self.numConcat(0,binary)

      return binary

  def dec_to_bin(self, n, size):

      bin_num = '0'*size
      b = bin(int(n)).replace("0b", "" )
      l = len(b)
      bin_num = bin_num[:size-l] + b
      return bin_num

  def float_to_bin(self, number,places): 
      '''
      used to convert a floating point number to its binary form
      '''
      import numpy
      if(type(number)==int or type(number)==numpy.int32 or type(number)==numpy.int64):

          return bin(number).lstrip("0b") + "."+"0"*places

      else:
          
          whole, dec = str(number).split(".") 
          whole = int(whole)
          dec = "0."+dec
          stri = ''
          res = bin(whole).lstrip("0b") + "."
          dec= float(dec)
          dec_val2 = dec
          num = dec
          countlen= 0
      
          while(dec_val2 != 0 and countlen <= places):

              num = float(num)*2
              arr = str(num).split(".")

              if (len(arr)==2):
                  whole1 = arr[0]
                  dec_val = arr[1]
              else:
                  whole1 = arr[0]
                  dec_val = '0'

              if whole1 == '0':
                  stri = stri + '0'
              else:
                  stri = stri+ '1'


              dec_val2 = float(dec_val)
              num = '0.'+dec_val
              countlen = len(stri)

          if (len(stri)<= places):
              stri = stri + '0'*(places - len(stri))
          elif(len(stri)>= places):
              stri = stri[:places]
          else:
              stri = stri

          s = bin(whole).lstrip("0b")+'.'+stri

      return s

  def data_size(self, X):
    '''
    currently the algorithm works on data size in powers of 2
    due to limitation on the numner of qubits, maximum 64 data samples can be used 

    '''
    size = int(math.log2(len(X)))

    if size>6:
      size=6

    return size

  def check_j(self):
    '''
    used for checking if the values of k and j are same or not 
    and then on the basis of that setting the auxilary qubit as True(1) or False(0)
    '''
    
    j_i = QuantumRegister(2,'j_i')
    k = QuantumRegister(2, 'k')
    qq = QuantumRegister(2, 'qq')
    i_1 = QuantumRegister(1,'i_1')

    qc = QuantumCircuit(j_i, k, qq, i_1, name = 'label j')
    
    qc.x(j_i[0])
    qc.x(j_i[1])
    
    qc.cx(j_i[0] ,k[0])
    qc.cx(j_i[1] ,k[1])
    
    qc.cx(k[0],qq[0])
    qc.cx(k[1],qq[1])

    qc.ccx(qq[0],qq[1],i_1)
    
    return qc
  
  def check_y(self):
      '''
      used for checking if the values of y and b are same or not 
      and then on the basis of that setting the auxilary qubit as True(1) or False(0)
      '''
      y = QuantumRegister(1,'j_i')
      b = QuantumRegister(1, 'b')
      i_2 = QuantumRegister(1,'i_2')

      qc = QuantumCircuit(b,y, i_2, name = 'check y')

      qc.cx(y, b)
      qc.cx(b, i_2)
      qc.x(i_2)
      
      return qc

  def new_qc(self, mc, data):
      '''
      Unitary which is able to produce a number encoded in the binary format for production of k and b
      '''
      qr_m = QuantumRegister(mc)    
      
      qc = QuantumCircuit(qr_m, name = 'init')
      
      ## each of data points should be smaller than 2**mc
      
      for i in range(0,len(data)):
          if data[i]>2**mc:
              print("Error! The value of the data to be stored is bigger then the 2**mc")
              return
      
      bin_data = ["" for x in range(len(data))]

      # the data needs to convert to binary from decimal
      for i in range(0,len(data)):
          bin_data[i] = self.decimalToBinary(data[i], mc)
          
      
      new_data = np.zeros([len(data), mc])
      
      for i in range(len(data)):
          for j in range(mc):
              new_data[i, j] = bin_data[i][j]
      
      # flipping the matrix around so the ordering is proper according QISKIT
      flip_new_data = np.flip(new_data,1)

      # this will be arranged in a row vector so that we can run a loop over it 
      new_data_row = np.reshape(flip_new_data,[1,mc*len(data)])
      
      for i in range(len(new_data_row[0])):
          if new_data_row[0,i] == 1:
              qc.x(qr_m[i])
              
      return qc

  def rot_circuit(self):
    '''
    used for doing conditional rotations on the values of Dti and Dbkti's
    Only 5 qubits are required as we are storing the values of Dti in these 4 qubits only.  
    '''
    
    theta = 1
    num_qubits = 5
    qc = QuantumCircuit(num_qubits, name = 'rot_circuit')
    qc.cry(theta/2,0,4)
    qc.cry(theta/4,1,4)
    qc.cry(theta/8,2,4)
    qc.cry(theta/16,3,4)

    return qc

from sklearn.cluster import KMeans
import sklearn
from sklearn import metrics

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

from qiskit.algorithms import AmplificationProblem
from qiskit.algorithms import Grover
from qiskit.providers.aer import AerSimulator

# backend = AerSimulator()
backend = Aer.get_backend('aer_simulator')

class qreal_oracles(helper_functions):

  def custom_oracle(self, qc, qr1, qr_list, data_dict):
      '''
      This oracle is used for entangling a particular value |Ki> with |xi>, by performing a transformation given by 
      
      |xi>|0> --> |xi>|Ki>
      
      this is used for encoding the |Dti>,|yi> and |jti>
      
      '''
      
      reg1_len = qr1.size
      reg2_len = len(qr_list)
      data_size = len(data_dict)

      # for application of mct we need an array which takes in all the qubits from qr1... [qr1[0],qr1[1],qr1[2]...]
      qr1_arr = []
      
      for i in range(reg1_len):
          qr1_arr.append(qr1[i])
      
      # application of the main gates that there are 
      
      for i in range(data_size):
          string1 = self.flip_string(list(data_dict.keys())[i])
          string2 = self.flip_string(list(data_dict.values())[i])

          # The oracle looks at the all the values of xi and the corresponding jti that needs to be attatched 
          # Then it makes the state 11111.. and applies an mct to make this control and target is applied on jti
          # Finally X is applied to revert the states back to the original xi's
          
          for j in range(len(string1)):
              if string1[j] == '0':
                  qc.x(qr1[j])
          
          for j in range(len(string2)):
              if string2[j] == '1':
                  qc.mct(qr1_arr, qr_list[j])
          
          for j in range(len(string1)):
              if string1[j] == '0':
                  qc.x(qr1[j])



  def custom_oracle_inv(self, qc, qr1, qr_list, data_dict):
      '''
      inverse of the function given above    
      '''
      
      reg1_len = qr1.size
      reg2_len = len(qr_list)
      data_size = len(data_dict)
      # for application of mct we need an array which takes in all the qubits from qr1... [qr1[0],qr1[1],qr1[2]...]
      qr1_arr = []
      
      for i in range(reg1_len):
        qr1_arr.append(qr1[i])
      
      # application of the main gates that there are 
      
      for i in range(data_size):
        string1 = self.flip_string(list(data_dict.keys())[data_size-i-1])
        string2 = self.flip_string(list(data_dict.values())[data_size-i-1])

        
        for j in range(len(string1)):
          if string1[j] == '0':
              qc.x(qr1[j])
        
        for j in range(len(string2)):
          if string2[j] == '1':
              qc.mct(qr1_arr, qr_list[j])
        
        for j in range(len(string1)):
          if string1[j] == '0':
              qc.x(qr1[j])



  def initial_circuit(self, size, Dti_asrt):

      '''
      This creates the initial circuit A for the amplification circuit
      using the custom oracle functions for encoding and entangling
      '''

      qr_xi = QuantumRegister(size, 'xi')
      qr_Dti = QuantumRegister(4, 'dti')
      qr_final_rot = QuantumRegister(1,  'final_rot')
      
      qc = QuantumCircuit(qr_xi, qr_Dti, qr_final_rot)
      

      qc.h(qr_xi)
      
      # now we are adding up all the Oh Dbk Customs in one - Dti 
      
      # list of all the qubits in order 
      listofqubits = []

      #dti
      for i in range(qr_Dti.size):
          listofqubits.append(qr_Dti[i])


      # making a list in this order 
      listofydj={}

      # this list will be passed to the Oh_Dbk_custom function for the encoding of the |xi>|Dti> function 
      for i in range(len(Dti_asrt)):
          listofydj[self.dec_to_bin(i, qr_xi.size)] = str(self.flip_string(self.float_to_bin(Dti_asrt[i], qr_Dti.size)[1:5]))
      
      self.custom_oracle(qc,qr_xi,listofqubits,listofydj)
      
      qc = qc.compose(self.rot_circuit(),[qr_Dti[0],qr_Dti[1],qr_Dti[2],qr_Dti[3],qr_final_rot[0]])
      
      qcinv = QuantumCircuit(qr_xi, qr_Dti, qr_final_rot)
      self.custom_oracle_inv(qc, qr_xi,listofqubits,listofydj)
      
      return qc

from qiskit.algorithms import EstimationProblem
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms import IterativeAmplitudeEstimation
backend = Aer.get_backend('aer_simulator')
quantum_instance_qae = QuantumInstance(backend,shots = 10)

class amplification_updation(qreal_oracles):

  def amplification(self, Dti, reps):
      '''
      Creation of the amplification circuit to obtain the values of Dti 
      after the amplification has been done
      '''

      size = self.data_size(Dti)
      Ainit = self.initial_circuit(size, Dti)

      w = Ainit.width()
      # oracle
      qo = QuantumCircuit(w)
      qo.z(w - 1)
      oracle =  qo

      problem = AmplificationProblem(oracle, state_preparation=Ainit)
      
      n = Ainit.width()
      qc = QuantumCircuit(n)
      qc  = qc.compose(Ainit)

      G = problem.grover_operator
      
      for rep in range(reps):
          qc = qc.compose(G)
          
      
      cr = ClassicalRegister(qc.width() - 5) # here 4 is for the number of Dti's used +1 is for rot qubit
      qr = QuantumRegister(qc.width())
      qc_qaa = QuantumCircuit(qr, cr)

      # appendiing the amplification circuit
      qc_qaa.append(qc, qr)

      # the qubits to be measured
      meas_qr = []
      meas_cr = []
      for i in range(qc.width() - 5):
          meas_qr.append(qr[i])
          meas_cr.append(cr[i]) 


      qc_qaa.measure(meas_qr,meas_cr)
      backend = AerSimulator()
      shots = 1500
      result = execute(qc_qaa, backend, shots = shots).result().get_counts()

      listofnew_dti = {}
      for i in range(len(result.keys())):
          listofnew_dti[(list(result.keys())[i])] = (list(result.values())[i]/shots)

      sorted_dict = listofnew_dti.keys()
      sorted_dict = sorted(sorted_dict) # ascending order sorting of keys

      # Dictonary to store the values of Dti in sorted ordering 
      dti_sort = []
      for i in range(len(listofnew_dti.keys())):
          dti_sort.append((listofnew_dti[sorted_dict[i]]))

          
      return dti_sort


  # Estimating partition label weigths 


  def A(self, y ,k , b, preds, Dti):
      '''
      This circuit will estitmate the partition label weigths(the W's). 
      
      '''
      w = Dti
      size = self.data_size(y)
      qr_xi = QuantumRegister(size, 'xi')
      qr_yi = QuantumRegister(1, 'yi')
      qr_Dti = QuantumRegister(4, 'dti')
      qr_jti = QuantumRegister(2, 'jti')
      qr_i_1 = QuantumRegister(1,'i_1')# For I1 and I2
      qr_i_2 = QuantumRegister(1,'i_2')
      qr_kk = QuantumRegister(2,'k') # For initialization of different k and b
      qr_b = QuantumRegister(1,'b')
      qr_Dbk = QuantumRegister(4, 'dbk')
      qr_qq = QuantumRegister(2, 'qq')
      qr_final_rot = QuantumRegister(1,  'final_rot')
      cr = ClassicalRegister(1)
      
      qc = QuantumCircuit(qr_xi,qr_yi, qr_Dti, qr_jti, qr_i_1, qr_i_2, qr_kk, qr_b,qr_Dbk, qr_qq, qr_final_rot)
      
      qc.h(qr_xi)
      
      # now we are summing up all the Custom Oracles Oh for respective Dbk in one - yi - Dti - jti
      
      # list of all the qubits in order 
      listofqubits = []
      
      #yi
      listofqubits.append(qr_yi[0])
      
      #dti
      for i in range(qr_Dti.size):
          listofqubits.append(qr_Dti[i])
      
      #jti
      for i in range(qr_jti.size):
          listofqubits.append(qr_jti[i])
      

      # making a list in this order
      listofydj={}

      for i in range(len(y)):
          
          listofydj[self.dec_to_bin(i, qr_xi.size)] = str(self.dec_to_bin(preds[i],qr_jti.size)) + str(self.flip_string(self.float_to_bin(Dti[i] ,qr_Dti.size)[1:5])) +str(self.dec_to_bin(y[i],qr_yi.size))
      
      self.custom_oracle(qc,qr_xi,listofqubits,listofydj)
      
      qc = qc.compose(self.new_qc(2,[k]), [qr_kk[0],qr_kk[1]])
      qc = qc.compose(self.new_qc(1,[b]), [qr_b[0]])
      qc = qc.compose(self.check_j(),[qr_jti[0],qr_jti[1],qr_kk[0],qr_kk[1],qr_qq[0],qr_qq[1],qr_i_1[0]])
      qc = qc.compose(self.check_y(), [qr_yi[0], qr_b[0], qr_i_2[0]])
      
      # copying dbkti
      for i in range(4):
          qc.mct([qr_Dti[i],qr_i_1[0],qr_i_2[0]],qr_Dbk[i])
      
      qc = qc.compose(self.rot_circuit(),[qr_Dbk[0],qr_Dbk[1],qr_Dbk[2],qr_Dbk[3],qr_final_rot[0]])
      

      return qc

  def iteration_iqae(self, y, preds,Dti):
      '''
      This function executes the cirucit given above 
      and uses IQAE on it to get the values of W+ and W- 
      once these are obtained it returns the value of betas 
      '''
      
      w = Dti
      wkb = []
      wp = []
      wn = []
      beta_j = []
      Zt = 0
          
      # the iterative amplitude estimation
      iae = IterativeAmplitudeEstimation(
          epsilon_target=0.03,  # target accuracy
          alpha=0.07,  # width of the confidence interval
          quantum_instance=quantum_instance_qae,
      )
      
      Ztj = []

      for k in range(0,3):
        print('k value -', k)

        # For label 0 (b=0) 
        q0 = self.A(y,k,0, preds,Dti)
        
  
        problem0 = EstimationProblem( 
        state_preparation = q0,  # A operator
        objective_qubits = [q0.width()-1],  # we want to see if the last qubit is in the state |1> or not!
        grover_operator = None)
          
        est_amp0 = iae.estimate(problem0).estimation

        if(est_amp0!=0):

            wkb.append(est_amp0)
            den = est_amp0

        else:
            den = 1e-8 #smoothing
            wn.append(den)


        # For label 1 (b=1) 
        q1 = self.A(y, k, 1, preds, Dti)

        problem1 = EstimationProblem(
        state_preparation=q1,  # A operator
        objective_qubits=[q0.width()-1],  # we want to see if the last qubit is in the state |1> or not!!
        grover_operator=None
        )

        est_amp1 = iae.estimate(problem1).estimation
      
        if(est_amp1!=0):

            wkb.append(est_amp1)
            num = est_amp1


        else:
            num = 1e-8 #smoothing
            wp.append(num)

        b = (1/2)*np.log(num/den)

        beta_j.append(b)

            ## testing 
          
        ## for Z
        Ztj.append(np.sqrt(den*num))
        print("w for b =0,", den)
        print("w for b =1," , num)
      
      Zt= 2*sum(Ztj)
      
      return beta_j, Zt

  def update_dti(self, X, Dti, y, no_of_Q = 4):
      '''
      Used for updation of Dti values using the updation rule by using Beta_j values
      '''
      
      # normalization of the Dti
      Dti_norm = Dti/sum(Dti)
      Dti_arc_sin_sqrt = np.arcsin(np.sqrt(Dti_norm))

      ''' 
      Amplification and Obtaining the Ht 
      '''
      
      time_start_amp = time.time()

      # Unormalized Dti are passed to the amplification function as normalization will take place inside it
      Dti_amp = self.amplification(Dti_arc_sin_sqrt, 2*round(np.log(25)*np.sqrt(len(X))))
      time_end_amp = time.time()

      print("->  Dti values obatined after amplification :")
      print(Dti_amp)

      # running the classical classifier
      # this gives us the values of partitions obtained after training weights with w_i
      preds,classifier = self.weak_hypothesis(X, Dti_amp, no_of_Q)
      

      '''
      Estimation
      '''
      time_start_iqae = time.time()
      beta_j, Zt = self.iteration_iqae(y,  preds, Dti_arc_sin_sqrt)
      time_end_iqae = time.time()

      dti_up = [] # To store updated Dti values
      dti_copy = Dti # Copying the original Dti values
      final_beta = []
      
      # Changing labels from 0,1 to -1,1 
      # Since iteration_iqae cannot take -1 
      y_mod = []
      for i in range(len(y)):
          if y[i]==0:
              y_mod.append(-1)
          else:
              y_mod.append(1)
      
      '''
      Updation of Dti
      ''' 

      for i in range(len(Dti)):
          dti_up.append(dti_copy[i]*np.exp((-1)*y[i]*beta_j[preds[i]]))
          final_beta.append(beta_j[preds[i]])

      print("->  Time for amplification", time_end_amp - time_start_amp )
      print("->  Time for Estimation", time_end_iqae - time_start_iqae )
              
      return dti_up,beta_j, final_beta, preds, classifier, Zt

import matplotlib.pyplot as plt
from matplotlib import style

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

    return

  def predict(self, X, y):
    '''
    here beta_j is the function which contains the betas in the form - 
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