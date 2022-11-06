from qiskit import *
from qiskit.algorithms import EstimationProblem
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms import AmplificationProblem
from qiskit.algorithms import Grover
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms import IterativeAmplitudeEstimation
from qiskit import Aer
from oracles import qreal_oracles
from learner import weaklearner
import numpy as np
import math
import time

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

      # appending the amplification circuit
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
      This circuit will estimate the partition label weights(the W's). 
      
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

  def A_qada(self, y, preds, Dti):
    '''
    This circuit will estimate the partition label weights(the W's). 
    
    '''
    w=Dti
    size = self.data_size(y)
    qr_xi = QuantumRegister(size, 'xi')
    qr_yi = QuantumRegister(1, 'yi')
    qr_Dti = QuantumRegister(4, 'dti')
    qr_ht = QuantumRegister(1, 'ht')
    
    qr_Dbk = QuantumRegister(4, 'dbk') # this is required as we're doing Dti amplification in a separate circuit

    qr_final_rot = QuantumRegister(1,  'final_rot')
    cr = ClassicalRegister(1)
    
    qc = QuantumCircuit(qr_xi,qr_yi, qr_Dti, qr_ht,qr_Dbk, qr_final_rot)
    
    qc.h(qr_xi)
    
    ## now we are adding up all the Oh Dbk Customs in one - yi - Dti - ht
    
    # list of all the qubits in order 
    listofqubits = []
    
    #yi
    listofqubits.append(qr_yi[0])
    
    #dti
    for i in range(qr_Dti.size):
        listofqubits.append(qr_Dti[i])
    
    #ht
    listofqubits.append(qr_ht[0])
    

    # Making a list in this order
    listofydj={}

    # print(str(flip_string(f2bin(Dti[i] ,qr_Dti.size))))
 
    for i in range(len(y)):
        
        #listofydj[dec_to_bin(i, qr_xi.size)] = str(dec_to_bin(preds[i],qr_ht.size)) + str(flip_string(f2bin(Dti[i] ,qr_Dti.size)[1:5])) +str(dec_to_bin(y[i],qr_yi.size))
        listofydj[self.dec_to_bin(i, qr_xi.size)] = str(self.dec_to_bin(preds[i],qr_ht.size)) + str(self.flip_string(self.float_to_bin(Dti[i] ,qr_Dti.size)[1:5])) +str(self.dec_to_bin(y[i],qr_yi.size))
    
    # print(listofydj)
    
    self.custom_oracle(qc,qr_xi,listofqubits,listofydj)
    
    # as we want to measure ht != yi
    qc.x(qr_ht)
    
    # copying dbkti
    for i in range(4):
        qc.mct([qr_Dti[i],qr_yi[0],qr_ht[0]],qr_Dbk[i])
    
    ## this will also have this part for making sure that the '00' state is also cared for 
    qc.x(qr_ht)
    qc.x(qr_yi)
    for i in range(4):
        qc.mct([qr_Dti[i],qr_yi[0],qr_ht[0]],qr_Dbk[i])
    qc.x(qr_yi)
    
    ## end of copying - note all the htis and yis should be what they have been originally encoded to be 
    
    
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


  def iteration_iqae_qada(self, y, preds,Dti):
    '''
    This function executes the circuit given above 
    and uses IQAE on it to get the values of W+ and W- 
    once these are obtained it returns the value of alphas 
    '''
    
    w = Dti
    wkb = []
    alpha_t = 0
    Zt = 0
        
    ## the iterative amplitude estimation
    iae = IterativeAmplitudeEstimation(
        epsilon_target=0.02,  # target accuracy
        alpha=0.05,  # width of the confidence interval
        quantum_instance=quantum_instance_qae)

        
    ## doing for label 0 
    qq = self.A_qada(y, preds,Dti)
        
    problem0 = EstimationProblem( 
    state_preparation=qq,  # A operator
    objective_qubits=[qq.width()-1],  # we want to see if the last qubit is in the state |1> or not!!
    grover_operator=None)

    est_amp0 = iae.estimate(problem0).estimation

    if(est_amp0!=0):
        wkb.append(est_amp0)
        den = est_amp0*8

    else:
        den = 1e-8 #smoothing
        # wn.append(den)

    # just in case epsilon t is going above 1 due to approximate values
    if den > 0.99:
        den = 0.99

    print("--------------------------------------------------------------------")
    print("epsilon_t = ", den)
    print("--------------------------------------------------------------------")

    return den


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

  def update_dti_qada(self, X, Dti, y, T, no_of_Q):
    '''
    Used for updation of Dti values using the updation rule by using Beta_j values
    '''

      # normalization of the Dti

    print("-> Dti updated:")
    print(Dti)

    Dti_norm = Dti/sum(Dti)


    Dti_arc_sin_sqrt = np.arcsin(np.sqrt(Dti_norm))


    # Amplification and Obtaining the Ht 

    reps = 3*round(math.log10(T)*np.sqrt(len(X)))

    # we pass unnormalized Dti to this as normalization will take place automatically inside it!
    time_start_amp = time.time()
    Dti_amp = self.amplification(Dti_arc_sin_sqrt, reps)
    time_end_amp = time.time()

    print("->  Dti values obatined after amplification :")
    print(Dti_amp)

    # running the classical classifier
    # this gives us the values of partitions obtained after training weights with w_i
    preds,classifier = self.weak_hypothesis_binary(X,y, Dti_amp, no_of_Q)

    print(preds)
        
    # estimation 

    time_start_iqae = time.time()
    eps_t = self.iteration_iqae_qada(y,  preds, Dti_arc_sin_sqrt)
    time_end_iqae = time.time()

    dti_up = []

    dti_itr = Dti

    delta = 1/(10*(no_of_Q * T*T))

    # updation of Dti 
    Q = no_of_Q
    ## here we will be taking the 'yes'/'no' condition into account
    if eps_t >= (1-delta)/(64*Q*T*T):
        # print('yes')
        alpha_t = np.log(np.sqrt(1-eps_t)/np.sqrt(eps_t))
        Zt = 2*np.sqrt((1-eps_t)*eps_t)
            
        for i in range(len(Dti)):
            if(y[i]==preds[i]):
                dti_up.append(dti_itr[i]*np.exp(-alpha_t)/((1+2*delta)*Zt))

            else:
                dti_up.append(dti_itr[i]*np.exp(alpha_t)/((1+2*delta)*Zt))

    else:
        # print('no')
        alpha_t = np.log(np.sqrt((Q*T*T) - 1))
        Zt = 2*(np.sqrt((Q*T*T) - 1)/(Q*T*T))
            
        for i in range(len(Dti)):
            if(y[i]==preds[i]):
                dti_up.append(dti_itr[i]*np.exp(-alpha_t)*(2 - 1/(Q*T*T))/((1+2/(Q*T*T))*Zt))

            else:
                dti_up.append(dti_itr[i]*np.exp(alpha_t)*(1/(Q*T*T))/((1+2/(Q*T*T))*Zt))

    print("-> Zt", Zt)
    print("-> alpha_t", alpha_t)

    # their normalization is not working 
    dti_up_norm = dti_up/sum(dti_up)

    print("->  Time for amplification", time_end_amp - time_start_amp )
    print("->  Time for Estimation", time_end_iqae - time_start_iqae )

    return dti_up_norm, alpha_t, preds, classifier
