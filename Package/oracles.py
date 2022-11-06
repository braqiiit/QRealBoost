from qiskit import *
from qiskit.algorithms import AmplificationProblem
from qiskit.algorithms import Grover
from qiskit.providers.aer import AerSimulator
from learner import weaklearner
from helpers import helper_functions
from qiskit import Aer

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

      # for the application of mct we need an array that takes in all the qubits from qr1... [qr1[0],qr1[1],qr1[2]...]
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
      the inverse of the function given above    
      '''
      
      reg1_len = qr1.size
      reg2_len = len(qr_list)
      data_size = len(data_dict)
      # for the application of mct we need an array that takes in all the qubits from qr1... [qr1[0],qr1[1],qr1[2]...]
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


      # Making a list in this order 
      listofydj={}

      # this list will be passed to the Oh_Dbk_custom function for the encoding of the |xi>|Dti> function 
      for i in range(len(Dti_asrt)):
          listofydj[self.dec_to_bin(i, qr_xi.size)] = str(self.flip_string(self.float_to_bin(Dti_asrt[i], qr_Dti.size)[1:5]))
      
      self.custom_oracle(qc,qr_xi,listofqubits,listofydj)
      
      qc = qc.compose(self.rot_circuit(),[qr_Dti[0],qr_Dti[1],qr_Dti[2],qr_Dti[3],qr_final_rot[0]])
      
      qcinv = QuantumCircuit(qr_xi, qr_Dti, qr_final_rot)
      self.custom_oracle_inv(qc, qr_xi,listofqubits,listofydj)
      
      return qc