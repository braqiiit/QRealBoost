import numpy as np
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