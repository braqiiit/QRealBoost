#!/usr/bin/env python
# coding: utf-8

# # Q Real Boost implementation 
# 
# New boosting implementation without QRAM, and with estimation and all for conference

# In[1]:


from qiskit import *


# In[29]:


import numpy as np
import time
import pandas as pd
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit import *
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.tools.jupyter import *
from qiskit.visualization import *
#from ibm_quantum_widgets import *
from qiskit import BasicAer
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit import IBMQ
from sklearn.cluster import KMeans
import sklearn
from sklearn import svm

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier


# ### Helper functions

# In[3]:


def flip_string(x):
    return x[::-1]


# In[4]:


def check_j():
    '''
    use for checking if the value of k and j is same or not and then on the basis of that setting the auxilary qubit as True or False
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

#     qc.cx(j_i[0] ,k[0])
#     qc.cx(j_i[1] ,k[1])
    
#     qc.x(j_i[0])
#     qc.x(j_i[1])
    
    return qc


# In[5]:


def check_y():
    '''
    use for checking if the value of y and b is same or not and then on the basis of that setting the auxilary qubit as True or False
    '''
    y = QuantumRegister(1,'j_i')
    b = QuantumRegister(1, 'b')
    i_2 = QuantumRegister(1,'i_2')

    qc = QuantumCircuit(b,y, i_2, name = 'check y')

    qc.cx(y, b)
    qc.cx(b, i_2)
    qc.x(i_2)
    
#     qc.cx(y, b)
    
    return qc


# In[6]:


def new_qc( mc, data):
    '''
    Unitary which is able to produce a number encoded in the binary format for production of k and b
    '''
    qr_m = QuantumRegister(mc)    
    
    qc = QuantumCircuit(qr_m, name = 'init')
    
    ## each of data points should be smaller than 2**mc
    
    for i in range(0,len(data)):
        if data[i]>2**mc:
            print("Error!! The value of the data to be stored is bigger then the 2**mc")
            return
    
    bin_data = ["" for x in range(len(data))]
    ## the data needs to convert to binary from decimal
    for i in range(0,len(data)):
        bin_data[i] = decimalToBinary(data[i], mc)
        
    
    new_data = np.zeros([len(data), mc])
    
    # now we will be dividing all our divided
    for i in range(len(data)):
        for j in range(mc):
            new_data[i, j] = bin_data[i][j]
    
    ## fliping the matrix around so the ordering is proper according QISKIT
    flip_new_data = np.flip(new_data,1)
    ## this will be arranged in a row vector so that we can run a loop over it 
    new_data_row = np.reshape(flip_new_data,[1,mc*len(data)])
    
    for i in range(len(new_data_row[0])):
        if new_data_row[0,i] == 1:
            qc.x(qr_m[i])
            
    return qc


# In[7]:


def binaryToDecimal(binary):
     
    binary1 = binary
    decimal, i, n = 0, 0, 0
    while(binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary//10
        i += 1
    return decimal 
    
def numConcat(num1, num2): # this should actually do all the additions in the form of strings and then when you finally
                           # take out whatever is stored in the matrix then you should actually convert that to int
  
     # find number of digits in num2
    digits = len(str(num2))
    num2 = str(num2)
    num1 = str(num1)
  
     # add zeroes to the end of num1
#     num1 = num1 * (10**digits)
  
     # add num2 to num1
    num1 += num2
  
    return num1

## for convertign from decimal to binary 
def decimalToBinary(n,no_of_places):
    num = no_of_places ## this will be equal to mc
    binary = bin(n).replace("0b", "")
    if (len(binary) != num):
        i = num - len(binary)
        for j in range(0,i):
            binary = numConcat(0,binary)

    return binary


# In[8]:


def dec_to_bin(n, size):
    bin_num = '0'*size
    b = bin(int(n)).replace("0b", "" )
    l = len(b)
    bin_num = bin_num[:size-l] + b
    return bin_num


# In[9]:


def f2bin(number,places): 
    '''
    
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


# #### Conditional Rotations
# 
# The function given below is used to do conditional rotations. 

# In[10]:


def rot_circuit():
    '''
    used for doing conditional rotations on the values of Dti and Dbkti's 
    '''
    # here only 5 qubits are required as we are storing the values of Dti in these 4 qubits only. 
    theta = 1#np.pi
    num_qubits = 5
    qc = QuantumCircuit(num_qubits, name = 'rot_circuit')
    qc.cry(theta/2,0,4)
    qc.cry(theta/4,1,4)
    qc.cry(theta/8,2,4)
    qc.cry(theta/16,3,4)


    return qc


# # Data set
# 
# Dataset will be extracted in a different file and we will use the stored arrays here. Look ito Breast cancer test train split file. 

# ### Breast Cancer data set
# 
# look at sklearn documentation for more information regarding this data set!!

# In[95]:



## from now on we will 
X64 = np.array([[-0.80589907, -0.59042291, -0.05324325,  0.10774176, -0.04505932],
       [ 0.06256734, -0.37244006,  0.0939138 , -0.2033949 , -0.94811908],
       [-0.84338317, -0.6326198 , -0.43462352,  0.24523402, -0.52480534],
       [ 0.13286258,  0.37155658, -0.87001548,  0.70985865,  0.6677744 ],
       [-0.75234354, -0.21787303,  0.12046462, -0.59419401, -0.56680623],
       [ 0.28907508,  0.9888873 , -0.36947398, -0.3852527 ,  0.12494092],
       [-0.94193876, -0.07222975, -0.51910519, -0.34613681, -0.50121778],
       [-0.63461962, -0.66262997, -0.01359157, -0.20885209,  0.32881507],
       [ 0.26177535,  0.73210506,  0.14869942, -0.08739544, -0.41974897],
       [-0.87627756, -0.05459176, -0.79168413, -0.44217547, -0.76591014],
       [-0.31582251, -0.65735629,  0.80118443, -0.14383058,  0.05621122],
       [-0.56914731,  0.02827011, -0.70470652, -0.62345748,  0.07444681],
       [ 0.35079842,  0.84817601, -0.38494076, -0.29560828, -0.07015869],
       [ 0.60451149, -0.77078297, -0.00920281,  0.17409424, -0.29259797],
       [ 0.23017942,  0.13002239,  0.1935483 ,  0.2473952 , -0.39510758],
       [ 0.94377071, -0.63914724, -0.44823675, -0.35877896, -0.2279727 ],
       [-0.41203935, -0.63002251, -0.25803975, -0.13481458,  1.        ],
       [-0.84141311, -0.37295448,  0.23098801, -0.09279513, -0.30896296],
       [-0.63473418, -0.64862994, -0.34478505,  0.30493717, -0.31294345],
       [ 0.54624343, -0.05513775, -0.37989372,  0.44105009, -0.54167204],
       [ 0.41786759,  0.90609941, -0.47142637, -0.12383133,  0.07869459],
       [ 0.83676509, -0.60596934, -0.56257506, -1.        , -0.12157583],
       [ 0.09787885,  0.41570962,  0.05702348,  0.7133461 , -0.32991341],
       [ 0.56101992,  0.62038019, -0.28584764,  0.05462626, -0.96150592],
       [ 0.49179288, -0.1536601 , -0.20058942,  1.        , -0.23212819],
       [ 0.42401093, -0.29381199,  0.02061487,  0.68741985, -0.63409955],
       [-0.67235847, -0.68699306, -1.        ,  0.12998496, -0.64246139],
       [ 0.68981195, -0.7822776 , -0.45056316,  0.17824033, -0.28578779],
       [-0.8566584 , -0.04364493, -0.26908778, -0.35775043, -0.87088248],
       [-0.09183266,  0.77482135,  0.05847358, -0.03713341, -0.63098033],
       [ 0.82501212, -0.80820505, -0.18331782,  0.43321777, -0.49106175],
       [ 0.39893683,  0.94363918, -0.45686527,  0.24407125, -0.05909602],
       [ 0.69339013, -0.91467442, -0.23083126, -0.61828926, -0.31150299],
       [ 0.1360446 ,  0.4176652 ,  0.16521813,  0.28788609, -0.51930635],
       [ 0.44215386,  0.98543464, -0.37949666, -0.41211015,  0.05908244],
       [ 0.24224883,  0.81801481, -0.1429181 , -0.66253758, -0.39744601],
       [ 0.78826496, -0.72251302, -0.07685949, -0.01906448, -0.81606224],
       [ 0.64234617, -0.90884405, -0.01065299, -0.14171669, -0.5395039 ],
       [ 0.78331642, -0.79602718, -0.29497289, -0.38296365, -0.25876049],
       [ 0.58122718, -0.50820971, -0.39361292, -0.24785874,  0.49198399],
       [ 0.82427747, -0.77037464, -0.33859388,  0.59875024, -0.54046202],
       [-0.15491631,  0.42594546,  0.28120175, -0.12650848, -0.39095192],
       [-0.41265761, -0.778387  ,  0.38536974, -0.56097033,  0.27519472],
       [ 0.81281306, -0.82835321, -0.43001264,  0.54553452, -0.37301733],
       [-0.79622109, -0.04406497, -0.92528577, -0.37335828, -0.85511111],
       [ 0.12804201,  0.86948804, -0.01275777,  0.10466105,  0.11716134],
       [ 0.6583697 , -0.90448218, -0.07580081, -0.24559957, -0.43876332],
       [ 0.16191584,  0.4965871 ,  0.0479586 ,  0.21035375,  0.15673964],
       [ 0.30857441,  0.23363984,  0.06129492,  0.15414908,  0.07127983],
       [ 0.74245608, -0.6790161 , -0.23658274, -0.83579257, -0.14014044],
       [-0.15075894, -0.46560074,  0.50815128, -0.2837593 , -0.29533477],
       [-0.34931682,  0.03215297,  0.8454444 , -0.01232477, -0.14328294],
       [-0.85718266, -0.35816008, -0.53285026,  0.14980072, -0.38222436],
       [ 0.25723461,  0.63688708, -0.28527578, -0.29904661, -0.44305987],
       [ 0.17414132,  0.58749761,  0.09322319, -0.53893208, -0.92804504],
       [-0.39998583, -0.20541629,  0.78988065,  0.12372396, -0.14401889],
       [-0.66096707, -0.60768684, -0.08385178,  0.40744248, -0.24225684],
       [-0.87921722, -0.71534639, -0.43692453,  0.19842837,  0.1174938 ],
       [-0.81297545, -0.46532236, -0.31752754, -0.17918259,  0.04073442],
       [-0.91092209, -0.07973995, -0.33283798,  0.32994368, -0.60723175],
       [ 0.89593489, -1.        , -0.18398778, -0.54238746, -0.40547704],
       [-0.94852912,  0.07138546,  0.01882709, -0.01040221, -0.87775444],
       [-0.81245264, -0.36296099, -0.96669405,  0.22999089,  0.05613461],
       [-0.33243217, -0.33300052,  1.        , -0.05262967, -0.07337535]])

X_test64 = np.array([[-0.68707325, -0.50735238, -0.20700664, -0.62336223,  0.39806279],
       [ 0.46234745,  0.81775949, -0.44946658,  0.1140975 , -0.17951633],
       [ 0.61516826,  0.29117629, -0.20364858,  0.46090612,  0.00305875],
       [-0.50432232, -0.64050608,  0.13880843, -0.42755397,  0.47996396],
       [-0.58363044, -0.59302057, -0.36068798, -0.20546695,  0.42447254],
       [-0.77781706, -0.52822617,  0.1248982 , -0.39869912, -0.65804837],
       [-0.31182115, -0.16652907, -0.02315063,  0.05638778, -0.47035166],
       [-0.35714232, -0.56180839,  0.02707521, -0.3134225 ,  0.7454719 ],
       [ 0.40935228,  1.        , -0.52272328,  0.10199332,  0.13573632],
       [-0.5209226 , -0.19721416, -0.90344255, -0.43913667, -0.10011288],
       [-0.40235592, -0.6796702 , -0.49861898, -0.07365373,  0.85654452],
       [ 0.81742342, -0.54015171, -0.13383637, -0.1039951 , -0.93044641],
       [ 0.3254841 , -0.50934448,  0.04402021,  0.36110195, -0.37572541],
       [-0.09288116, -0.13640449,  0.05706831,  0.16841045, -0.95817001],
       [-0.90998517, -0.40430919, -0.86355457, -0.26217541, -0.064167  ],
       [-0.57775116, -0.76387782,  0.29964566, -0.30444234,  0.54161409],
       [ 0.21042602,  0.53453165,  0.05184494, -0.23258185, -0.75751172],
       [ 0.48320202,  0.85674321, -0.33808615, -0.17128284, -0.28369987],
       [-0.19770072, -0.68336291,  0.18640211, -0.35917056,  0.8291267 ],
       [ 0.65655214, -0.73443385, -0.42628468,  0.60888795,  0.01226754],
       [ 0.50016485,  0.0690652 , -0.45885945,  0.62442559,  0.06448148],
       [-0.38902654, -0.43851828,  0.66980889, -0.02364348,  0.01094794],
       [ 0.40796953, -0.85934891,  0.02097323, -0.25803196, -0.53040584],
       [-0.4956964 , -0.56031991,  0.74363441, -0.1163954 , -0.08862714],
       [-0.10882911,  0.64180596,  0.1375903 , -0.111592  , -0.30924287],
       [ 0.28332875,  0.79302765, -0.3118484 , -0.04220763, -0.12566824],
       [ 0.42676069, -0.61204992,  0.05231732, -0.25486205, -0.61785986],
       [ 0.60709358, -0.95289575, -0.18889892, -0.39078964, -0.212501  ],
       [-0.64455773, -0.20147303,  0.56736837, -0.29192325, -0.55664189],
       [ 0.45437401,  0.54042794, -0.13070526,  0.12284947, -0.55259124],
       [ 0.57638579, -0.67840819, -0.07071206, -0.21873946, -0.29933831],
       [-0.70554067, -0.51734514,  0.30348356, -0.5745156 , -0.12432887],
       [-0.51785336, -0.58458137,  0.53423814,  0.03205322, -0.05940223],
       [ 0.66912668, -0.78523877,  0.14110695, -0.00521186, -0.60630297],
       [-0.74108656, -0.13630257, -0.9809313 , -0.48456012, -0.29452577],
       [-0.64419453, -0.72115311, -0.14011673,  0.03085746, -0.41771499],
       [ 0.22878316, -0.07614892, -0.18493723,  0.4526321 , -0.1085523 ],
       [-0.8218347 , -0.28596485, -0.43286115, -0.48878151,  0.16100587],
       [ 0.53267724,  0.99697226, -0.51289751,  0.12752771, -0.36206251],
       [-0.46209981, -0.66716044, -0.10202221, -0.23609424,  0.47423212],
       [ 0.17014884,  0.40169187, -0.07279546,  0.04706547,  0.19721313],
       [ 0.59520936, -0.63509737, -0.39774277,  0.40608186, -0.057771  ],
       [-0.64006887, -0.33616701,  0.46128481, -0.0703224 , -0.05672336],
       [ 0.26505329, -0.21119132,  0.05232592,  0.21870707, -0.5844182 ],
       [ 0.25674746,  0.17546479,  0.12126193,  0.12813838, -0.9701474 ],
       [-0.46393382, -0.4577407 ,  0.69860785, -0.43353711, -0.15944243],
       [-0.58917778, -0.56371509,  0.41512294, -0.27969881, -0.02538025],
       [-0.66348932, -0.51538817,  0.25466218, -0.00358557, -0.35842899],
       [ 0.56951207, -0.68392313, -0.02628225,  0.17040521, -0.83197979],
       [-1.        , -0.32924987, -0.69087141, -0.44459176, -0.57811392],
       [ 1.        , -0.63579192, -0.49398102, -0.51258059, -0.30743351],
       [-0.83693098,  0.07785999, -0.41222783, -0.26083857, -0.66387623],
       [-0.05528876, -0.62425213,  0.25233409, -0.37310216,  0.73313602],
       [ 0.14568479,  0.43524995,  0.07727313,  0.0157491 , -0.53613886],
       [-0.02228015,  0.24813905,  0.03025536, -0.12589894, -0.97440804],
       [ 0.17895445,  0.64919227, -0.04496854, -0.23686312, -0.82210064],
       [-0.37993654, -0.45300412,  0.372467  , -0.02913794, -0.337318  ],
       [ 0.75550289, -0.85494843, -0.310949  , -0.81275764,  0.0746331 ],
       [ 0.38994431,  0.89259644, -0.24342355, -0.04604866, -0.03464256],
       [-0.73642587, -0.5752054 , -0.1700719 , -0.25048668,  0.33491302],
       [-0.73061136, -0.65033171, -0.55164925,  0.07918039,  0.44719878],
       [ 0.54081739,  0.17102128, -0.14932038,  0.52861686, -0.73708794],
       [ 0.46486667, -0.112343  ,  0.16271583, -0.00918502, -1.        ],
       [-0.74623427, -0.57101018,  0.00589433, -0.21381337,  0.34660433]])

y64 = [0,
 1,
 0,
 1,
 0,
 1,
 0,
 0,
 1,
 0,
 0,
 0,
 1,
 1,
 1,
 1,
 0,
 0,
 0,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 0,
 1,
 0,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 0,
 1,
 1,
 0,
 1,
 0,
 1,
 1,
 1,
 1,
 1,
 0,
 0,
 0,
 1,
 1,
 0,
 0,
 0,
 0,
 0,
 1,
 0,
 0,
 0]

y_test64 = [0,
 1,
 1,
 0,
 0,
 0,
 1,
 0,
 1,
 0,
 0,
 1,
 1,
 1,
 0,
 0,
 1,
 1,
 0,
 1,
 1,
 0,
 1,
 0,
 1,
 1,
 1,
 1,
 0,
 1,
 1,
 0,
 0,
 1,
 0,
 0,
 1,
 0,
 1,
 0,
 1,
 1,
 0,
 1,
 1,
 0,
 0,
 0,
 1,
 0,
 1,
 0,
 0,
 1,
 1,
 1,
 0,
 1,
 1,
 0,
 0,
 1,
 1,
 0]

X = X64[:32]
X_test = X_test64[:32]
y = y64[:32] 
y_test = y_test64[:32]

# here we will also define the vallue of Q
no_of_Q = 4


# In[96]:


# y


# In[97]:


# original_acc(X,y,4)


# In[98]:


def original_acc(X,y, no_of_Q):
    '''
    Function is created for checkign the accuracy of the classifier in the first iteration
    '''
    Dti = np.full(len(X),1/len(X))
    dti = Dti
    
    preds, cls = get_ht_new(X, X, Dti, no_of_Q)
    
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
        
        
    # what is the final y's
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


# In[99]:


# X


# ## Get Hypothesis

# In[100]:


def get_ht_new(X, Dti, no_of_Q):
    '''
    This function is used for returning the partitioning of the X's
    no_of_Q : represents the number of Q samples that must be choosen from all the M samples
    '''
    # we will extract the Q values with top Dti's
    Dti = np.array(Dti)
    ind_max = np.argpartition(Dti, -no_of_Q)[-no_of_Q:]

#     no of paritions = 3
    no_of_paritons = 3
    km = KMeans(
        n_clusters=no_of_paritons, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0)

    # now we will pass the corresponding X and parts with the Q samples to train the model 
    fitted_km = km.fit(X[ind_max])
    # prediction will be obtained for all the samples
    prediction = fitted_km.predict(X)
    d = prediction


    return  d, fitted_km


# # original distribution
# 
# 

# In[101]:


def original_distribution(parts, y, dti):
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


#     print("Classically calculated Dti for cross checking")
    print("0,0 -" , len(dti0_0), "sum - ", sum(dti0_0))
    print("0,1 -" , len(dti0_1), "sum - ", sum(dti0_1))
    print("1,0 -" , len(dti1_0), "sum - ", sum(dti1_0))
    print("1,1 -" , len(dti1_1), "sum - ", sum(dti1_1))
    print("2,0 -" , len(dti2_0), "sum - ", sum(dti2_0))
    print("2,1 -" , len(dti2_1), "sum - ", sum(dti2_1))


# In[102]:


# the new function 

def Oh_Dbk_custom_new(qc, qr1, qr_list, data_dict):
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
        string1 = flip_string(list(data_dict.keys())[i])
        string2 = flip_string(list(data_dict.values())[i])

        # the main idea is that the oracle looks at the all the values of xi and the corresponding jti that it wanna 
        # attach, then it makes the state 11111..(of the qubit register storing xi) and applies a mct to make this control and 
        # target is applied on jti finally we apply X to make the states back to the original xi's
        
        for j in range(len(string1)):
            if string1[j] == '0':
                qc.x(qr1[j])
        
        for j in range(len(string2)):
            if string2[j] == '1':
                qc.mct(qr1_arr, qr_list[j])
        
        for j in range(len(string1)):
            if string1[j] == '0':
                qc.x(qr1[j])


# In[103]:


def Oh_Dbk_custom_new_inv(qc, qr1, qr_list, data_dict):
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
        string1 = flip_string(list(data_dict.keys())[len(X)-i-1])
        string2 = flip_string(list(data_dict.values())[len(X)-i-1])

        
        # the main idea is that the oracle looks at the all the values of xi and the corresponding jti that it wanna 
        # attach, then it makes the state 11111.. and applies a mct to make this control and target is applied on jti
        # finally we apply X to make the states back to the original xi's
        
        for j in range(len(string1)):
            if string1[j] == '0':
                qc.x(qr1[j])
        
        for j in range(len(string2)):
            if string2[j] == '1':
                qc.mct(qr1_arr, qr_list[j])
        
        for j in range(len(string1)):
            if string1[j] == '0':
                qc.x(qr1[j])


# # Creating oracle(ht) for th iteration

# In[104]:


from qiskit.algorithms import AmplificationProblem
from qiskit.algorithms import Grover
from qiskit.providers.aer import AerSimulator
backend = AerSimulator()

def A_qaa( Dti_asrt):
    '''
    This executes the A for the amplification circuit
    '''
    
    qr_xi = QuantumRegister(5, 'xi')
    qr_Dti = QuantumRegister(4, 'dti')
    qr_final_rot = QuantumRegister(1,  'final_rot')
    
    qc = QuantumCircuit(qr_xi, qr_Dti, qr_final_rot)#cr)
    
#     here we will just apply hadamards instead of QRAM 
    qc.h(qr_xi)
    
    ## now we are adding up all the Oh Dbk Customs in one - Dti 
    
    # list of all the qubits in order 
    listofqubits = []

    #dti
    for i in range(qr_Dti.size):
        listofqubits.append(qr_Dti[i])


    ## making a list in this order 
    listofydj={}
#     print('Dti_asrt')
#     print(Dti_asrt)
#     print(type(Dti_asrt[0]))
#     print('----------------')
#     print('this one', f2bin(Dti_asrt[0], qr_Dti.size))
    # this list will be passed to the Oh_Dbk_custom function for the encoding of the |xi>|Dti> function 
    for i in range(len(X)):
        listofydj[dec_to_bin(i, qr_xi.size)] = str(flip_string(f2bin(Dti_asrt[i], qr_Dti.size)[1:5]))
#     print(listofydj)
    
    Oh_Dbk_custom_new(qc,qr_xi,listofqubits,listofydj)
    
    
    qc = qc.compose(rot_circuit(),[qr_Dti[0],qr_Dti[1],qr_Dti[2],qr_Dti[3],qr_final_rot[0]])
    
    qcinv=QuantumCircuit(qr_xi, qr_Dti, qr_final_rot)
    Oh_Dbk_custom_new_inv(qc,qr_xi,listofqubits,listofydj)
    
    
    return qc


# In[105]:


def amplification( Dti, reps):
    '''
    this will be able to actually be able to create the amplification circuit, run it and obtain the values of Dti after
    the amplification has been done!!
    '''

    Ainit = A_qaa(Dti)
#     Ainit.measure_all()


    widthh = Ainit.width()
    # oracle
    qo = QuantumCircuit(widthh)
    qo.z(widthh - 1)
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

    # appendiing the amplification
    qc_qaa.append(qc, qr)

    # the qubits you wanna measure
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
    sorted_dict = sorted(sorted_dict) ## ascending order sorting of keys

    ## now this dictonary will contain the values of Dti in sorted ordering 
    dti_sort = []
    for i in range(len(listofnew_dti.keys())):
        dti_sort.append((listofnew_dti[sorted_dict[i]]))

        
    return dti_sort


# # Estimating partition label weigths 

# In[106]:


def A(y ,k , b, preds, Dti):
    '''
    This circuit will estitmate the partition label weigths(the W's). 
    
    '''
    w=Dti


    qr_xi = QuantumRegister(5, 'xi')
    qr_yi = QuantumRegister(1, 'yi')
    qr_Dti = QuantumRegister(4, 'dti')
    qr_jti = QuantumRegister(2, 'jti')
    qr_i_1 = QuantumRegister(1,'i_1')# for I1 and I2
    qr_i_2 = QuantumRegister(1,'i_2')
    qr_kk = QuantumRegister(2,'k')# these are for initilaization of different k and b
    qr_b = QuantumRegister(1,'b')
    qr_Dbk = QuantumRegister(4, 'dbk')
    qr_qq = QuantumRegister(2, 'qq')
    qr_final_rot = QuantumRegister(1,  'final_rot')
    cr = ClassicalRegister(1)
    
    qc = QuantumCircuit(qr_xi,qr_yi, qr_Dti, qr_jti, qr_i_1, qr_i_2, qr_kk, qr_b,qr_Dbk, qr_qq, qr_final_rot)#cr)
    
#     here we will just apply hadamards instead of QRAM 
    qc.h(qr_xi)
    
    ## now we are adding up all the Oh Dbk Customs in one - yi - Dti - jti
    
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
    

    ## making a list in this order look at ordering doucmentation for explanantion
    listofydj={}
#     print(str(flip_string(f2bin(Dti[i] ,qr_Dti.size)))) 
    for i in range(len(X)):
        
        listofydj[dec_to_bin(i, qr_xi.size)] = str(dec_to_bin(preds[i],qr_jti.size)) + str(flip_string(f2bin(Dti[i] ,qr_Dti.size)[1:5])) +str(dec_to_bin(y[i],qr_yi.size))
    
#     print(listofydj)
#     print(listofqubits)
    Oh_Dbk_custom_new(qc,qr_xi,listofqubits,listofydj)
    

    qc = qc.compose(new_qc(2,[k]), [qr_kk[0],qr_kk[1]])
    qc = qc.compose(new_qc(1,[b]), [qr_b[0]])
    qc = qc.compose(check_j(),[qr_jti[0],qr_jti[1],qr_kk[0],qr_kk[1],qr_qq[0],qr_qq[1],qr_i_1[0]])
    qc = qc.compose(check_y(), [qr_yi[0], qr_b[0], qr_i_2[0]])
    
    # copying dbkti
    for i in range(4):
        qc.mct([qr_Dti[i],qr_i_1[0],qr_i_2[0]],qr_Dbk[i])
    
    qc = qc.compose(rot_circuit(),[qr_Dbk[0],qr_Dbk[1],qr_Dbk[2],qr_Dbk[3],qr_final_rot[0]])
    

    return qc


# In[107]:


from qiskit.algorithms import EstimationProblem
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms import IterativeAmplitudeEstimation
backend = AerSimulator()
quantum_instance_qae = QuantumInstance(backend,shots = 10)

def iteration_iqae( y, preds,Dti):
    '''
    This function executes the cirucit given above and uses IQAE on it to get the values of W+ and W- once these are obtained
    it returns the value of beta's 
    '''
    
    w = Dti
    wkb = []
    wp = []
    wn = []
    beta_j = []
    Zt = 0
        
    ## the iterative amplitude estimation
    iae = IterativeAmplitudeEstimation(
        epsilon_target=0.03,  # target accuracy
        alpha=0.07,  # width of the confidence interval
        quantum_instance=quantum_instance_qae,
    )
    
    Ztj = []
    for k in range(0,3):

      print('k value -', k)
        ## doing for label 0 
      qq = A(y,k,0, preds,Dti)
      
 
      problem0 = EstimationProblem( 
      state_preparation=qq,  # A operator
      objective_qubits=[qq.width()-1],  # we want to see if the last qubit is in the state |1> or not!!
      grover_operator=None)
        
      est_amp0 = iae.estimate(problem0).estimation

    
    
      if(est_amp0!=0):

          wkb.append(est_amp0)
          den = est_amp0


      else:
          den = 1e-8 #smoothing
          wn.append(den)

#       print('done for b=0')
        ## doing for label 1
      q2 = A( y,k,1,preds, Dti)

      problem1 = EstimationProblem(
      state_preparation=q2,  # A operator
      objective_qubits=[qq.width()-1],  # we want to see if the last qubit is in the state |1> or not!!
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


# In[ ]:





# In[108]:


def update_dti(X, Dti, y):
    '''
    Used for updation of Dti values using the updation rule by using Beta_j values
    '''
#     print("------------------------------------------------------------------------------------")

    ## first we will  normalize the Dti, basically such that 
    
    
    Dti_norm = Dti/sum(Dti)
#     print("-----------------------")
    print("->  Dti normalised:")
    print(Dti)
#     print("-----------------------")
    

    
    Dti_arc_sin_sqrt = np.arcsin(np.sqrt(Dti_norm))


#     print("-----------------------")
    print("->  arcsin(sqrt(Dti)) :")
    print(Dti_arc_sin_sqrt)
#     print("-----------------------")
    
    ##################### amplification - and obtiaing the Ht #############################################
    
    time_start_amp = time.time()
    # we pass unormalized Dti to this as normalization will take place automatically inside it!
    Dti_amp = amplification(Dti_arc_sin_sqrt, 2*round(np.log(25)*np.sqrt(len(X))))
    time_end_amp = time.time()
#     print("-----------------------")
    print("->  Dti values obatined after amplification :")
    print(Dti_amp)
#     print("-----------------------")

    # running the classical classifier
    no_of_Q = 4
    # this gives us the values of partitions obtained after training weights with w_i
    preds,classifier = get_ht_new(X, Dti_amp, no_of_Q)
    
    print("->  Domain partitions, hti")
    print(preds)
    
    print("->  Classical Distribution for verification of the weights")
    print(original_distribution(preds,y,Dti_arc_sin_sqrt)) #testing 2
    
    ########################### estimation ##################
    time_start_iqae = time.time()
    beta_j, Zt = iteration_iqae(y,  preds, Dti_arc_sin_sqrt)
    time_end_iqae = time.time()

    dti_up = []
    
    dtiii = Dti
    
    final_beta = []
    
    #change labels from 0,1 to -1,1 
    # we are doing this again and again as iteration_iqae cannot take -1 
    y_mod = []
    for i in range(len(y)):
        if y[i]==0:
            y_mod.append(-1)
        else:
            y_mod.append(1)
    
    ############################## updation of Dti
    for i in range(len(Dti)):
        dti_up.append(dtiii[i]*np.exp((-1)*y[i]*beta_j[preds[i]]))
        final_beta.append(beta_j[preds[i]])

    print("->  Time for amplification", time_end_amp - time_start_amp )
    print("->  Time for Estimation", time_end_iqae - time_start_iqae )
            
    return dti_up,beta_j, final_beta, preds, classifier, Zt


# In[109]:


def complete_imp3(num_iterations,X,y):
    '''
    The function which puts all the above functions together in order to obtain the accuracy and classifiers for the testing 
    
    '''
    print("Input X", X)
    print("Input y", y)
    print("Input Testing X", X_test)
    print("Input Testing y",y_test)

    
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
    beta_j_all = []
    classifiers_all = []
    Z_all = []
    for itr in range(num_iterations):
        
        print('-------------------------------------------------------------------------------------------------')
        print('ITERATION - ', itr, '\n')
              
        time_prtitr_start = time.time()
        dtii0,beta0,final_beta0,preds0, classifier0, Zt = update_dti(X, Dti, y)

        print("->  Updated Dti")
        print(dtii0)
        dti.append(dtii0)
        beta.append(final_beta0)
        beta_j_all.append(beta0)
        classifiers_all.append(classifier0)
        Z_all.append(Zt)
#         print(beta)
        final_bin, acc = final_bin_predictions(X,y_mod,Dti,beta)
        print("->  New Binary labels : ", final_bin)
        print("->  New Accuracy : ", acc)
        accuracy_final.append(acc)
        
        Dti = dtii0
        
        time_prtitr_end = time.time()
        
        print("->  Total time per iteration", time_prtitr_end - time_prtitr_start)
    
    import matplotlib.pyplot as plt
    from matplotlib import style
    
    plt.style.use('seaborn')
    plt.plot(list(range(num_iterations)), accuracy_final)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.show()
    
    return dti, beta, accuracy_final, beta_j_all, classifiers_all, Z_all


# # Predict
# 
# The training of the boosting algorithm provides us with the value of $D^t_i$'s. The prediction after each iteration would get better using the combination of the weights. Now we need a function which uses all these weights and creates a better result

# this function gets the H(x) after each iteration

# In[110]:


def final_bin_predictions(X,y_mod,Dti,betas):
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


# # Running

# In[111]:


no_of_itr = 25

dti, beta, accuracy_final, beta_j_all, classifiers_all, Z_all= complete_imp3(no_of_itr,X,y)


# In[ ]:


print("-------------------")
print("Training accuracy :" , accuracy_final)
print("-------------------")
print("beta_j_all :" , beta_j_all)
print("-------------------")
print("beta :" , beta)
print("-------------------")
print("Final dti :" , dti)
print("-------------------")
print("Z values for all the iterations :" , Z_all)
print("-------------------")


# # Testing accuracy

# In[ ]:


def testing(classifiers, Beta_j,X,y, No_of_itr, no_of_Q):
    '''
    here beta_j is the function which is contains the betas in the form - [[beta0,beta1,beta2]itr=1,[beta0,beta1,beta2]itr=2,[beta0,beta1,beta2]itr=3....]
    
    '''

    ## now all the clustering models are stored in classifiers
    
#     X = X_train
#     y = y_train
    
    ## getting y_mod
    T = No_of_itr
    #change labels from 0,1 to -1,1
    y_mod = []
    for i in range(len(y)):
        if y[i]==0:
            y_mod.append(-1)
        else:
            y_mod.append(1)
    
    ## lets do the partitioning to produce domains
    Dti = np.full(len(X),1/len(X))
    
    Beta_js = []# the array which will store all the betas
    accuracy_final_test= []
    for i in range(T):
        beta_j = Beta_j[i]
        predst = classifiers[i].predict(X)
        print(predst)
        final_beta =[]
        # updation of dti and distribution of betas
        for i in range(len(X)):
            if(predst[i]==0):
                final_beta.append(beta_j[0])

            elif(predst[i]==1):
                final_beta.append(beta_j[1])

            else:
                final_beta.append(beta_j[2])
            
        Beta_js.append(final_beta)
            
            # now lets add up the betas and get H(x)
#         print(len())
        final_bin, acc = final_bin_predictions(X,y_mod,Dti,Beta_js)
        accuracy_final_test.append(acc)
            
#     print(Beta_js)
    import matplotlib.pyplot as plt
    from matplotlib import style
    
    plt.style.use('seaborn')
    plt.plot(list(range(T)), accuracy_final_test)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Breast cancer Testing Accuracy M=16")
    plt.show()
        
    return Beta_js, accuracy_final_test


# In[32]:


no_of_Q =  4

Beta_js,accuracy_final_test = testing(classifiers_all, beta_j_all,X_test,y_test, no_of_itr, no_of_Q)


# In[33]:


print("-------------------")
print("Beta distribtuion for testing samples",Beta_js)
print("-------------------")
print("Final Testing accuracy", accuracy_final_test)
print("-------------------")


# In[ ]:





# In[ ]:





# In[ ]:




