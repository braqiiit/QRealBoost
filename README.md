# QRealBoost
Implementation of the QRealBoost algorithm as given in https://arxiv.org/abs/2110.12793.

We performed experiments with QRealBoost on two datasets - the breast cancer wisconsin dataset from the UCI
repository [1], and the random n-class classification dataset provided by Scikit-Learn [2]. The weak learner used here is the classical K-Means clustering also provided by Scikit-Learn [3]. The experiments were performed on Qiskit [4].

We perform 20-25 iterations for boosting the accuracy. The quantum circuits created here perform two tasks; Creating oracle ht for the t-th iteration, and Estimating partition label weights. The circuit architecture can be understood from the diagram below -

![QReal Boost Circuit](https://user-images.githubusercontent.com/78695257/154253475-8529c2f3-f755-451a-91c0-f0f83dc6916e.PNG)

The 'Breast_Cancer_Dataset' folder contains 3 sub-folders, each containing a Python file with experiment for a particular number of training samples of Breast Cancer Dataset and a Text file containing the detailed output of that experiment. All the experiments with this dataset have been performed with 25 iterations of the algorithm, for a fixed sample complexity Q = 4.

- m_16 - For 16 Training Samples
- m_32 - For 32 Training Samples
- m_8 - contains an ipynb file with code for the experiment as well as output printed

The 'Random_Generated_Dataset' folder contains 3 sub-folders, each containing a Python file with experiment for a particular number of training samples of Randomly generated binary dataset from sklearn.make_classification() and a Text file containing the detailed output of that experiment. All the experiments with this dataset have been performed with 20 iterations of the algorithm, for a fixed sample complexity Q = 4.

- m_16 - For 8 Training Samples
- m_16 - For 16 Training Samples
- m_32 - For 32 Training Samples

## References - 
- [1] [Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [2] [Randomly generated Binary Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)
- [3] [K-Means Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [4] [Qiskit](https://qiskit.org/)
