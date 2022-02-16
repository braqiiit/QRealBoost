# QRealBoost
Implementation of the QRealBoost algorithm as given in https://arxiv.org/abs/2110.12793.

We performed experiments with QRealBoost on two datasets - the breast cancer wisconsin dataset from the UCI
repository [1], and the random n-class classification dataset provided by Scikit-Learn [2]. The weak learner used here is the classical K-Means clustering also provided by Scikit-Learn [3]. The experiments were performed on Qiskit [4].

We perform 20-25 iterations for boosting the training accuracy. The quantum circuits created here perform two tasks; Creating oracle ht for the t-th iteration, and Estimating partition label weights. The circuit architecture can be understood from the diagram below -

![QReal Boost Circuit](https://user-images.githubusercontent.com/78695257/154253475-8529c2f3-f755-451a-91c0-f0f83dc6916e.PNG)


|D_t^i\rangle
