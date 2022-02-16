# QRealBoost
Implementation of the QRealBoost algorithm as given in https://arxiv.org/abs/2110.12793
We performed experiments with QRealBoost on two datasets - the breast cancer wisconsin dataset from the UCI
repository [1], and the random n-class classification dataset provided by Scikit-Learn [2]. The experiments were performed on Qiskit [3]. 


Each iteration of the implementation can be broken down into four different parts:
• Obtaining the $D^t_i$ values using quantum amplitude amplification.
• Using the weak learner to obtain the $t^{th}$ hypothesis $h_t$.
• Estimating the values of $W_b^{j,t}$ using iterative quantum amplitude estimation.
• Computing the $β_j$ and $Z_t$ values.
