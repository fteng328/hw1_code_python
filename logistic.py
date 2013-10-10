import numpy as np
from utils import sigmoid

def logistic(weights, data, targets, parameters):
    """
    Calculate log likelihood and derivatives with respect to weights.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        parameters: The parameters dictionary.

    Outputs:
        f:             The scalar error value.
        df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
        frac_correct:  Fraction of correctly classified examples.
    """

    # TODO: Finish this function
    #extract dimension parameter N
    f = 0
    N = data.shape[0]
    M = data.shape[1]
    new_col = np.ones(N)
    newData = np.column_stack((data, np.ones(np.shape(data)[0])))   #add extra colum to accomudate for w0
    z = np.dot(newData,weights)    # z is a N x 1 matix

    for i in range (N):
        #print "target = ", targets.item(i)," and z = ", z.item(i)
        f = f + (   targets.item(i)*z.item(i) - np.log((1+np.exp(z.item(i)))))

    df = np.zeros(M+1)

    for dim in range (M):
        for i in range(N):
            df[dim] = df[dim] + (targets.item(i)*data.item((i,dim)) - data.item((i,dim)) * (np.exp(z.item(i))) / (1 + np.exp(z.item(i))))

    frac_correct = 0

    z_sig = sigmoid(z)
    print z_sig
    correct_count = 0

    for i in range(targets.shape[0]):
        if (z_sig.item(i)>=0.5 and targets.item(i) == 1) or (z_sig.item(i)<0.5 and targets.item(i) == 0):
            correct_count = correct_count + 1
    frac_correct = correct_count * 1.0 / targets.shape[0]


    return f, df, frac_correct
