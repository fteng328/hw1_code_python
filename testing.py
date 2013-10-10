import numpy as np
from logistic import logistic

weight = np.matrix('1,1,1,1')
weights = weight.T
train_inputs = np.matrix('1 2 3; 100 4 5; 2 4 8')
train_targets = np.array([0,1,0])
parameters = 0
f, df, frac_correct_train = logistic(weights,
                                             train_inputs,
                                             train_targets,
                                             parameters)


print f 
print df
print frac_correct_train