import numpy as np
from check_grad import check_grad
from utils import load_train, load_train_small, load_valid
from logistic import logistic

def run_logistic_regression():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()

    # TODO: initialize parameters
    parameters = {
                    'learning_rate': 0.01 ,          
                    'weight_regularization': 0 ,
                    'num_iterations': 10
                 }

    # logistic regression weights
    dimension = 28*28
    z = np.ones([dimension+1, 1], int)
    z = z/100.0
    #weight = np.matrix('1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1')
    for i in xrange(0,28*28):
      if i%2 == 1:
        z[i] = 0
        
    weights = z


    #weights = 1,1,2,1

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    #run_check_grad(parameters)

    # Begin learning with gradient descent
    for t in xrange(parameters['num_iterations']):

        # TODO: you will need to modify this loop to create plots, etc.

        # find the negative log likelihood and derivatives w.r.t. weights
        f, df, frac_correct_train = logistic(weights,
                                             train_inputs,
                                             train_targets,
                                             parameters)

        _, _, frac_correct_valid = logistic(weights,
                                            valid_inputs,
                                            valid_targets,
                                            parameters)
        
        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        for i in range(weights.shape[0]):
          weights[i] = weights[i] + parameters['learning_rate'] * (df[i] - 0.001*(weights[i]))

        # print some stats
        print ("ITERATION:{:4d}   LOGL:{:4.2f}   "
               "TRAIN FRAC:{:2.2f}   VALID FRAC:{:2.2f}").format(t+1,
                                                                 f,
                                                                 frac_correct_train*100,
                                                                 frac_correct_valid*100)
def run_check_grad(parameters):
    """Performs gradient check on logistic function.
    """

    #This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1).reshape(-1, 1)
    data = np.random.randn(
                          num_examples*num_dimensions
                ).reshape(num_examples,num_dimensions)
    targets = np.random.randn(num_examples).reshape(-1, 1)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      parameters)

    print "diff =", diff

if __name__ == '__main__':
    run_logistic_regression()
