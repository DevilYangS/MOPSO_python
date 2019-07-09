import numpy as np


def P_objective(Operation,Problem,M,Input):
    [Output, Boundary, Coding] = P_DTLZ(Operation, Problem, M, Input)
    if Boundary == []:
        return Output
    else:
        return Output, Boundary, Coding

def P_DTLZ(Operation,Problem,M,Input):
    Boundary = []
    Coding = ""
    k = 1
    K = [5, 10, 10, 10, 10, 10, 20]
    K_select = K[k - 1]
    if Operation == "init":
        D = M + K_select - 1
        MaxValue = np.ones((1, D))
        MinValue = np.zeros((1, D))
        Population = np.random.random((Input, D))
        Population = np.multiply(Population, np.tile(MaxValue, (Input, 1))) +\
            np.multiply((1-Population), np.tile(MinValue, (Input, 1)))
        Boundary = np.vstack((MaxValue, MinValue))
        Coding = "Real"
        return Population, Boundary, Coding
    elif Operation == "value":
        Population = Input
        FunctionValue = np.zeros((Population.shape[0], M))
        if Problem == "DTLZ1":
            # g = 100*(K_select+np.sum( (Population[:, M-1:] - 0.5)**2 - np.cos(20*np.pi*(Population[:, M-1:] - 0.5)), axis=1, keepdims = True))
            g = 100*(K_select+np.sum( (Population[:, M-1:] - 0.5)**2 - np.cos(20*np.pi*(Population[:, M-1:] - 0.5)), axis=1))
            for i in range(M):
                FunctionValue[:, i] = 0.5*np.multiply( np.prod(Population[:, :M-i-1], axis=1), (1+g))
                if i>0:
                    FunctionValue[:, i] = np.multiply(FunctionValue[:, i], 1-Population[:, M-i-1])
        elif Problem == "DTLZ2":
            g = np.sum( (Population[:, M-1:] - 0.5)**2,axis=1)
            for i in range(M):
                FunctionValue[:, i] = (1+g)*np.prod( np.cos( 0.5*np.pi*(Population[:, :M-i-1]) ),axis=1 )
                if i>0:
                    FunctionValue[:, i] = np.multiply(FunctionValue[:, i], np.sin( 0.5*np.pi* ( Population[:, M-i-1]) ) )

        return FunctionValue, Boundary, Coding








