import numpy as np

def signum(type, no):
    if type == "unipolar":
        if no >= 0:
            return 1
        else:
            return 0
    elif type == "bipolar":
        if no < 0:
            return -1
        elif no == 0:
            return 0
        else:
            return 1
        
def perceptron(x, w, b, c, fn, d):
    data = len(x)
    wgts = np.asarray(w).astype(float)
    xi = np.asarray(x).astype(float)
    it, max_epochs = 1, 1000
    r = [0,0,0,0]
    error = -1
    # print(xi, wgts)
    while it < max_epochs and error != 4:
        print(f"\nIteration {it}-")
        it += 1
        for i in range(data):
            # net = 0
            print(wgts, x[i])
            net = np.dot(wgts, x[i]) + b
            oi = signum(fn,net)
            r[i] = d[i] - oi
            delta_w = c * r[i] * np.asarray(x[i])
            wgts += delta_w
            b += r[i]
            error = r.count(0)
            print(f"Net{i+1}: {net}\nObserved Output: {oi}\nError: {r[i]}\n\u0394 W: {delta_w}\nUpdated Weights: {wgts}\nBias: {b}")
            

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
out_and = [0, 0, 0, 1]
# inputs = [[2, 2], [1, -2], [-2, 2], [-1, 1]]
# out_nor = [1, 0, 0, 0]
# out_or = [0, 1, 1, 1]
# weights = [rd.randint(-3, 3) for i in range(2)]
weights = [0, 0]
lcr = 1 # int(input("Enter the Learning Constant Rate: "))
# bias = rd.randint(-1,1)
bias = 0 # int(input("Enter the Initial Bias: "))
act_fun = "unipolar" #int(input("Select the type of Activating function- \n1) Unipolar, \n2) Bipolar \nEnter your choice: "))
# print(f"{inputs[:, 1]}, {inputs[:, 0]}, {inputs[:, 1] * inputs[:, 0]}")
print("Weights:",weights)
print("Bias:", bias)

perceptron(inputs, weights, bias, lcr, act_fun, out_and)

"""
# Output-

Weights: [0, 0]
Bias: 0

Iteration 1-
[0. 0.] [0, 0]
Net1: 0.0
Observed Output: 1
Error: -1
Δ W: [0 0]
Updated Weights: [0. 0.]
Bias: -1
[0. 0.] [0, 1]
Net2: -1.0
Observed Output: 0
Error: 0
Δ W: [0 0]
Updated Weights: [0. 0.]
Bias: -1
[0. 0.] [1, 0]
Net3: -1.0
Observed Output: 0
Error: 0
Δ W: [0 0]
Updated Weights: [0. 0.]
Bias: -1
[0. 0.] [1, 1]
Net4: -1.0
Observed Output: 0
Error: 1
Δ W: [1 1]
Updated Weights: [1. 1.]
Bias: 0

Iteration 2-
[1. 1.] [0, 0]
Net1: 0.0
Observed Output: 1
Error: -1
Δ W: [0 0]
Updated Weights: [1. 1.]
Bias: -1
[1. 1.] [0, 1]
Net2: 0.0
Observed Output: 1
Error: -1
Δ W: [ 0 -1]
Updated Weights: [1. 0.]
Bias: -2
[1. 0.] [1, 0]
Net3: -1.0
Observed Output: 0
Error: 0
Δ W: [0 0]
Updated Weights: [1. 0.]
Bias: -2
[1. 0.] [1, 1]
Net4: -1.0
Observed Output: 0
Error: 1
Δ W: [1 1]
Updated Weights: [2. 1.]
Bias: -1

Iteration 3-
[2. 1.] [0, 0]
Net1: -1.0
Observed Output: 0
Error: 0
Δ W: [0 0]
Updated Weights: [2. 1.]
Bias: -1
[2. 1.] [0, 1]
Net2: 0.0
Observed Output: 1
Error: -1
Δ W: [ 0 -1]
Updated Weights: [2. 0.]
Bias: -2
[2. 0.] [1, 0]
Net3: 0.0
Observed Output: 1
Error: -1
Δ W: [-1  0]
Updated Weights: [1. 0.]
Bias: -3
[1. 0.] [1, 1]
Net4: -2.0
Observed Output: 0
Error: 1
Δ W: [1 1]
Updated Weights: [2. 1.]
Bias: -2

Iteration 4-
[2. 1.] [0, 0]
Net1: -2.0
Observed Output: 0
Error: 0
Δ W: [0 0]
Updated Weights: [2. 1.]
Bias: -2
[2. 1.] [0, 1]
Net2: -1.0
Observed Output: 0
Error: 0
Δ W: [0 0]
Updated Weights: [2. 1.]
Bias: -2
[2. 1.] [1, 0]
Net3: 0.0
Observed Output: 1
Error: -1
Δ W: [-1  0]
Updated Weights: [1. 1.]
Bias: -3
[1. 1.] [1, 1]
Net4: -1.0
Observed Output: 0
Error: 1
Δ W: [1 1]
Updated Weights: [2. 2.]
Bias: -2

Iteration 5-
[2. 2.] [0, 0]
Net1: -2.0
Observed Output: 0
Error: 0
Δ W: [0 0]
Updated Weights: [2. 2.]
Bias: -2
[2. 2.] [0, 1]
Net2: 0.0
Observed Output: 1
Error: -1
Δ W: [ 0 -1]
Updated Weights: [2. 1.]
Bias: -3
[2. 1.] [1, 0]
Net3: -1.0
Observed Output: 0
Error: 0
Δ W: [0 0]
Updated Weights: [2. 1.]
Bias: -3
[2. 1.] [1, 1]
Net4: 0.0
Observed Output: 1
Error: 0
Δ W: [0 0]
Updated Weights: [2. 1.]
Bias: -3

Iteration 6-
[2. 1.] [0, 0]
Net1: -3.0
Observed Output: 0
Error: 0
Δ W: [0 0]
Updated Weights: [2. 1.]
Bias: -3
[2. 1.] [0, 1]
Net2: -2.0
Observed Output: 0
Error: 0
Δ W: [0 0]
Updated Weights: [2. 1.]
Bias: -3
[2. 1.] [1, 0]
Net3: -1.0
Observed Output: 0
Error: 0
Δ W: [0 0]
Updated Weights: [2. 1.]
Bias: -3
[2. 1.] [1, 1]
Net4: 0.0
Observed Output: 1
Error: 0
Δ W: [0 0]
Updated Weights: [2. 1.]
Bias: -3
"""