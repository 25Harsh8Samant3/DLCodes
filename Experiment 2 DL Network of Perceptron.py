import numpy as np
import random as rd

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

def network_of_preceptron(x, w, b, c, d):
    xi = np.asarray(x).astype(float)
    weights = np.asarray(w).astype(float)
    it, max_epochs = 1, 1000
    error = -1
    while it < max_epochs and error != 0:
        print(f"\nIteration {it}-")
        it += 1
        net = np.dot(weights, xi) + b
        oi = signum(type="unipolar",no=net)
        r = d - oi
        delta_w = c * r * xi
        weights += delta_w
        b += r
        error = r
        print(f"Net: {net}\nObserved Output: {oi}\nError: {r}\n\u0394 W: {delta_w}\nUpdated Weights: {weights}\nUpdated Bias: {b}")


input = [1, 1]
hidden = [[0, 0], [0, 1], [1, 0], [1, 1]]
weights = [rd.randint(-2,2) for i in range(4)]# [2, 2, 2, -5]
out_xor = {(-1, -1) : 0, (-1,1) : 1, (1, -1) : 1, (1, 1) : 0}
out_and = {(0, 0) : 0, (0, 1) : 0, (1, 0) : 0, (1, 1) : 1}
bias = -2
new_inputs = []
for i in hidden:
    new_inputs.append(signum(no=np.dot(i, input) + bias, type="unipolar"))
print(f"Hidden Layer Input: {new_inputs}\nInitial Weights: {weights}\nInitial Bias: {bias}\nLearning Rate: {1}")
network_of_preceptron(new_inputs, weights, bias, 1, out_and[tuple(input)])


"""
# Output-

Hidden Layer Input: [0, 0, 0, 1]
Initial Weights: [1, 1, 1, -2]
Initial Bias: -2
Learning Rate: 1

Iteration 1-
Net: -4.0
Observed Output: 0
Error: 1
Δ W: [0. 0. 0. 1.]
Updated Weights: [ 1.  1.  1. -1.]
Updated Bias: -1

Iteration 2-
Net: -2.0
Observed Output: 0
Error: 1
Δ W: [0. 0. 0. 1.]
Updated Weights: [1. 1. 1. 0.]
Updated Bias: 0

Iteration 3-
Net: 0.0
Observed Output: 1
Error: 0
Δ W: [0. 0. 0. 0.]
Updated Weights: [1. 1. 1. 0.]
Updated Bias: 0

"""