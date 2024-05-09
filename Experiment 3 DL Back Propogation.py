import numpy as np

def sigmoid(no, type='unipolar', lam=1):
    if type == "unipolar":
        return 1 / (1 + np.exp(-no * lam))
    elif type == "bipolar":
        return (2 / (1 + np.exp(-no * lam))) - 1
        
def der_sigmoid(oi, type='unipolar'):
    if type == "unipolar":
        return oi * (1 - oi)
    elif type == "bipolar":
        return (1 - oi**2) / 2

def forward_pass(b1,b2):
    net_h1 = np.dot(w_x1, x) + b1
    out_h1 = sigmoid(net_h1)
    net_h2 = np.dot(w_x2, x) + b1
    out_h2 = sigmoid(net_h2)
    global h
    h = [out_h1, out_h2]

    net_o1 = np.dot(w_h1, h) + b2
    out_o1 = sigmoid(net_o1)
    net_o2 = np.dot(w_h2, h) + b2
    out_o2 = sigmoid(net_o2)
    global o
    o = [out_o1, out_o2]

    global e
    e = []
    for oi, di in zip(o, d):
        e.append(((oi - di)**2) / 2)
    print("Forward Pass:")
    print(f"Net_h1: {net_h1:.4f}, Out_h1: {out_h1:.4f}\nNet_h2: {net_h2:.4f}, Out_h2: {out_h2:.4f}")
    print(f"Net_o1: {net_o1:.4f}, Out_o1: {out_o1:.4f}\nNet_o2: {net_o2:.4f}, Out_o2: {out_o2:.4f}")
    print(f"E_o1: {e[0]:.4f}, E_o2: {e[1]:.4f}\nTotal Error: {sum(e):.4f}\n")

def backward_pass(eta):
    w_h1[0] -= eta * ((o[0]- d[0]) * der_sigmoid(o[0]) * h[0])
    w_h1[1] -= eta * ((o[0]- d[0]) * der_sigmoid(o[0]) * h[1])
    w_h2[0] -= eta * ((o[1]- d[1]) * der_sigmoid(o[1]) * h[0])
    w_h2[1] -= eta * ((o[1]- d[1]) * der_sigmoid(o[1]) * h[1])

    w_x1[0] -= eta * (((o[0]- d[0]) * der_sigmoid(o[0]) * w_h1[0] * der_sigmoid(h[0]) * x[0]) + ((o[1]- d[1]) * der_sigmoid(o[1]) * w_h2[0] * der_sigmoid(h[0]) * x[0]))
    w_x1[1] -= eta * (((o[0]- d[0]) * der_sigmoid(o[0]) * w_h1[0] * der_sigmoid(h[0]) * x[1]) + ((o[1]- d[1]) * der_sigmoid(o[1]) * w_h2[0] * der_sigmoid(h[0]) * x[1]))
    w_x2[0] -= eta * (((o[0]- d[0]) * der_sigmoid(o[0]) * w_h1[1] * der_sigmoid(h[1]) * x[0]) + ((o[1]- d[1]) * der_sigmoid(o[1]) * w_h2[1] * der_sigmoid(h[1]) * x[0]))
    w_x2[1] -= eta * (((o[0]- d[0]) * der_sigmoid(o[0]) * w_h1[0] * der_sigmoid(h[1]) * x[1]) + ((o[1]- d[1]) * der_sigmoid(o[1]) * w_h2[1] * der_sigmoid(h[1]) * x[1]))

    print("Backward Pass:")
    print(f"Updated_w5: {w_h1[0]:.4f}, Updated_w6: {w_h1[1]:.4f}\nUpdated_w7: {w_h2[0]:.4f}, Updated_w8: {w_h2[1]:.4f}")
    print(f"Updated_w1: {w_x1[0]:.4f}, Updated_w2: {w_x2[0]:.4f}\nUpdated_w3: {w_x1[1]:.4f}, Updated_w4: {w_x2[1]:.4f}\n\n")


global x, w_x1, w_x2, w_h1, w_h2, d
x = [0.10, 0.50]      
w_x1 = [0.10, 0.30]   #[w1, w2]
w_x2 = [0.20, 0.40]   #[w3, w4]
b1 = 0.25             
w_h1 = [0.50, 0.60]   #[w5, w6]
w_h2 = [0.70, 0.80]   #[w7, w8]
b2 = 0.35             
d = [0.05, 0.95]      
eta = 0.6       

# x = [0.05, 0.10]
# w_x1 = [0.15, 0.20]
# w_x2 = [0.25, 0.30]
# b1 = 0.35
# w_h1 = [0.40, 0.45]
# w_h2 = [0.50, 0.55]
# b2 = 0.60
# d = [0.01, 0.99]
# eta = 0.25

def backpropogation():
    forward_pass(b1, b2)
    backward_pass(eta)
"""
        w1         w5
    X1-------H1--------O1
      \      / \      / 
     w3\    /   \    /w6
        \  /     \  /   
         \/       \/    
         /\       /\     
        /  \     /  \    
     w2/    \   /    \w7    
      /      \ /      \   
    X2--------H2-------O2
        w4        w8
"""
it, max_epochs = 1, 10
e = [float('inf')]
while it <= max_epochs and sum(e) != 0:
    print(f"Iteration {it}-")
    backpropogation()
    it += 1

"""
# Output-

Iteration 1-
Forward Pass:
Net_h1: 0.4100, Out_h1: 0.6011
Net_h2: 0.4700, Out_h2: 0.6154
Net_o1: 1.0198, Out_o1: 0.7349
Net_o2: 1.2631, Out_o2: 0.7796
E_o1: 0.2346, E_o2: 0.0145
Total Error: 0.2491

Backward Pass:
Updated_w5: 0.4519, Updated_w6: 0.5507
Updated_w7: 0.7106, Updated_w8: 0.8108
Updated_w1: 0.0994, Updated_w2: 0.1993
Updated_w3: 0.2972, Updated_w4: 0.3974


Iteration 2-
Forward Pass:
Net_h1: 0.4085, Out_h1: 0.6007
Net_h2: 0.4686, Out_h2: 0.6151
Net_o1: 0.9602, Out_o1: 0.7232
Net_o2: 1.2756, Out_o2: 0.7817
E_o1: 0.2266, E_o2: 0.0142
Total Error: 0.2407

Backward Pass:
Updated_w5: 0.4033, Updated_w6: 0.5010
Updated_w7: 0.7209, Updated_w8: 0.8214
Updated_w1: 0.0989, Updated_w2: 0.1987
Updated_w3: 0.2947, Updated_w4: 0.3952


Iteration 3-
Forward Pass:
Net_h1: 0.4073, Out_h1: 0.6004
Net_h2: 0.4675, Out_h2: 0.6148
Net_o1: 0.9002, Out_o1: 0.7110
Net_o2: 1.2879, Out_o2: 0.7838
E_o1: 0.2184, E_o2: 0.0138
Total Error: 0.2323

Backward Pass:
Updated_w5: 0.3544, Updated_w6: 0.4509
Updated_w7: 0.7311, Updated_w8: 0.8318
Updated_w1: 0.0986, Updated_w2: 0.1981
Updated_w3: 0.2928, Updated_w4: 0.3935


Iteration 4-
Forward Pass:
Net_h1: 0.4062, Out_h1: 0.6002
Net_h2: 0.4665, Out_h2: 0.6146
Net_o1: 0.8398, Out_o1: 0.6984
Net_o2: 1.3000, Out_o2: 0.7858
E_o1: 0.2102, E_o2: 0.0135
Total Error: 0.2237

Backward Pass:
Updated_w5: 0.3052, Updated_w6: 0.4005
Updated_w7: 0.7410, Updated_w8: 0.8420
Updated_w1: 0.0982, Updated_w2: 0.1977
Updated_w3: 0.2912, Updated_w4: 0.3922


Iteration 5-
Forward Pass:
Net_h1: 0.4054, Out_h1: 0.6000
Net_h2: 0.4658, Out_h2: 0.6144
Net_o1: 0.7792, Out_o1: 0.6855
Net_o2: 1.3119, Out_o2: 0.7878
E_o1: 0.2019, E_o2: 0.0131
Total Error: 0.2151

Backward Pass:
Updated_w5: 0.2559, Updated_w6: 0.3500
Updated_w7: 0.7508, Updated_w8: 0.8520
Updated_w1: 0.0980, Updated_w2: 0.1973
Updated_w3: 0.2902, Updated_w4: 0.3913


Iteration 6-
Forward Pass:
Net_h1: 0.4049, Out_h1: 0.5999
Net_h2: 0.4654, Out_h2: 0.6143
Net_o1: 0.7185, Out_o1: 0.6723
Net_o2: 1.3237, Out_o2: 0.7898
E_o1: 0.1936, E_o2: 0.0128
Total Error: 0.2064

Backward Pass:
Updated_w5: 0.2065, Updated_w6: 0.2995
Updated_w7: 0.7603, Updated_w8: 0.8618
Updated_w1: 0.0979, Updated_w2: 0.1971
Updated_w3: 0.2896, Updated_w4: 0.3909


Iteration 7-
Forward Pass:
Net_h1: 0.4046, Out_h1: 0.5998
Net_h2: 0.4652, Out_h2: 0.6142
Net_o1: 0.6578, Out_o1: 0.6588
Net_o2: 1.3354, Out_o2: 0.7917
E_o1: 0.1853, E_o2: 0.0125
Total Error: 0.1978

Backward Pass:
Updated_w5: 0.1573, Updated_w6: 0.2491
Updated_w7: 0.7697, Updated_w8: 0.8714
Updated_w1: 0.0979, Updated_w2: 0.1969
Updated_w3: 0.2895, Updated_w4: 0.3910


Iteration 8-
Forward Pass:
Net_h1: 0.4045, Out_h1: 0.5998
Net_h2: 0.4652, Out_h2: 0.6142
Net_o1: 0.5973, Out_o1: 0.6450
Net_o2: 1.3469, Out_o2: 0.7936
E_o1: 0.1770, E_o2: 0.0122
Total Error: 0.1893

Backward Pass:
Updated_w5: 0.1082, Updated_w6: 0.1989
Updated_w7: 0.7790, Updated_w8: 0.8808
Updated_w1: 0.0980, Updated_w2: 0.1968
Updated_w3: 0.2899, Updated_w4: 0.3916


Iteration 9-
Forward Pass:
Net_h1: 0.4047, Out_h1: 0.5998
Net_h2: 0.4655, Out_h2: 0.6143
Net_o1: 0.5371, Out_o1: 0.6311
Net_o2: 1.3583, Out_o2: 0.7955
E_o1: 0.1689, E_o2: 0.0119
Total Error: 0.1808

Backward Pass:
Updated_w5: 0.0596, Updated_w6: 0.1490
Updated_w7: 0.7880, Updated_w8: 0.8901
Updated_w1: 0.0981, Updated_w2: 0.1969
Updated_w3: 0.2907, Updated_w4: 0.3926


Iteration 10-
Forward Pass:
Net_h1: 0.4052, Out_h1: 0.5999
Net_h2: 0.4660, Out_h2: 0.6144
Net_o1: 0.4773, Out_o1: 0.6171
Net_o2: 1.3697, Out_o2: 0.7973
E_o1: 0.1608, E_o2: 0.0117
Total Error: 0.1725

Backward Pass:
Updated_w5: 0.0113, Updated_w6: 0.0996
Updated_w7: 0.7969, Updated_w8: 0.8992
Updated_w1: 0.0984, Updated_w2: 0.1970
Updated_w3: 0.2920, Updated_w4: 0.3941

"""
