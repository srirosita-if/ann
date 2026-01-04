import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)
w_ih = np.random.rand(2,2)
w_ho = np.random.rand(2,1)
b_h = np.random.rand(1,2)
b_o = np.random.rand(1,1)

lr = 0.5
epoch = 10000

for i in range(epoch):
    h_in = np.dot(X, w_ih) + b_h
    h_out = sigmoid(h_in)

    o_in = np.dot(h_out, w_ho) + b_o
    output = sigmoid(o_in)

    error = Y - output

    d_o = error * sigmoid_derivative(output)
    d_h = d_o.dot(w_ho.T) * sigmoid_derivative(h_out)

    w_ho += h_out.T.dot(d_o) * lr
    w_ih += X.T.dot(d_h) * lr
    b_o += np.sum(d_o, axis=0, keepdims=True) * lr
    b_h += np.sum(d_h, axis=0, keepdims=True) * lr

print("Hasil ANN XOR:")
for i in range(len(X)):
    print(X[i], "=>", round(output[i][0], 3))
