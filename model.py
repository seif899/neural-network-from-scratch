import pandas as pd
import numpy as np
import math

labels_file = "csvTrainLabel 60k x 1.csv"
labels = pd.read_csv(labels_file, header=None)


features_file = "csvTrainImages 60k x 784.csv"
features = pd.read_csv(features_file, header=None)

dataset = pd.concat([features, labels], axis=1)


column_names = [f"pixel_{i}" for i in range(features.shape[1])] + ["label"]
dataset.columns = column_names


X = features.values / 255.0  
y = labels.values.flatten()
print(y[0])
hidden_layer=np.zeros(128)
output_layer=np.zeros(10)


input_size=784
hidden_size=128
output_size=10

weights1 = np.random.randn(input_size, hidden_size) 
weights2 = np.random.randn(hidden_size, output_size)

cost_weight_derivative = np.zeros(input_size, hidden_size)

biases1 = np.zeros(hidden_size)
biases2 = np.zeros(output_size)

weightedInputs1 = np.zeros(hidden_size)
weightedInputs2 = np.zeros(output_size)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def calculate_layer(weights, biases, input_layer, output_layer, output_size, input_size):
    for i in range(output_size):
        s = np.dot(input_layer, weights[:, i]) + biases[i]
        if output_size == hidden_size:
            weightedInputs1[i] = s
        else:
            weightedInputs2[i] = s
        output_layer[i] = sigmoid(s)
    return output_layer

def calculate_cost(expected,output_layer,output_size):
    return sum(np.array([(expected[i]-output_layer[i])**2 for i in range(output_size)]))

def sigmoid_deriv(f):
    return f * (1-f)

def backpropagate(output_layer, expected, hidden_layer, weights1, weights2, biases1, biases2, learning_rate):
    output_error = output_layer - expected
    output_delta = output_error * sigmoid_deriv(output_layer)  

    weights2_grad = np.outer(hidden_layer, output_delta)
    biases2_grad = output_delta


    hidden_error = np.dot(weights2, output_delta) * sigmoid_deriv(hidden_layer)  
    weights1_grad = np.outer(X[0], hidden_error)  
    biases1_grad = hidden_error

    weights2 -= learning_rate * weights2_grad
    biases2 -= learning_rate * biases2_grad
    weights1 -= learning_rate * weights1_grad
    biases1 -= learning_rate * biases1_grad

    return weights1, weights2, biases1, biases2

    


            


