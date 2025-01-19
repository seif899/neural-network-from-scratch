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

weights = np.random.randn(input_size, hidden_size) 
biases1 = np.zeros(hidden_size)
biases2 = np.zeros(output_size)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calculate_layer(weights,biases,input_layer,output_layer,output_size,input_size):
    for i in range(output_size): 
        s=0
        for j in range(input_size):
            s+=weights[j,i]*input_layer[j]

        output_layer[i] = sigmoid(s+biases[i])
    
    #print(output_layer)
    return output_layer
def calculate_cost(expected,output_layer,output_size):
    return np.array([(expected[i]-output_layer[i])**2 for i in range(output_size)])

def backpropogate(output_layer,output_size):
    expected=np.array([1 if i==y[0] else 0 for i in range(output_size)])
    cost_vector = calculate_cost(expected,output_layer,output_size)
    print(cost_vector)


calculate_layer(weights,biases1,X[0],hidden_layer,hidden_size,input_size)
calculate_layer(weights,biases2,hidden_layer,output_layer,output_size,hidden_size)
backpropogate(output_layer,output_size)