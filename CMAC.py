"""
@author: psanhuez
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Weights
W = 35
LR = 0.00015
epochs = 9000


X = np.linspace(0, 1.5, 100) #Evenly spaced points
Y = np.sin(X)

X_shuffle = X
np.random.shuffle(X_shuffle)

X_train = X_shuffle[0:70]
X_test = X_shuffle[70:100]

Y = np.sin(X_shuffle)

Y_train = Y[0:70]
Y_test = Y[70:100]

def CMAC(X, X_train, Y_train, X_test, Y_test, W, num_cells, epochs, LR):
    epoch_count = 0
    weights = np.ones(W)
    
    x = np.linspace(min(X), max(X), W - num_cells + 1)
    
    for i in enumerate(x):
        x[i[0]]= round(i[1],1)
        
    look_up = np.zeros((len(x), W))
    
    for i in enumerate(x):
        look_up[i[0], i[0]:num_cells+i[0]-1] = 1
    
    
    map = dict()
    
    for i in enumerate(x):
        map[hash(i[1])] = [np.where(look_up[i[0]][:] == 1), look_up[i[0]][:]]
    
    
    while epoch_count < epochs:
        epoch_count = epoch_count + 1
        for i in X_train:
            
            x_round = round(i,1)
            index = (np.abs(x-x_round)).argmin()
            x_round = x[index]
            
            hash_temp = hash(x_round)
            
            if hash_temp in map.keys():
                
                temp_error = np.sin(i) - np.sum(map[hash_temp][1]*weights)
                weighted_learning = (LR *temp_error)/num_cells
                for j in map[hash_temp][0][0]:
                    weights[j] = weights[j] + weighted_learning
                    
             
            
    X_test = np.sort(X_test)
    Y_test = np.sin(X_test)
    Y_output = []
    
    for i in X_test:
        x_round = round(i,1)
        index = (np.abs(x-x_round)).argmin()
        x_round = x[index]
        
        hash_temp = hash(x_round)
        
        if hash_temp in map.keys():
            output = np.sum(map[hash_temp][1]*weights)
            Y_output.append(output)
 
    plt.figure()
    title = "Overlap of "+str(num_cells)
    plt.title(title)
    plt.plot(X_test, Y_test, '*', 'r')
    plt.plot(X_test, Y_output, '--', 'b')
    file_name = title + '.png'
    plt.savefig(file_name)
    
    return np.abs(np.mean((Y_output-Y_test)))
    
errors = []
overlap = np.arange(2,3)

for i in overlap:
    errors_avg = CMAC(X, X_train, Y_train, X_test, Y_test, 35, i, epochs, LR)
    errors.append(100-errors_avg*100)

    
plt.figure()
plt.title("Accuracy")
plt.plot(overlap, errors)  
plt.show() 
