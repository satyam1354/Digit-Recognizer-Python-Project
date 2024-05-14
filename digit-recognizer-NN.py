import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('digit-recognizer/train.csv')
#print(data.head())

data = np.array(data)
m,n = data.shape  #rows,columns
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T #transpose
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
#X_dev = X_dev / 255

data_train = data[10000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
#X_train = X_train / 255
#_,m_dev = X_dev.shape
#_,m_train = X_train.shape
#data loaded above

print(Y_train)
# print(X_train)
print(X_train[0].shape)
print(X_train[:,0].shape)

def init_params():
    w1 = np.random.rand(10,784) -0.5
    b1 =  np.random.rand(10,1) - 0.5
    w2 = np.random.rand(10,10) -0.5
    b2 = np.random.rand(10,1) -0.5
    return w1,b1,w2,b2
def ReLU(z):
    return np.maximum(z, 0)
def softmax(z):
    #z -= np.max(z)  # Subtract the maximum value of z
    a =  np.exp(z) / sum(np.exp(z))
    return a
def forward_prop(w1,b1,w2,b2,x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1,a1,z2,a2
def deriv_ReLU(z):
    return z > 0
def one_hot(y):
    one_hot_y = np.zeros((y.size ,y.max()+1))
    one_hot_y[np.arange(y.size),y] = 1 
    one_hot_y = one_hot_y.T
    return one_hot_y  
def backward_prop(z1,a1,z2,a2,w1,w2,x,y):
    m = y.size
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2 ) # ,2)
    dz1 = w2.T.dot(dz2) * deriv_ReLU(z1)
    dw1 = 1 /m * dz1.dot(x.T)
    db1 =  1 / m * np.sum(dz1) # (  ,2)
    return dw1 ,db1 , dw2  ,db2
def update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha *db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha *db2
    return w1 ,b1 ,w2 ,b2

def get_predictions(a2):
    return np.argmax(a2, 0)
def get_accuracy(predictions, y):
    print("predictions - y :" ,predictions ,"-", y)
    return np.sum(predictions == y) / y.size

def gradient_descent(x,y,alpha ,iterations):
    w1,b1,w2,b2 = init_params()  #//
    for i in range(iterations):
        z1 , a1, z2 ,a2 = forward_prop(w1 , b1, w2 ,b2 , x)
        dw1 ,db1 , dw2, db2 = backward_prop(z1 ,a1 ,z2,a2 ,w1 ,w2 , x, y)
        w1, b1 ,w2, b2 = update_params(w1, b1, w2 ,b2,dw1,db1 ,dw2 ,db2, alpha)
        if i % 50 ==0:    #10 or 50
            print("Iteration: ", i)
            predictions = get_predictions(a2)
            print("Accuracy: ", get_accuracy(predictions, y))
    return w1 ,b1, w2,b2        
    
w1 ,b1,w2 ,b2 = gradient_descent(X_train , Y_train , 0.10, 500)  


#predictions tiem
def make_predictions(x , w1, b1,w2, b2):
    _,_,_,a2 = forward_prop(w1, b1, w2, b2, x)
    predictions = get_predictions(a2)
    return predictions

def test_predictions(index ,w1,b1 ,w2 , b2):
    curren_image = X_train[:,index,None]
    prediction = make_predictions(curren_image, w1, b1, w2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    curr_img = curren_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(curr_img, interpolation='nearest')
    plt.show()

test_predictions(0, w1, b1, w2, b2)
test_predictions(1, w1, b1, w2, b2)
test_predictions(2, w1, b1, w2, b2)
test_predictions(3, w1, b1, w2, b2)

# dev_predictions = make_predictions(X_dev ,w1,b1,w2,b2)
# get_accuracy(dev_predictions,Y_dev)