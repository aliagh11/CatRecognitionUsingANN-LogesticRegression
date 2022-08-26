
#This code is developed by Ali Aghadadashi / MSc student / Sharif University of Technology
#In this code we are going to recognizing ... Guess what? ... Cats using logistic regression based on neural networks framework and Binary classification

#Importing required libraries
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

#Mounting google drive for loading datasets
from google.colab import drive
drive.mount('/content/drive')

#Importing Train_set data_sets
train_dataset = h5py.File('/content/drive/MyDrive/AIProjects/CatRecognition/train_catvnoncat.h5', 'r')
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) #Train_set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:]) #Train_set labels

#Importing Test_set data_sets
test_dataset = h5py.File('/content/drive/MyDrive/AIProjects/CatRecognition/test_catvnoncat.h5', 'r')
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) #Test_set features
test_set_y_orig = np.array(test_dataset["test_set_y"][:]) #Test_set labels

classes = np.array(test_dataset["list_classes"][:]) #The list of classes (Cat - Not Cat)

# Reshaping train and test data_sets into a row vector
train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

# Example of a picture that is not a Cat
index1 = 4
plt.imshow(train_set_x_orig[index1])
if train_set_y_orig[0][index1] == 1:
  print("This is a Cat!")
else:
  print("This not a Cat!")

#Example of a picture that is a Cat
index2 = 7
plt.imshow(train_set_x_orig[index2])
if train_set_y_orig[0][index2] == 1:
  print("This is a Cat!")
else:
  print("This is not a Cat!")

#Number of train and test examples 
m_train = train_set_x_orig.shape[0] #Number of training_set examples
m_test = test_set_x_orig.shape[0] #Number of testing_set examples
px = train_set_x_orig.shape[1] #Size of images in pixels

print("Number of training_set examples : " + str(m_train))
print("Number of testing_set examples : " + str(m_test))
print("Each image has size of : " + str(px) + "*" + str(px) + "*" + "3")

#flatten the train and test data into a single vector shape
#(64*64*3,1) > (shape[0],-1).T
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print(train_set_x_flatten.shape)
print(train_set_y_orig.shape)
print(test_set_x_flatten.shape)
print(test_set_y_orig.shape)

#Normalizing data by dividing all of data into maximum value of each pixel (255)
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

#sigmoid function 
def sigmoid(z):
  s = 1 / (1+np.exp(-z))
  return s

#Initialize Weights and Biases with ZERO
def initialize_with_zeros(dim):
  w = np.zeros((dim,1))
  b=0.0
  return w, b

#Propogation function
def propagate(w,b,X,Y): #inputs are weights, biases, features and labels
  m = X.shape[1] #Number of training examples

  A = sigmoid(np.dot(w.T,X)+b)  #Actication function (sigmoid)
  cost = -1/m * (np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))) #Cost function of logestic regression

  dw = (1/m)*np.dot((X),((A-Y).T)) #derivitative for weights update
  db = (1/m)*(np.sum(A-Y)) #derivitatives for biases update

  cost = np.squeeze(np.array(cost)) 
  
  grads = {"dw": dw,
           "db": db}
    
  return grads, cost

#Optimization function
def optimize(w,b,X,Y, iterations = 100, learning_rate = 0.009, print_cost = False):
  
  w = copy.deepcopy(w)
  b = copy.deepcopy(b)

  costs = []

  for i in range(iterations):
    grads, cost = propagate(w,b,X,Y)

    dw = grads["dw"]
    db = grads["db"]

    w = w - (learning_rate * dw)
    b = b - (learning_rate * db)

    if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training itterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

#predict function
def predict(w,b,X):

  m = X.shape[1]

  Y_prediction = np.zeros((1,m))
  w = w.reshape(X.shape[0],1)

  A = sigmoid(np.dot(w.T,X)+b)

  for i in range(A.shape[1]):
    if A[0,i] > 0.5:
      Y_prediction[0,i] = 1
    else:
      Y_prediction[0,i] = 0

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, iterations=2000, learning_rate=0.5, print_cost=False):
    
    w, b = initialize_with_zeros(X_train.shape[0])
    
    params, grads, costs = optimize(w, b, X_train, Y_train, iterations, learning_rate, print_cost)
    
    w = params["w"]
    b = params["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    # YOUR CODE ENDS HERE

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    return w,b

model(train_set_x_flatten, train_set_y_orig, test_set_x_flatten, test_set_y_orig)

