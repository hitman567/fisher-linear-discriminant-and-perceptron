

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def training_perceptron(x, y,train,image):
    w = np.ones(len(x[0]))
    n = 0

    errorarray = np.ones(len(y)) # vector for errors (actual - predictions)
    J = []  # vector for the SSE(sum of squared error ) cost function
    yhatmain_vec = np.ones(len(y))  # vector for predictions
    while n < t:
        for i in range(0, len(x)):
            # Activation Function
            f = np.dot(x[i], w)
            if f >= z:
                yhat = 1.
            else:
                yhat = 0.
            yhatmain_vec[i] = yhat
            # Update the weights
            for j in range(0, len(w)):
                w[j] = w[j] + eta*(y[i]-yhat)*x[i][j]


        n += 1
        x1 = 0
        y1 = 0

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('ML')
        count = 0
        for i in np.linspace(np.amin(x[:,1]),np.amax(x[:,1])):
                slope = -(w[0]/w[2])/(w[0]/w[1])
                intercept = -w[0]/w[2]
                if count==0:
                    x1 = i
                    y1 = (slope*i) + intercept
                y2 = (slope*i) + intercept
                x2 = i
                count+=1
              # plotting training dataset line that updates after each epoch.
        plt.plot([x2,x1],[y2,y1],"k-")
        plt.title('Training')
        # points to be plotted
        plt.scatter(train.values[:,1], train.values[:,2], c = train[3], alpha=1.0)
        plt.axis('equal')
        image += 1
        plt.savefig('images/foo' + str(image)+".png", bbox_inches='tight')
        plt.clf()
           # computing the sum-of-squared errors
        for i in range(0,len(y)):
           errorarray[i] = (y[i]-yhatmain_vec[i])**2
        J.append(0.5*np.sum(errorarray))
    return w, J,image
    # function to draw  and save initial input dataset
def draw(filename,k,image):

    df = pd.read_csv(filename, header=None)
    plt.scatter(df.values[:,1], df.values[:,2], c = df[3], alpha=1.0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('ML' + "dataset" + str(k))
    plt.axis('equal')
    image +=1
    plt.savefig('images/foo' + str(image)+".png", bbox_inches='tight')
    plt.clf()
    df[0] = 1
    return  df[0:int(0.7*len(df))] , df[int(0.7*len(df)):int(len(df))],image
    # testing perceptron on training dataset and plotting yellow for positive points and purple for negative. if any point is predicted wrong it is marked as red.
def testing_perceptron(x, y,w,x1,y1,x2,y2,image):
    y_pred = []
    for i in range(0, len(x-1)):
        f = np.dot(x[i], w)
        if f > z:
            yhat = 1
        else:
            yhat = 0
        y_pred.append(yhat)
        rang = ""
        if yhat == 0:
            rang="purple"
            plt.plot(x[i][1],x[i][2], color='purple', marker='o');
        else:
            rang = "yellow"
            plt.plot(x[i][1],x[i][2], color='yellow', marker='x');
        if yhat!=y[i]:
            plt.plot(x[i][1],x[i][2], color='red', marker='*');

    plt.axis('equal')
    # plotting testing dataset.
    plt.plot([x2,x1],[y2,y1],"k-")
    plt.title('Testing')
    image += 1
    plt.savefig('images/foo' + str(image)+".png", bbox_inches='tight')
    plt.clf()
    return  y_pred ,image

def training(train,test,x_train ,y_train,image):
    print(x_train)
    print(y_train)

    w ,J,image= training_perceptron(x_train, y_train,train,image)
# w = weight vector
    print (w)
    print (J)
    x1 = 0
    y1 = 0
# drawing the final line that classify points
    plt.xlabel('x')
    plt.ylabel('y')
    count = 0
    for i in np.linspace(np.amin(x_train[:,1]),np.amax(x_train[:,1])):
            slope = -(w[0]/w[2])/(w[0]/w[1])
            intercept = -w[0]/w[2]
            if count==0:
               x1 = i
               y1 = (slope*i) + intercept
            y2 = (slope*i) + intercept
            x2 = i
            count+=1
    plt.plot([x2,x1],[y2,y1],"k-")
    plt.title('Training')
    plt.scatter(train.values[:,1], train.values[:,2], c = train[3], alpha=1.0)
    plt.axis('equal')
    image += 1
    plt.savefig('images/foo' + str(image)+".png", bbox_inches='tight')
    plt.clf()

    return w,x1,x2,y1,y2,image

# fucntion to retrieve data from datasets and convert them into new (70%) training and (30%)testing sets
def datasets(filename,k,image):
    train,test ,image= draw(filename,k,image)
    x_train = train.values[:, 0:3]
    y_train = train.values[:, 3]
    x_test = test.values[:, 0:3]
    y_test = test.values[:, 3]

       # training perceptron
    w,x1,x2,y1,y2,image= training(train,test,x_train ,y_train,image)
    # testing perceptron
    y_pred ,image= testing_perceptron(x_test,y_test,w,x1,y1,x2,y2,image)
    return image

#         z: activation function threshold
#         eta: learning rate
#         t: number of iterations
image1 = 0
z = 0.0
eta = 0.1
t = 50
#testing and training each dataset
image1  = datasets("dataset_11.csv",1,image1)
image1 = datasets("dataset_22.csv",2,image1)
image1 = datasets("dataset_33.csv",3,image1)
