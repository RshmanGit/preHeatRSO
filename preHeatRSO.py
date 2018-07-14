from pybrain.structure import FeedForwardNetwork,LinearLayer,SigmoidLayer,TanhLayer,FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pickle
import math
import csv

#creators
def createNetwork():
    print("[+]Creating network...")
    global net
    net = FeedForwardNetwork()
    inLayer = LinearLayer(2,name='in')
    hiddenLayer = TanhLayer(5,name='hidden')
    outLayer = LinearLayer(4,name='out')

    net.addInputModule(inLayer)
    net.addModule(hiddenLayer)
    net.addOutputModule(outLayer)

    inToHidden = FullConnection(inLayer,hiddenLayer)
    hiddenToOut = FullConnection(hiddenLayer,outLayer)

    net.addConnection(inToHidden)
    net.addConnection(hiddenToOut)

    net.sortModules()
    print("[+] network created!")

def createDataSet():
    print("[+] Creating dataset....")
    global DS
    DS = SupervisedDataSet(2,4)
    print("[+] dataset created!")

def createTrainer():
    print("[+] Creating trainer....")
    global Trainer
    Trainer = BackpropTrainer(net,DS)
    print("[+] Trainer Created!")

#mediators
def appendTrainData(a,b,c,d,e,f):
    DS.appendLinked([a,b],[c,d,e,f])
    print("[+] appending done| length: "+str(len(DS)))

def trainData():
    print("[+] Training Started....")
    Trainer.trainUntilConvergence()
    print("[+] training finished")
    error = Trainer.train()
    print("Error: "+str(error))

def iterAppend(x):
    for i in range(0,x):
        print("Enter all 2 inputs")
        a = input("-> ")
        b = input("-> ")
        print("Enter all 4 outputs")
        c = input("-> ")
        d = input("-> ")
        e = input("-> ")
        f = input("-> ")
        appendTrainData(a,b,c,d,e,f)

def normalize():
    print("[+]Starting normalisation...")
    a = 2
    b = 6
    global deNormKey
    deNormKey = [[0 for x in range(a)] for y in range(b)]

    for key in ['input','target']:
        if(key == 'input'):
            ite = 2
            addon = 0
        if(key == 'target'):
            ite = 4
            addon = 2
        print(ite)
        for i in range(0,ite):
            mean = 0
            standardDeviation = 0
            #mean
            l = len(DS[key])
            s = 0
            for j in range(0,l):
                s += DS[key][j][i]
            mean = s/l
            #standard deviation
            dif = 0
            for j in range(0,l):
                dif += ((DS[key][j][i]-mean)*(DS[key][j][i]-mean))
            dif = dif/l
            standardDeviation = math.sqrt(dif)
            for j in range(0,l):
                DS[key][j][i] = (DS[key][j][i]-mean)/standardDeviation
            print("mean: "+str(mean)+" SD: "+str(standardDeviation))
            deNormKey[addon+i][0] = mean
            deNormKey[addon+i][1] = standardDeviation
    print("[+]normalisation done...")

def predict(a,b):
    print("Predicting..")
    a = (a - deNormKey[0][0])/deNormKey[0][1]
    b = (b - deNormKey[1][0])/deNormKey[1][1]

    out = []
    out = net.activate((a,b))

    for i in range(0,4):
        out[i] = (out[i]*deNormKey[i+2][1])+deNormKey[i+2][0]

    print(out)
    return out

def printData():
    for i in range(0,len(printDS)):
        print("\t"+str(printDS['input'][i])+" |\t "+str(printDS['target'][i]))

def printNormalizedData():
    for i in range(0,len(DS)):
        print("\t"+str(DS['input'][i])+" |\t "+str(DS['target'][i]))

def printDeNormKey():
    for i in range(0,6):
        print(str(i)+ " Mean: "+ str(deNormKey[i][0]) +" standardDeviation: "+str(deNormKey[i][1]))

#save functions
def saveAll():
    pickle.dump(net, open("network.p","wb"))
    pickle.dump(DS, open("dataSet.p","wb"))
    pickle.dump(printDS, open("printDS.p","wb"))
    pickle.dump(deNormKey, open("deNormKey.p","wb"))
    pickle.dump(Trainer,open("trainer.p","wb"))
    pickle.dump(generatedDataSet,open("generatedDataSet.p","wb"))
    print("All data saved..|")

def loadAll():
    global net
    net = pickle.load(open("network.p","rb"))
    print("[+]net loaded")
    global DS
    DS = pickle.load(open("dataSet.p","rb"))
    print("[+]DS loaded")
    global printDS
    printDS = pickle.load(open("printDS.p","rb"))
    print("[+]PrintDS loaded")
    global deNormKey
    deNormKey = pickle.load(open("deNormKey.p","rb"))
    print("[+]deNormKey loaded")
    global Trainer
    Trainer = pickle.load(open("trainer.p","rb"))
    print("[+]Trainer loaded")
    printData()
    global generatedDataSet
    generatedDataSet = pickle.load(open("generatedDataSet.p","rb"))

#plotData

def plotData():

    cm = plt.get_cmap("RdYlGn")

    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')

    print("[+]Plotting Started")

    X = generatedDataSet['input']
    Y = generatedDataSet['input']
    target = generatedDataSet['target']

    x = []
    y = []
    z = []

    for i in X:
        x.append(i[0])

    for i in Y:
        y.append(i[1])

    for i in target:
        z.append(i[0])

    colRange = len(z)

    col = [cm(float(i)/(colRange)) for i in xrange(colRange)]

    ax.scatter(x, y, z, c=col)

    ax.set_xlabel('Power(kW)')
    ax.set_ylabel('RSO at diff Temp')
    ax.set_zlabel('NOx g/kWh')
    #plt.show()


    #fig = plt.figure()
    ax = fig.add_subplot(222, projection='3d')

    z = []

    for i in target:
        z.append(i[1])

    ax.scatter(x, y, z, c=col)

    ax.set_xlabel('Power(kW)')
    ax.set_ylabel('RSO at diff Temp')
    ax.set_zlabel('HC g/kWh')
    #plt.show()


    #fig = plt.figure()
    ax = fig.add_subplot(223, projection='3d')

    z = []

    for i in target:
        z.append(i[2])

    ax.scatter(x, y, z, c=col)

    ax.set_xlabel('Power(kW)')
    ax.set_ylabel('RSO at diff Temp')
    ax.set_zlabel('CO g/kWh')
    #plt.show()

    #fig = plt.figure()
    ax = fig.add_subplot(224, projection='3d')

    z = []

    for i in target:
        z.append(i[3])

    ax.scatter(x, y, z, c=col)

    ax.set_xlabel('Power(kW)')
    ax.set_ylabel('RSO at diff Temp')
    ax.set_zlabel('smoke')
    plt.show()

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def plot2D():
    pred = generatedDataSet
    x_pred = []
    y_pred = []

    train = printDS
    x_train = []
    y_train = []

    fig = plt.figure()
    ax = fig.add_subplot(221)

    for i in train:
        if(i[0][0] == 4.32):
            x_train.append(i[0][1])
            y_train.append(i[1][0])

    ax.scatter(x_train, y_train, c='blue')

    for i in pred:
        if(isclose(i[0][0],4.32)):
            x_pred.append(i[0][1])
            y_pred.append(i[1][0])

    ax.scatter(x_pred,y_pred, s=1, c='red')

    ax.set_title("At Load 4.32")
    ax.set_xlabel('RSO at diff Temp')
    ax.set_ylabel('NOx')
#----------------------------------------------------------
    ax = fig.add_subplot(222)

    x_pred = []
    y_pred = []

    x_train = []
    y_train = []

    for i in train:
        if(i[0][0] == 4.32):
            x_train.append(i[0][1])
            y_train.append(i[1][1])

    ax.scatter(x_train, y_train, c='blue')

    for i in pred:
        if(isclose(i[0][0],4.32)):
            x_pred.append(i[0][1])
            y_pred.append(i[1][1])

    ax.scatter(x_pred,y_pred, s=1, c='red')

    ax.set_title("At Load 4.32")
    ax.set_xlabel('RSO at diff Temp')
    ax.set_ylabel('HC')
#----------------------------------------------------------
    ax = fig.add_subplot(223)

    x_pred = []
    y_pred = []

    x_train = []
    y_train = []

    for i in train:
        if(i[0][0] == 4.32):
            x_train.append(i[0][1])
            y_train.append(i[1][2])

    ax.scatter(x_train, y_train, c='blue')

    for i in pred:
        if(isclose(i[0][0],4.32)):
            x_pred.append(i[0][1])
            y_pred.append(i[1][2])

    ax.scatter(x_pred,y_pred, s=1, c='red')

    ax.set_title("At Load 4.32")
    ax.set_xlabel('RSO at diff Temp')
    ax.set_ylabel('CO')
#----------------------------------------------------------
    ax = fig.add_subplot(224)

    x_pred = []
    y_pred = []

    x_train = []
    y_train = []

    for i in train:
        if(i[0][0] == 4.32):
            x_train.append(i[0][1])
            y_train.append(i[1][3])

    ax.scatter(x_train, y_train, c='blue')

    for i in pred:
        if(isclose(i[0][0],4.32)):
            x_pred.append(i[0][1])
            y_pred.append(i[1][3])

    ax.scatter(x_pred,y_pred, s=1, c='red')

    ax.set_title("At Load 4.32")
    ax.set_xlabel('RSO at diff Temp')
    ax.set_ylabel('Smoke')
    plt.show()

def to_csv():
    output = generatedDataSet
    with open("output.csv","wb") as f:
        towrite = []
        writer = csv.writer(f)
        for i in output:
            j = []
            j = [i[0][0],i[0][1],i[1][0],i[1][1],i[1][2],i[1][3]]
            towrite.append(j)
        writer.writerows(towrite)
#main

def main():
    try:
        while True:
            choice = input("------------------------------------------\n1:All new\n2:Load all\n3:Add new data\n4:predict\n5:Print Data\n6:Recreate components\n7:Save\n8:Normalize\n9:print deNormKey\n10:print normalized data\n11:Generate Predicted Dataset\n12:Print generated DataSet\n13:plot generated DataSet\n14:2D plot\n15:To csv\n16:Exit\n-----------------------------------------\n")
            if(choice == 1):
                global printDS
                printDS = SupervisedDataSet(2,4)
                createNetwork()
                createDataSet()
                createTrainer()
                i = input("Enter the number of data lines you want to add: ")
                iterAppend(i)
                printDS = deepcopy(DS)
                normalize()
                print("printDS--------------------------------")
                print(printDS)
                print("DS-------------------------------------")
                print(DS)
                trainData()
                saveAll()

            if(choice == 2):
                loadAll()
                trainData()

            if(choice == 3):
                i = input("Enter the number of data lines you want to add: ")
                iterAppend(i)
                trainData()

            if(choice == 4):
                a = input("Enter the first input: ")
                b = input("Enter the second input: ")
                out = []
                out = predict(a,b)
                #remeber to add code that can append the output data to the network


            if(choice == 5):
                printData()

            if(choice == 6):
                inChoice = input("1:New network\n2:New Dataset\n")
                if(inChoice == 1):
                    createNetwork()
                    trainData()
                if(inChoice == 2):
                    createDataSet()
                    i = input("Enter the number of data lines you want to add: ")
                    iterAppend(i)
                    trainData()

            if(choice == 7):
                saveAll()

            if(choice == 8):
                normalize()

            if(choice == 9):
                printDeNormKey()

            if(choice == 10):
                printNormalizedData()

            if(choice == 11):

                global generatedDataSet
                generatedDataSet = SupervisedDataSet(2,4)

                ll1 = input("Enter the lower limit of the first input")
                hl1 = input("Enter the higher limit of the first input")
                ll2 = input("Enter the lower limit of the second input")
                hl2 = input("Enter the higher limit of the second input")

                for i1 in np.arange(ll1,hl1,0.02):
                    for i2 in np.arange(ll2,hl2,0.5):
                        out = []
                        out = predict(i1,i2)
                        generatedDataSet.appendLinked([i1,i2],[out[0],out[1],out[2],out[3]])

            if(choice == 12):
                print(generatedDataSet)

            if(choice == 13):
                plotData()

            if(choice == 14):
                plot2D()

            if(choice == 15):
                to_csv()

            if(choice == 16):
                break

    except Exception as e:
        print(str(e))

if __name__ == '__main__':
    main()
