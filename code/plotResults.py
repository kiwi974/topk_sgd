import matplotlib.pyplot as plt

# Plot the training errors located in a file



################# PARAMETERS ON WHICH YOU CAN ACT TO PLOT #################

choice = 0

while ( (choice != 1) & (choice != 2) & (choice != 3) & (choice != 4) & (choice != 5)):
    print( "Chose what you want to see : \n 1 : dense with step multiplied by 0.9 at every server iteration \n 2 : dense with step coming from the paper and a = d/k \n 3 : dense with step coming from the paper and a = 10*d/k \n 4 : sparse with step coming from the paper and a = 10*d/k  \n 5 : sparse with step multiplied by 0.9 at every server iteration")
    choice = input()





###########################################################################





dense09StepPath = '/home/kiwi974/cours/epfl/opti_ma/project/code/data/denseTopkStep09.txt'
densePaperStepPath = '/home/kiwi974/cours/epfl/opti_ma/project/code/data/denseTopkRightStep.txt'
densePaperStepD10Path = '/home/kiwi974/cours/epfl/opti_ma/project/code/data/denseTopkRightStepDividedby10.txt'
sparseRightStep = '/home/kiwi974/cours/epfl/opti_ma/project/code/data/sparseRightStep.txt'
sparse09Step = '/home/kiwi974/cours/epfl/opti_ma/project/code/data/sparse09Step.txt'



if (choice == 1):
    filePath = dense09StepPath
elif (choice == 2):
    filePath = densePaperStepD10Path
elif (choice == 3):
    filePath = densePaperStepPath
elif (choice == 4):
    filePath = sparseRightStep
elif (choice == 5):
    filePath = sparse09Step
else:
    print("This choice is not valid.")


############### OPENING OF THE FILES AND PREPARATION ##############

file = open(filePath, 'r')

components = []
errorsTab = []

# Extract data of the file

for line in file:
    data = line.split("<nbCompo>")
    components.append(int(data[0]))
    err = data[1].split((", "))
    n = len(err)
    errors = []
    for k in range(n):
        if (k == 0):
            errors.append(float(err[k][1:]))
        elif (k == (n-1)):
            errors.append(float(err[k][:-2]))
        else:
            errors.append((float(err[k])))
    errorsTab.append(errors)
file.close()


# Plot data

colors = ['firebrick', 'darkorange', 'rebeccapurple', 'gold', 'darkgreen', 'dodgerblue', 'magenta','brown']

n = len(components)

###################################################################


if (filePath == dense09StepPath):
    figure = plt.figure(figsize=(10, 10))
    splitComp = 30
    plt.plot([k + splitComp for k in range(len(errorsTab[n - 1]) - splitComp)], errorsTab[n - 1][splitComp:],colors[n - 1], label="Error for classic SGD (all components, " + str(len(errorsTab[n-1])) + " iterations).")
    for i in range(n-1):
        plt.plot([k+splitComp for k in range(len(errorsTab[i])-splitComp)], errorsTab[i][splitComp:], colors[i], label="Error for "+str(components[i])+" components chosen in topk (" + str(len(errorsTab[i])) + " iterations).")
        plt.xlabel("Server iterations.")
        plt.ylabel("Error")
        plt.title("Dense data : learning rate multiplied by 0.9 at each server iteration.")
        plt.legend()




elif (filePath == densePaperStepD10Path):
    for i in range(n):
        figure = plt.figure(figsize=(15, 5))
        plt.plot([k for k in range(len(errorsTab[i]))], errorsTab[i], colors[i], label="Error for "+str(components[i])+" components chosen in topk (" + str(len(errorsTab[i])) + " iterations).")
        plt.xlabel("Server iterations.")
        plt.ylabel("Error")
        plt.title("Dense data : paper learning rate with a = d/k.")
        plt.legend()



elif (filePath == densePaperStepPath):
    for i in range(n):
        figure = plt.figure(figsize=(15, 5))
        plt.plot([k for k in range(len(errorsTab[i]))], errorsTab[i], colors[i], label="Error for "+str(components[i])+" components chosen in topk (" + str(len(errorsTab[i])) + " iterations).")
        plt.xlabel("Server iterations.")
        plt.ylabel("Error")
        plt.title("Dense data : paper learning rate with a = 10*d/k.")
        plt.legend()

elif (filePath == sparseRightStep):
    figure = plt.figure(figsize=(10, 10))
    plt.plot([k for k in range(len(errorsTab[n - 1]))], errorsTab[n - 1],colors[n - 1], label="Error for classic SGD (all components, " + str(len(errorsTab[n - 1])) + " iterations).")
    for i in range(n-1):
        plt.plot([k for k in range(len(errorsTab[i]))], errorsTab[i], colors[i], label="Error for "+str(components[i])+" components chosen in topk (" + str(len(errorsTab[i])) + " iterations).")
        plt.xlabel("Server iterations.")
        plt.ylabel("Error")
        plt.title("Sparse data : paper learning rate with a = 10*d/k.")
        plt.legend()


elif (filePath == sparse09Step):
    figure = plt.figure(figsize=(10, 10))
    plt.plot([k for k in range(len(errorsTab[n - 1]))], errorsTab[n - 1], colors[n - 1],
             label="Error for classic SGD (all components, " + str(len(errorsTab[n - 1])) + " iterations).")
    for i in range(n-1):
        plt.plot([k for k in range(len(errorsTab[i]))], errorsTab[i], colors[i], label="Error for "+str(components[i])+" components chosen in topk (" + str(len(errorsTab[i])) + " iterations).")
        plt.xlabel("Server iterations.")
        plt.ylabel("Error")
        plt.title("Sparse data : learning rate multiplied by 0.9 at each server iteration.")
        plt.legend()


plt.show()
