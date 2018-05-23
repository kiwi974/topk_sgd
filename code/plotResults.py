import matplotlib.pyplot as plt

# Plot the training errors located in a file



################# PARAMETERS ON WHICH YOU CAN ACT TO PLOT #################

### Chose the representation you want
# 1 : dense with step multiplied by 0.9 at every server iteration
# 2 : dense with step coming from the paper and a = d/k
# 3 : dense with step coming from the paper and a = 10*d/k
# 4 : sparse with step coming from the paper and a = 10*d/k
sparsity = 1





###########################################################################





dense09StepPath = '/home/kiwi974/cours/epfl/opti_ma/project/code/denseTopkStep09.txt'
densePaperStepPath = '/home/kiwi974/cours/epfl/opti_ma/project/code/denseTopkRightStep.txt'
densePaperStepD10Path = '/home/kiwi974/cours/epfl/opti_ma/project/code/denseTopkRightStepDividedby10.txt'

sparsePath = '/home/kiwi974/cours/epfl/opti_ma/project/code/sparseTopKresult.txt'


if (sparsity == 1):
    filePath = dense09StepPath
elif (sparsity == 2):
    filePath = densePaperStepD10Path
elif (sparsity == 3):
    filePath = densePaperStepPath
else:
    filePath = sparsePath



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
    plt.plot([k + splitComp for k in range(len(errorsTab[n - 1]) - splitComp)], errorsTab[n - 1][splitComp:],colors[n - 1], label="Error for classic SGD (all components).")
    for i in range(n-1):
        plt.plot([k+splitComp for k in range(len(errorsTab[i])-splitComp)], errorsTab[i][splitComp:], colors[i], label="Error for "+str(components[i])+" components choose in topk.")
        plt.xlabel("Server iterations.")
        plt.ylabel("Error")
        plt.title("Dense data : learning rate multiplied by 0.9 at each server iteration.")
        plt.legend()




elif (sparsity == 2):
    for i in range(n):
        figure = plt.figure(figsize=(15, 5))
        plt.plot([k for k in range(len(errorsTab[i]))], errorsTab[i], colors[i], label="Error for "+str(components[i])+" components choose in topk.")
        plt.xlabel("Server iterations.")
        plt.ylabel("Error")
        plt.title("Dense data : paper learning rate with a = d/k.")
        plt.legend()



elif (sparsity == 3):
    for i in range(n):
        figure = plt.figure(figsize=(15, 5))
        plt.plot([k for k in range(len(errorsTab[i]))], errorsTab[i], colors[i], label="Error for "+str(components[i])+" components choose in topk.")
        plt.xlabel("Server iterations.")
        plt.ylabel("Error")
        plt.title("Dense data : paper learning rate with a = 10*d/k.")
        plt.legend()


plt.show()
