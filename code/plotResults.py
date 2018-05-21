import matplotlib.pyplot as plt

# Plot the training errors located in a file

sparsePath = '/home/kiwi974/cours/epfl/opti_ma/project/code/sparseTopKresult.txt'
densePath = '/home/kiwi974/cours/epfl/opti_ma/project/code/topkresult.txt'

filePath = densePath

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

colors = ['firebrick', 'darkorange', 'rebeccapurple', 'gold', 'darkgreen', 'dodgerblue', 'magenta']

plt.figure(figsize=(10,10))

splitComp = 30

for i in range(len(components)):
    plt.plot([k+splitComp for k in range(len(errorsTab[i])-splitComp)], errorsTab[i][splitComp:], colors[i], label="Error for "+str(components[i])+" components choose in topk.")
plt.xlabel("Iteration.")
plt.ylabel("Error.")
plt.title("Learning curves for the dense set of voice recognition, with step *= 0.9.")
plt.legend()
plt.show()