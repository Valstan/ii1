import pickle

fileObject = open('MyNet_Fish.txt', 'rb')
net2 = pickle.load(fileObject)
fileObject.close()

y = net2.activate([2, 3, 80, 1])
print('Хорошая погода = ', y)

y = net2.activate([10, 7, 40, 3])
print('Средняя погода = ', y)

y = net2.activate([20, 11, 10, 5])
print('Плохая погода = ', y)
