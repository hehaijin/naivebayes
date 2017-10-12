import matplotlib.pyplot as plt

beta=[0.00001,0.0001,0.001,0.01,0.1,1]
result=[5,4,3,2,1,0]

plt.semilogx(beta, result)
plt.title('beta and accuracy')
plt.grid(True)
plt.show()
 








