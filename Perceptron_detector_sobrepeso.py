

import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import rand

class Perceptron:

  def __init__(self, n_inputs, learning_rate):
    self.w = - 1 + 2 * np.random.rand(n_inputs)
    self.b = - 1 + 2 * np.random.rand()
    self.eta = learning_rate

  def predict(self, X):
    _, p = X.shape
    y_est = np.zeros(p)

    for i in range(p):
      y_est[i] = np.dot(self.w, X[:,i])+self.b
      if y_est[i] >= 0:
        y_est[i]=1
      else:
        y_est[i]=0
    return y_est

  def fit(self, X, Y, epochs=50):
    _, p = X.shape
    for _ in range(epochs):
      for i in range(p):
        y_est = self.predict(X[:,i].reshape(-1,1))
        self.w += self.eta * (Y[i]-y_est) * X[:,i]
        self.b += self.eta * (Y[i]-y_est)

def draw_2d_percep(model):
  w1, w2, b = model.w[0], model.w[1], model.b 
  plt.plot([-2, 2],[(1/w2)*(-w1*(-2)-b),(1/w2)*(-w1*2-b)],'--k')
  plt.show()


  # Instanciar el modelo
model = Perceptron(2, 0.1)

# Datos

X = np.zeros((2,100))
Y = np.zeros((100))
escalar = True

for i in range(100):
    h = 1.2 + (2.4 - 1.2) * rand()
    p = 40 + (180 - 40) * rand()
    imc = p / (h**2)
    if imc > 25:
        Y[i] = 1    #Sobrepeso
    else:
        Y[i] = 0    #PersoNormal
    X[0,i] = h
    X[1,i] = p
print(X)

if escalar:
    for i in range(2):
        maxX = np.max(X[i,:])
        minX = np.min(X[i,:])
        X[i,:] = (X[i,:] - minX)/(maxX-minX)

# Entrenar
model.fit(X,Y)

# Predicción
model.predict(X)

# Primero dibujemos los puntos
_, p = X.shape
for i in range(p):
  if Y[i] == 0:
    plt.plot(X[0,i],X[1,i], 'or')
  else:
    plt.plot(X[0,i],X[1,i], 'ob')

# Dibujamos la tabla

plt.title('Perceptron')
plt.grid('on')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.xlabel(r'Altura (m)')
plt.ylabel(r'Peso (Kg)')

draw_2d_percep(model)