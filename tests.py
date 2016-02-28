import os
os.environ['PATH'] = r'C:\Anaconda2\Library\bin;'+os.environ['PATH']
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
def rosen(x):
     """The Rosenbrock function"""
     print x
     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
     xm = x[1:-1]
     xm_m1 = x[:-2]
     xm_p1 = x[2:]
     der = np.zeros_like(x)
     der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
     der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
     der[-1] = 200*(x[-1]-x[-2]**2)
     return der

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

print res.x


res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': True})
print res.x

fig, ax = plt.subplots()
x = np.linspace(-50,50, 200)
xx=[np.array(i) for i in x]
rosenVec = np.vectorize(rosen)
print xx[0]
y = rosenVec(xx)
print xx.shape
print y.shape
ax.plot(x,y)