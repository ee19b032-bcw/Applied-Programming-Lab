
"""
		EE2703: Applied Programming Lab
		Assignment - 4
		Prof. Harishankar Ramachandran
		
			Done By:
			KATARI HARI CHANDAN
			EE19B032
"""

# Note: After we run the code, we get the 1st graph as output and once we close it, we get the 2nd graph and once we close it, we get the 3rd graph and so on till the 11th graph.
#	Once we get all graphs, after we close the last(11th) graph, we get the output statements giving the Mean Errors in each function wrt their Fourier Series.  

import os
import sys
import numpy as np				#We are imported all the necessariy modules required for the code.
import scipy as sp
import matplotlib.pyplot as plt
from scipy import integrate


#We defined explonential function as f(x)
def f(x):
    return np.exp(x)

#We defined cos(cos(x)) function as g(x)
def g(x):
    return np.cos(np.cos(x))
    
#Rather than writing np.pi or np.e again and an=gain we define varialbles as pi and e.
pi = np.pi
e = np.e

# The values input variable x should take are stated.
x = np.linspace(-2*pi,4*pi,100)

# We are plotting e^x in linear scale.
plt.grid()
plt.plot(x,f(x))
plt.title('Plot of e^x in linear scale')
plt.xlabel('x')
plt.ylabel('e^x (linear)')
plt.show()

#We are plotting e^x in semilog scale because the exponential raises rapidly as we increase x.
plt.grid()
plt.semilogy(x,f(x))
plt.title('Plot of e^x in semilog scale')
plt.xlabel('x')
plt.ylabel('e^x (semilog)')
plt.show()

# We are plotting cos(cos(x)) in linear scale.
plt.grid()
plt.plot(x,g(x))
plt.title('Actual plot of cos(cos(x))')
plt.xlabel('x')
plt.ylabel('cos(cos(x))')
plt.show()


# We know from the expressions for Fourier Coefficients that we are suppose to find the integral of f(x)*cos(k*x) and f(x)*sin(k*x). Hence we defined u,v,w,z functions namely.
def u(x,k):
    return f(x)*np.cos(k*x)
def v(x,k):
    return f(x)*np.sin(k*x)
def w(x,k):
    return g(x)*np.cos(k*x)
def z(x,k):
    return g(x)*np.sin(k*x)

# We initialize 26,25 element arrays containing all 0's intially for storing Fourier Coefficiets.
fa = np.zeros(26,)
fb = np.zeros(25,)
ga = np.zeros(26,)
gb = np.zeros(25,)


for i in range(26):			#We are calculating a_0,a_1,a_2,.....,a_25
    fa[i],_ = sp.integrate.quad(u,0,2*pi,(i,))		# a_k = integral(f(x)*cos(k*x)) from 0 to 2*pi
    ga[i],_ = sp.integrate.quad(w,0,2*pi,(i,))		# a_k = integral(g(x)*cos(k*x)) from 0 to 2*pi
    
for i in range(25):			#We are calculating b_1,b_2,.....,b_25
    fb[i],_ = sp.integrate.quad(v,0,2*pi,(i+1,))		# b_k = integral(f(x)*sin(k*x)) from 0 to 2*pi
    gb[i],_ = sp.integrate.quad(z,0,2*pi,(i+1,)) 		# b_k = integral(g(x)*sin(k*x)) from 0 to 2*pi

# Since there is a factor of (1/pi) multiplied to the integral(for all k != 0), we divide each Fourier Coefficient by pi.
fa /= pi
fb /= pi
ga /= pi
gb /= pi

# Since a_0 for f(x) and g(x) should be divided divided by 2*pi. We further divide the 1st coefficent by 2.
fa[0] /= 2
ga[0] /= 2

#We define an array F containing 26+25=51 elements in it which represents all the 51 Fourier Coefficients of e^x
F = [None]*(len(fa)+len(fb))
F[0] = fa[0]
F[1::2] = fa[1:]
F[2::2] = fb
F = np.asarray(F)
#If we wish to print the array of coefficients we can do so using the command print(F).

# We are plotting the the function e^x obtained from Fourier Coefficients in semilog scale. 
plt.grid()
plt.semilogy(abs(F),'o',color = 'r',markersize = 4)
plt.title('Semilog plot of Fourier Coefficients for e^x by direct integration')
plt.xlabel('n')
plt.ylabel('e^x Fourier Coefficients') 
plt.show()

# We are plotting the the function e^x obtained from Fourier Coefficients in log-log scale. 
plt.grid()
plt.loglog(abs(F),'o',color = 'r',markersize = 4)
plt.title('log-log plot of Fourier Coefficients for e^x by direct integration')
plt.xlabel('n')
plt.ylabel('e^x Fourier Coefficients') 
plt.show()

#We define an array G containing 26+25=51 elements in it which represents all the 51 Fourier Coefficients of cos(cos(x))
G = [None]*(len(ga)+len(gb))
G[0] = ga[0]
G[1::2] = ga[1:]
G[2::2] = gb
G = np.asarray(G)
#If we wish to print the array of coefficients we can do so using the command print(G).

# We are plotting the the function cos(cos(x)) obtained from Fourier Coefficients in semilog scale. 
plt.grid()
plt.semilogy(abs(G),'o',color = 'r',markersize = 4)
plt.title('Semilog plot of Fourier Coefficients for cos(cos(x)) by direct integration')
plt.xlabel('n')
plt.ylabel('cos(cos(x)) Fourier Coefficients') 
plt.show()

# We are plotting the the function cos(cos(x)) obtained from Fourier Coefficients in log-log scale. 
plt.grid()
plt.loglog(abs(G),'o',color = 'r',markersize = 4)
plt.title('log-log plot Fourier Coefficients for cos(cos(x)) by direct integration')
plt.xlabel('n')
plt.ylabel('cos(cos(x)) Fourier Coefficients') 
plt.show()


X = np.linspace(0,2*pi,401)
X = X[:-1]
b1 = f(X)	# Defining f(X) to b1
b2 = g(X)	# Defining g(X) to b2
A = np.zeros((400,51))
A[:,0] = 1

for k in range(1,26):
    A[:,2*k-1] = np.cos(k*X)
    A[:,2*k] = np.sin(k*X)

# We are finding the Fourier Coefficients of e^x and cos(cos(x)) using Least Square Method using the command np.linalg.lstsq
c1 = np.linalg.lstsq(A,b1,rcond = -1)[0]
c2 = np.linalg.lstsq(A,b2,rcond = -1)[0]

# We are plotting Fourier Coefficients of e^x calculated using Least Square Method.
plt.grid()
plt.plot(c1,'o',color = 'g')
plt.title('Fourier Coefficients of e^x calculated using Least Square Method')
plt.xlabel('n')
plt.ylabel('e^x Fourier Coefficients')
plt.show()

# We are calculating Fourier Coefficients of cos(cos(x)) calculated using Least Square Method.
plt.grid()
plt.plot(c2,'o',color = 'g')
plt.title('Fourier Coefficients of cos(cos(x)) calculated using Least Square Method')
plt.xlabel('n')
plt.ylabel('cos(cos(x)) Fourier Coefficients')
plt.show()

# Now we are plotting, Original function e^x and the same function generated by it's Fourier Coefficients.
final_f = A.dot(c1)
plt.grid()
plt.plot(X,final_f,'o',color = 'g')
plt.plot(X,f(X),'-',color = 'b')
plt.title(' Original and Reconstructed Function from Fourier Coefficients of e^x')
plt.xlabel('x')
plt.ylabel('e^x')
plt.legend(['original','reconstructed'])
plt.show()

# And now we are plotting, Original function cos(cos(x)) and the same function generated by it's Fourier Coefficients.
final_g = A.dot(c2)
plt.grid()
plt.plot(X,final_g,'o',color = 'g')
plt.plot(X,g(X),'-',color = 'b')
plt.title(' Original and Reconstructed Function from Fourier Coefficients of cos(cos(x))')
plt.xlabel('x')
plt.ylabel('cos(cos(x))')
plt.legend(['original','reconstructed'])
plt.show()

#Finally we are printing the MEAN ERROR between the original function and the function generated by their Fourier Coefficients.
print("Mean Error in f(x) = exp(x) is {}".format(np.mean(abs(c1 - F))))


print("Mean Error in g(x) = cos(cos(x)) is {}".format(np.mean(abs(c2 - G))))
