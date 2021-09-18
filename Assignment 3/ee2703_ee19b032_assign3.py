
"""
		EE2703: Applied Programming Lab
		Assignment - 3
		Prof. Harishankar Ramachandran
		
			Done By:
			KATARI HARI CHANDAN
			EE19B032
			
Note: The plots might vary if the input data is different, i.e., if generate_data.py is ran once we'll get some plots and if it is ran again then we'll get different plots.
"""


# We Import all the useful modules required to write the code.

import numpy as np
from matplotlib.pylab import *
import scipy.special as sp
import matplotlib.pyplot as plt
from statistics import stdev				
from random import random
from scipy.linalg import lstsq

#Q2)Load the file "fitting.dat" and extract the data

data=np.loadtxt("fitting.dat")		#We are aLoading the file 'fitting.dat'
x=data[:,0]					#We are extract the 1st column from the file , i.e; assigned 'time' to 'x'.
y=data[:,1:]					#We are extract the remaining columns, which contain f(t)+noise for various values of sigma assigned to the variable y.

def h_list(A,B):				#Defining the function h for any co-efficients A,B
	h_val= [A*sp.jn(2,t)+B*t for t in x] 
	return h_val

#Q4)Plot the function values with and without noise for different sigma values.

plt.figure(1)

plot(x,y)								#Plotting f(t) + noise for various sigma values
plot(x,h_list(1.05,-0.105),color='black')				#Plotting original value of the function without any noise
title(r'f(t) with noise for different sigma (Q4)')
xlabel(r'$time$',size=14)									
ylabel(r'$f(t)+noise$',size=14)
scl=np.logspace(-1,-3,9)
legend(scl) 
grid()  

#Q5)Plot the function and the error bars for sigma=0.10.

plt.figure(2)

noise=[]
for t in range(len(y[:,0])):
	noise.append(y[:,0][t]-h_list(1.05,-0.105)[t])		#We are Calculating noise for sigma=0.10
std_dev=stdev(noise)							#We are Calculating standard devation of the noise through numpy
plot(x,h_list(1.05,-0.105),color='black')				#Plotting original function
errorbar(x[::5],y[:,0][::5],std_dev,fmt='ro')				#Plotting error bars
title('Error Bars for sigma = 0.1 (Q5)')
xlabel(r'$t$',size=14)
legend(["f(t)","Error bars"])
grid()

#Q6)Set up the matrix M and check whether the matrix multiplication of M with column vector [A0,B0] is equal to the given function for random values of A0,B0?

Jn_func_values=[sp.jn(2,t) for t in x]				#sp.jn(2,t)= Bessel function of 2nd order
M=c_[Jn_func_values , x]						#Setting up the matrix M


A0=random();B0=random()			#We are Creating some random values of A,B to check whether h(t,A0,B0) is equal to M * column vector [A0,B0]
A_B_matrix=[A0,B0]
A_B_matrix=c_[A_B_matrix]

h_matrix=np.dot(M,A_B_matrix)

for n in range(len(x)):
	if h_matrix[n][0]!=h_list(A0,B0)[n]:
		flag=False
		break		
if(flag):
	print("The Output vector obtained by multiplying matrix M with column matrix [A0  B0]  is h(t,A0,B0).")
	
else:
	print("The Output vector obtained by multiplying matrix M with column matrix [A0  B0]  is NOT h(t,A0,B0).")
	
#Q7)Calculate the error values.
				
A=[i/10 for i in range(21)]						#Set up the lists A,B as specified.
B=[j/100 for j in range(-20,1)]
E=[]
for i in range(len(A)):
	E.append([])
	for j in range(len(B)):
		E[i].append(0)
		E[i][j]=((y[:,0]-h_list(A[i],B[j]))**2).mean()				#We are Calculating the mean squared error.
			

#Q8)Plot the contour plot of E[i][j]

plt.figure(3)
		
contour(A,B,E,[0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5])		#Plotting the contour plot of the mean squared error
title('Contour Plot of Error Eij for sigma=0.1(Q8)')
xlabel(r'$A$',size=14)
ylabel(r'$B$',size=14)
grid()

#Q9)Obtain the best estimate of A,B using the lstsq function

estimate=[]
for i in range(len(y[0])):
	p,resid,rank,sig=lstsq(M,y[:,i])
	estimate.append(p)									#We are Calculating the estimated values of A,B using lstsq function
	
A_error=[abs(estimate[i][0]-1.05) for i in range(len(estimate))]				#A_error = absolute value of (estimate(A) - actual value of A)
B_error=[abs(estimate[i][1]+0.105) for i in range(len(estimate))]				#B_error = absolute value of (estimate(B) - actual value of B)

plt.figure(4)

#Q10)Plot the error in approximation of A,B for different data files versus noise sigma

plot(scl,A_error,'r--')									#Plotting A_error versus sigma
plot(scl,B_error,'g--')									#Plotting B_error versus sigma
title("Error in A and B(Q10)")
xlabel(r'$Noise standard deviation$',size=10)
ylabel(r'MS Error',size=10)									
legend(["Aerr","Berr"])
grid()

#Q11)Give the above plot in log-log scale
plt.figure(5)

loglog(scl,A_error,'r--')									#Plotting A_error vs. sigma in log-log scale
loglog(scl,B_error,'b--')									#Plotting B_error vs. sigma in log-log scale
legend(["Aerr","Berr"])
title("Variation Of error with Noise in log-log scale(Q11)")
xlabel(r'$\sigma_n$',size=10)
ylabel(r'MS Error',size=10)
grid()

plt.show()											#All the plots are shown
