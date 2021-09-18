
"""
		EE2703: Applied Programming Lab
		Assignment - 5
		Prof. Harishankar Ramachandran
		
			Done By:
			KATARI HARI CHANDAN
			EE19B032
"""

from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
from numpy import *
from sys import *

if(len(sys.argv)==5):
	Nx=int(sys.argv[1]) #  X length
	Ny=int(sys.argv[2]) #  Y 
	radius=int(sys.argv[3]) #radius of central lead
	Iter=int(sys.argv[4]) #number of iterations to perform
elif(len(sys.argv)==1):
    Nx=25
    Ny=25
    radius=8
    Iter=1500
	
else:
    print("Give proper arguments")
    exit()
	

Old_Phi=zeros((Nx,Ny),dtype=float)                                      # Creating Matrices for Potentials
New_Phi=zeros(Old_Phi.shape)


X=linspace(-0.5,0.5,Nx)
Y=linspace(-0.5,0.5,Ny)
x,y=meshgrid(X,Y)
i=where((x)**2+(y)**2<0.32**2)
New_Phi[i]=1                                                            #Intializing the Wire's Potential
error=[]

contour(y,x,New_Phi)                                                    #Intial Contour
plot((i[0]-Nx//2)/Nx,(i[1]-Ny//2)/Ny,'ro',markersize=2)
title("Intial Contour Plot")
xlabel('x')
ylabel('y')
show()

for k in range(Iter):
    Old_Phi=New_Phi.copy()
    New_Phi[1:-1,1:-1]=0.25*(Old_Phi[0:-2,1:-1]+Old_Phi[2:,1:-1]+Old_Phi[1:-1,0:-2]+Old_Phi[1:-1,2:])           #Top,Bottom,Left,Right members subarray Respectively
    New_Phi[i]=1         
    New_Phi[1:-1,-1]=New_Phi[1:-1,-2]        #Top Boundary
    New_Phi[0,1:-1]=New_Phi[1,1:-1]         #Right Boundary 
    New_Phi[-1,1:-1]=New_Phi[-2,1:-1]       #Left Boundary
    error.append((New_Phi-Old_Phi).max())
   
Mat1=zeros((int(2*Iter/3),2),dtype=float)         # Coefficient Matrices For Fitting Mat1 for 500 Iter, Mat2 for 1500 iter
Mat2=zeros((int(Iter),2))

Mat1[:,0]=1
try:
    Mat1[:,1]=array(range(int(Iter/3),int(Iter)) )           #Intializing them
except Exception as e:
    print("Give iteration number as multiple of 3",e)
    exit()

Mat2[:,0]=1
Mat2[:,1]=array(range(1,Iter+1))

error1=array(error[int(Iter/3):])
error2=array(error[:Iter])

Sol1=lstsq(Mat1,log(error1),rcond=None)[0]          #Solution for the fitted data
Sol2=lstsq(Mat2,log(error2),rcond=None)[0]
print(Sol1,Sol2)

semilogy(range(1,int(Iter)+1,50),(e**Sol1[0])*e**(Sol1[1]*range(1,int(Iter),50)),markersize=6,label='Fit1')
semilogy(range(1,Iter+1,50),(e**Sol2[0])*e**(Sol2[1]*range(1,Iter,50)),markersize=4,label='Fit2')
semilogy(range(1,Iter+1,50),error[::50],markersize=2,label='Error')
title('Error Plots')
xlabel('Iteration Number')
ylabel('Error')
legend()
show()

semilogy(range(1,int(Iter)+1,50),(e**Sol1[0])*e**(Sol1[1]*range(1,int(Iter),50)),'bo',markersize=6,label='Fit1')
semilogy(range(1,Iter+1,50),(e**Sol2[0])*e**(Sol2[1]*range(1,Iter,50)),'yo',markersize=4,label='Fit2')
semilogy(range(1,Iter+1,50),error[::50],'ro',markersize=2,label='Error')
title('Error Plots')
xlabel('Iteration Number')
ylabel('Error')
legend()
show()

fig1=contour(y,x,New_Phi,10)
clabel(fig1,fontsize=5)                         #Contour Plot
plot((i[0]-Nx//2)/Nx,(i[1]-Ny//2)/Ny,'ro',markersize=2)
title('Contour Plot of Potential')
xlabel('x')
ylabel('y')
show()

fig2=figure(1)                                  #Surface Plot
ax=p3.Axes3D(fig2)                      
surf=ax.plot_surface(y,x,New_Phi.T,rstride=1,cstride=1,cmap=cm.jet)
title('Surface Plot of Potential')
xlabel('x')
ylabel('y')
show()

Jx=zeros(New_Phi.shape)                                             #Current plotting
Jy=zeros(New_Phi.shape)
Jx[1:-1,1:-1]=0.5*(New_Phi[1:-1,0:-2]-New_Phi[1:-1,2:])
Jy[1:-1,1:-1]=0.5*(New_Phi[0:-2,1:-1]-New_Phi[2:,1:-1])
quiver(y,x,Jy[::-1,:],Jx[::-1,:],scale=7)
plot((i[0]-Nx//2)/Nx,(i[1]-Ny//2)/Ny,'ro',markersize=2)                                  #Dots for Central Wire
title('J Profile')
xlabel('x')
ylabel('y')
show()


