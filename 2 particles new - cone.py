#LIBRARIES

import numpy as np
import math as mt
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import itertools
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams["text.usetex"]=True


#VALUE DEFINITION

s=19         #dimension of lattice
n=2         #number of particles
a=5         #initial position
mu=1        #value of mu
tau=1       #value of tau
U=1         #value of U

t=0         #initial time
t_f=10      #final time
nt=500      #number of iterations

d=mt.comb(s+n-1,n) #dimension of the Hilbert space

#OPERATOR MATRICES

#Hilbert space basis

def generate_vector_basis(dimension,particles):
    basis=[]
    for combo in itertools.combinations_with_replacement(range(dimension),particles):
        vector=[0]*dimension
        for index in combo:
            vector[index]+=1
        basis.append(vector)
    return basis

basis=np.array(generate_vector_basis(s,n))
file_path="basis.txt"
np.savetxt(file_path,basis,fmt='%.0f')

#Number operator matrices

n_op={}
for i in range(s):
    name=f"n_{i+1}"
    matrix=np.zeros((d,d))
    for j in range(d):
        vector=basis[j]
        matrix[j,j]=np.sqrt(vector[i])
    n_op[name]=matrix

file_path="n_op.txt"
np.savetxt(file_path,n_op["n_4"],fmt='%.4f')

#Hamiltonian operator matrix

H_int=np.zeros((d,d))                       #interaction terms
for vector in basis:
    index=np.where((basis==vector).all(axis=1))
    for value in vector:
        if value > 1:
            H_int[index[0],index[0]]+=value*(value-1)

H_hop=np.zeros((d,d))                       #hopping terms
shift_v1=np.zeros(s)
shift_v2=np.zeros(s)
for v1 in basis:
    index1=np.where((basis==v1).all(axis=1))
    for v2 in basis:
        if (v1==v2).all():
            continue
        index2=np.where((basis==v2).all(axis=1))
        for i in range(s):
            shift_v1[:]=v1[:]
            shift_v1[i]-=1
            for j in range(s):
                if (j!=i+1) or (j!=i+1):
                    continue
                shift_v2[:]=v2[:]
                shift_v2[j]-=1
                if (shift_v1==shift_v2).all():
                    H_hop[index1,index2]=np.sqrt(np.dot(basis[index1],np.transpose(basis[index2])))
                    H_hop[index2,index1]=np.sqrt(np.dot(basis[index1],np.transpose(basis[index2])))

H=tau*H_hop+U/2*H_int                       #Hamiltonian
file_path="H.txt"
np.savetxt(file_path,H,fmt='%.3f')

#Data storage matrices

psi_t=np.empty((nt+1,d),dtype=complex)
time=np.empty((nt+1,1))
prob=np.empty((nt+1,s))

#INITIAL STATE

psi_0_old=np.zeros(s)
psi_0_old[9]=2
psi_0_old[0]=0

psi_0=np.zeros((d,1))
index=np.where((basis==psi_0_old).all(axis=1))
psi_0[index[0]]=1

#COMPUTATION

#Time evolution of \Psi

for k in range(nt+1):
    psi_tt=(np.dot(expm(-1j*t*H),psi_0))
    psi_t[k,:d]=np.transpose(psi_tt)
    time[k,0]=t
    t=(k+1)*t_f/nt

#Expected value <n>:

    for j in range(s):
        prob[k,j]=np.abs(np.vdot(np.dot(np.transpose(np.conj(psi_tt)),n_op[f"n_{j+1}"]),psi_tt))

#Data storage

data=np.concatenate(((prob),time),axis=1)
file_path="data.txt"
np.savetxt(file_path,data,fmt='%.8f')

#PLOT

colormap=plt.get_cmap()             #type of map
norm=plt.Normalize(0,1)             #normalize probabilities
fig,ax=plt.subplots()

for i in range(data.shape[1]-1):    #iterate over "pictures" in time
    x=data[:,i]                     #probability values
    y=data[:,s]                     #time values

    colors=colormap(norm(x))        #convert probabilities to colors based on the colormap

    sc=ax.scatter(np.repeat(i+1,len(x)),y,s=100,c=x,cmap='binary',label=f'{i+1}',marker='_',alpha=1)

vmin,vmax=sc.get_clim()

ax.set_ylabel('Time',fontsize=15)
ax.set_xticks(np.arange(data.shape[1]-1)+1)
ax.set_xticklabels([r'$\langle n_{%d} \rangle$' % (i+1) for i in range(data.shape[1]-1)],fontsize=15) #name the columns
ax.set_xticks(ax.get_xticks()[::2])

cbar=fig.colorbar(sc,ax=ax,ticks=[0,vmax/2,vmax])
cbar.ax.set_yticklabels(['0 particles','1 particle','2 particles'],fontsize=12)

#CONE PLOT

time=np.arange(0,t_f,t_f/nt)
space=np.linspace(0,s+1,2*s+1)
cone=np.zeros((len(time),len(space)))

C=10
v_0=3.59*(np.linalg.norm(H_hop,np.inf)/2)*tau
D=2
v=v_0+D*tau

# Fill cono array
for i, t in enumerate(time):
    for j, s_val in enumerate(space):
        # Calculate the distance from the center
        dist_center=abs((s+1)/2-s_val)
        if C*n*np.exp(v*t-dist_center)>1:
            cone[i,j]=1
        else:
            cone[i,j]=C*n*np.exp(v*t-dist_center)

colors=[(0,0,1),(1,1,1)]
cmap_blue=LinearSegmentedColormap.from_list('CustomMap',colors)

ax.imshow(cone[::-1,:],extent=[space[0],space[-1],time[0],time[-1]],aspect='auto',cmap=cmap_blue,alpha=1)

plt.grid(True)
plt.show()