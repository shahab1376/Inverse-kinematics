
# Solving the Inverse Kinematics problem using Particle Swarm Optimization

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import time

PopSize = 200
dim = 4        # Dimension of X
iters = 500

# Target position  ( X = -2 , Y = 2 , Z = 3)
target = np.array([-2,2,3])

# PSO parameters
Vmax=17
wMax=1
wMin=0.01
c1=2
c2=2

# Upper and lower bounds
lb = [-0.8611*np.pi , -0.8055*np.pi , -200 , -2*np.pi ]
ub = [0.8611*np.pi  ,  0.8055*np.pi , 200 ,  2*np.pi ]


def Distance(x, target):

    # Input parameters
    varin = x.copy()

    tetha = [0 for i in range(4)]

    for i in range(len(tetha)):
        if i != 2:
            tetha[i] = varin[i]
    d2 = varin[2]

    # D-H Table parameters
    d0 = 566 
    alfa0=0    *(np.pi/180)
    l0 = 125
    alfa1=314  *(np.pi/180)
    l1 = 255
    d1=0
    alfa2=0    *(np.pi/180)
    l2=0
    d3 = 6
    alfa3=0    *(np.pi/180)
    l3=0

    # Define transform matrices
    T0 = np.array([[np.cos(tetha[0]), -np.sin(tetha[0])*np.cos(alfa0),0,l0*np.cos(tetha[0])],
                   [np.sin(tetha[0]), np.cos(tetha[0]),0,l0*np.sin(tetha[0])],
                   [0 , 0 ,1 , d0],
                   [0 ,0 , 0 , 1]])


    T1 = np.array([[np.cos(tetha[1]), -np.sin(tetha[1])*np.cos(alfa1),0,l1*np.cos(tetha[1])],
                  [np.sin(tetha[1]), -np.cos(tetha[1]),0,l1*np.sin(tetha[0])],
                  [0 , 0 , -1 , 0],
                  [0 ,0 , 0 , 1]])

    T2 = np.array([[1 , 0 , 0 , 0],
                  [0 , np.cos(tetha[2])*np.cos(alfa2), 0 ,0],
                  [0 , np.sin(alfa2), 1  ,d2],
                  [0 ,0 , 0 , 1]])

    T3 = np.array([[np.cos(tetha[3]), -np.sin(tetha[3])*np.cos(alfa3),0,0],
                  [np.sin(tetha[3]), np.cos(tetha[3])*np.cos(alfa3),0,0],
                  [0 , 0 ,0 , d3],
                  [0 ,0 , 0 , 1]])

    T4=np.array([1,1,1,1])

    # Find final transform matrix!
    Ttotal = np.matmul(np.matmul(np.matmul(np.matmul(T0,T1),T2),T3),T4.T)

    # Find distance between end tip position and target position
    x = Ttotal[0];
    y = Ttotal[1];
    z = Ttotal[2];
    # Output the distance
    out = np.sqrt((target[0] - x)**2+(target[1] - y)**2+(target[2] - z)**2)
    
    return out,Ttotal[:3]


# Creating solution class
class solution():
    pass

# Defining functions




s=solution()

# Initializations

vel=np.zeros((PopSize,dim))

pBestScore=np.zeros(PopSize) 
pBestScore.fill(float("inf"))

pBest=np.zeros((PopSize,dim))
gBest=np.zeros(dim)


gBestScore=float("inf")

pos = np.zeros((PopSize, dim))
for i in range(dim):
    pos[:, i] = np.random.uniform(0,1, PopSize) * (ub[i] - lb[i]) + lb[i]

convergence_curve=np.zeros(iters)

print ('\nParticle Swarm Optimization\n')
print ('PARAMETERS\n','-'*9)
print ('Population size : ', PopSize)
print ('Dimensions      : ', dim)
print ('c1              : ', c1)
print ('c2              : ', c2)
print ('function        : ',Distance.__name__)


print("\nPSO is optimizing  \""+Distance.__name__+"\"")    

timerStart=time.time() 
s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")

for l in range(0,iters):
    for i in range(0,PopSize):
        #pos[i,:]=checkBounds(pos[i,:],lb,ub)
        for j in range(dim):
            pos[i, j] = np.clip(pos[i,j], lb[j], ub[j])
        #Calculate objective function for each particle
        fitness,_ = Distance(pos[i],target)

        if(pBestScore[i]>fitness):
            pBestScore[i]=fitness
            pBest[i]=pos[i].copy()

        if(gBestScore>fitness):
            gBestScore=fitness
            gBest=pos[i].copy()

    #Update the W of PSO
    w=wMax-l*((wMax-wMin)/iters);
    
    # Update position of each particles
    for i in range(0,PopSize):
        for j in range (0,dim):
            r1=np.random.random()
            r2=np.random.random()
            vel[i,j]=w*vel[i,j]+c1*r1*(pBest[i,j]-pos[i,j])+c2*r2*(gBest[j]-pos[i,j])

            if(vel[i,j]>Vmax):
                vel[i,j]=Vmax

            if(vel[i,j]<-Vmax):
                vel[i,j]=-Vmax

            pos[i,j]=pos[i,j]+vel[i,j]

    convergence_curve[l]=gBestScore

    if (l%1==0):
        print('At iteration '+ str(l+1)+ ' the best fitness is '+ str(
            gBestScore));
timerEnd=time.time()  
s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
s.executionTime=timerEnd-timerStart
s.convergence=convergence_curve
s.best = pos[-1]
s.optimizer="PSO"
s.params = Distance(s.best,target)[1]

print ('\nRESULTS\n', '-'*7)
print ('gbest fitness   : ', s.best)
print ('gbest params    : ', s.params)
print ('iterations      : ', iters)
print ('time elapsed    : ', s.executionTime)

# Plot Results
plt.plot(convergence_curve)