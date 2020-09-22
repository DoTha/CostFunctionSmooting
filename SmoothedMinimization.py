from __future__ import division
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
rng = np.random
import scipy.sparse.linalg as ss

class SmoothedMinimization(object):
    
    def __init__(self,Iterations, T, dt, N, M,D, xinit,delta,epsilon = 1.,gamma=1.,snoise = 0):
        #parameters
        self.Iterations = Iterations

        self.T = T #Horizont
        self.dt = dt
        self.N = N #number of rollouts

        self.M = M #number of basis functions
        self.D = D #dimensions of the dynamical system
        self.xinit = xinit #initial condition of the system
        self.gamma = gamma
        
        self.delta = delta #particle number independent weight entropy
        
        self.epsilon = epsilon #stepsize
        #delta2 = 100

        self.ln=int(T/dt)
        self.timeline = np.linspace(0,T,self.ln+1)
        self.intline = np.arange(self.ln)
        
        
        self.Costs = np.zeros([self.Iterations,N])
        self.VCosts = np.zeros([self.Iterations,N])
        
        self.weights = np.zeros([self.Iterations,N])
        
        #self.xiterations = np.zeros([self.Iterations,self.ln+1,N,D]) #container to store the whole trajectories
        self.noise = rng.randn(self.ln+1,N)/np.sqrt(dt)
        self.snoise = snoise
        
    def run(self,mode = "PICE",ls = 0,verbose = 0,maxit = -1,learningrate=0,cemethod=0):
        
        N = self.N
        M = self.M
        D = self.D
        Iterations = self.Iterations
        print(mode)
        #make definitions
        x = np.zeros([self.ln+1,N,D])
        S = np.zeros([1,N])
        VC = np.zeros([1,N])
        xiPhi = np.zeros([self.ln,M,N])
        Psi = np.zeros([self.ln,M])
        grad = np.zeros([self.ln,M])

        ESStrue = np.zeros(Iterations)
        ESS = np.zeros(Iterations)
        EntESStrue = np.zeros(Iterations)
        EntESS = np.zeros(Iterations)
        totalCost = np.zeros(Iterations)
        gammas = np.zeros(Iterations)

        for j in np.arange(Iterations):

            if verbose == 1:
                #print(j,flush=True)
                print(j)

            #do rollouts
            totalCost[j], GN, noise = self.dorollouts(x,Psi,xiPhi,S,VC)
            w, wtrue, gammas[j] = self.ComputeTRweights(S)
            #print("totalCost")
            #print(totalCost[j])
            if mode == "TRPO": #direct implementation of TRPO to look if it is really more or less the same as the limit of small TR
                
                Rew = -(S-S.mean());
                for i in self.intline:
                    meanGN = np.sum(GN[i],axis=2) #computes fisher matrix
                    if maxit==-1:
                        grad[i] = np.dot(np.linalg.pinv(meanGN, rcond=1e-4),np.dot(Rew,xiPhi[i].T)[0,:])
                    else:
                        puregrad = np.dot(Rew,xiPhi[i].T)[0,:]
                        grad[i] = ss.cg(meanGN,puregrad,maxiter = maxit,x0 = grad[i])[0]
            elif mode == "naturalPICE":
                Rew = (w-w.mean());
                for i in self.intline:
                    meanGN = np.sum(GN[i],axis=2) #computes fisher matrix
                    if maxit==-1:
                        grad[i] = np.dot(np.linalg.pinv(meanGN, rcond=1e-4),np.dot(Rew,xiPhi[i].T)[0,:])
                    else:
                        puregrad = np.dot(Rew,xiPhi[i].T)[0,:]
                        grad[i] = ss.cg(meanGN,puregrad,maxiter = maxit,x0 = grad[i])[0]
                    
                    
            beta, b, c = self.LineSearch(grad,x,Psi,noise,GN,totalCost[j],cemethod)
            
            Psi += beta*grad

            #Compute evaluations:
            ESStrue[j] = np.sum(wtrue)**2./np.dot(wtrue,wtrue.T)
            ESS[j] = np.sum(w)**2./np.dot(w,w.T)
            #EntESStrue[j] = 1-np.dot(wtrue*N,np.log(wtrue.T*N+1/N*0.001))/(N*np.log(N))
            EntESStrue[j] = -np.dot(wtrue,np.log(wtrue.T+1/N*0.001))/(np.log(N))
            #EntESS[j] = 1-np.dot(w*N,np.log(w.T*N+1/N*0.001))/(N*np.log(N))
            EntESS[j] = -np.dot(w,np.log(w.T+1/N*0.001))/np.log(N)
            self.Costs[j,:] = S
            self.VCosts[j,:] = VC
            #self.xiterations[j,:,:,:]=x
            self.weights[j,:] = w
            #print(ESStrue[j])
            
        #output
        self.ESStrue = ESStrue
        self.ESS = ESS
        self.EntESStrue = EntESStrue
        self.EntESS = EntESS
        self.totalCost = totalCost
        self.gammas = gammas
        self.x = x
        
        
        
        
    
    def basisfunctions(self,x):
        arrays = [x[:,i] for i in range(self.D)]
        arrays.append(np.ones(x[:,0].shape))
        return np.stack(arrays)

    def V(self,x,T):
        #gives back the cost at each point in time. gets x which is a matrix that has N and D
        raise NotImplementedError
        return 0
    
    def dynamics(self,x,uplusnoise):
        raise NotImplementedError
        return x

    def GaussNewton(self,Phi):
        return np.einsum("ik,jk->ijk",Phi,Phi)

    def ComputeTRweights(self,S):
        
        #usage of global variables
        N = self.N
        delta = self.delta
        
        gamma = 1
        KL = 2*delta
        while KL>delta:
            S = S/gamma #renormalize the action according to the trust region
            eS = np.exp(-(S-S.min())); #compute the unnormalized weights in a nummerically stable manner
            Z = N*eS.mean() #compute the parition sum/ normalization constant
            w = eS/Z #normalize the weights

            if gamma ==1:
                wtrue = w

            KL = np.log(N)+np.dot(w,np.log(w.T+1/N*0.0001)) #compute KL divergence between current dynamics and optimal dynamics
            gamma = 1.05
            #print KL, w.min(), w.max()
            
            
        #print KL
        return w, wtrue, gamma #return the weights such that they sum up to 1

    def LineSearch(self,grad,x,Psi,noise, GN,totalCost,cemethod = 0): 
        
        #usage of global variables
        epsilon = self.epsilon
        
        gradTFgrad = 0.
        for i in self.intline:
            F = np.mean(GN[i],2) #compute the fisher
            gradTFgrad += np.einsum("j,jk,k",grad[i],F,grad[i])
        beta = np.sqrt(2*epsilon/gradTFgrad) #computes an upper bound for the stepsize by using the fisher as a proxy to the KL divergence. This follows the TRPO paper
        
        i = 1
        KL=2*epsilon
        while KL>epsilon: #this does the actual linesearch. beta is decreases until the cost is increasing and the KL divergence is smaller epsilon
            Psinew = Psi + beta*grad
            KL = self.offpolicyrollouts(x,Psi,Psinew,noise)
            beta = beta*0.9
            beta1 = beta
            
            #print([beta,totalCostnewt,totalCost,benchmarkcost])
        #print(totalCostnew)
        return beta, beta1, i



    def dorollouts(self,x,Psi,xiPhi,S,VC):
        
        #
        intline = self.intline
        dt = self.dt
        N = self.N
        T = self.T
        
        Cost = 0
        x.fill(0.)
        x[0,:,:]=self.xinit        
        
        S.fill(0)
        VC.fill(0)
        xiPhi.fill(0)
        if self.snoise == 0:
            noise = rng.randn(self.ln+1,N)/np.sqrt(dt)
        else:
            noise = self.noise
        GN = np.zeros([self.ln,self.M,self.M,self.N])
        for i in intline:
                Phi = self.basisfunctions(x[i,:,:])
                #u= np.dot(Psi,Phi)
                u= np.dot(Psi[i,:],Phi) 
                x[i+1,:,:] = self.dynamics(x[i,:,:],u+noise[i,:])
                xiPhi[i] = noise[i,:]*Phi*dt
                pC = (self.V(x[i,:,:],i)+self.gamma*0.5*u*u)*dt
                Cost += np.mean(pC)
                S += pC+self.gamma*u*noise[i,:]*dt
                VC += self.V(x[i,:,:],i)*dt
                GN[i] = self.GaussNewton(self.basisfunctions(x[i,:,:]))*dt
        return Cost, GN, noise


    def offpolicyrollouts(self,x,Psiold,Psinew,noise): #computes the cost for a new Psinew  and the KL divergence between Psinew and Psiold
        dt = self.dt
        intline = self.intline
        T = self.T
        N = self.N
        
        KL = 0
        Cost = 0
        CE = 0.
        CEold = 0.
        S = np.zeros([1,N])
        SV = np.zeros([1,N])
        for i in intline:
            Phi = self.basisfunctions(x[i,:,:])
            unew= np.dot(Psinew[i,:],Phi)
            uold= np.dot(Psiold[i,:],Phi)
            #KL += np.mean(dt*0.5*(unew-uold)**2) #compute the KL divergence between the new and the old policy
            KL += dt*0.5*(unew-uold)**2 #compute the KL divergence between the new and the old policy
 
        totalKL2 = np.mean(KL.T)
        return totalKL2
    
    
    def plot(self):
        plt.subplot(2, 2, 1)
        plt.plot(self.timeline,-np.cos(self.x[:,1:10,1]));
        #plt.plot(self.Psiseries[:,1],self.Thetaseries[:,0])
        plt.subplot(2, 2, 2)
        plt.plot(self.EntESS);
        plt.plot(self.EntESStrue);
        plt.subplot(2, 2, 3)
        plt.plot(self.ESStrue);
        plt.plot(self.ESS);
        plt.subplot(2, 2, 4)
        plt.plot(self.totalCost);