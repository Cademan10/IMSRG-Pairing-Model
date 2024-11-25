import numpy as np
import scipy
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


########################################################################################################################
########################################################################################################################
######################################## IMSRG #########################################################################
########################################################################################################################
########################################################################################################################

class IMSRG:

	def __init__(self,N,g,singlePartEnergy):
		self.N = N
		self.singlePartEnergy = singlePartEnergy
		self.g = g

		self.J = (self.N-1)/2+1
		self.mult = int(2*self.J+1)
		self.omega = self.mult/2

		self.ms = np.array([m-self.J for m in range(self.mult)])
		self.eps = np.abs(self.ms)*self.singlePartEnergy


		self.runIMSRG()

	def runIMSRG(self):
		eps = self.eps
		N = self.N
		mult = self.mult
		omega = self.omega
		g = self.g
		

		
		f0 = eps-(N/(4*omega))*(2*g)
		gamma0 = g*np.tile([np.tile([-1,1],int(mult/2)),np.tile([1,-1],int(mult/2))],(int(mult/2),1))
		E0 = np.sum(eps)*N/(2*omega)-(g*N/4*(2*omega-N+2))
									 
		flowEqs0 = np.insert(np.concatenate((f0,gamma0.flatten())),0,E0)

		self.dGds = np.zeros((mult,mult))

		s_span = (0,5)
				#s = np.linspace(0,10,10)

		#solve_ivp_rk45 = solve_ivp(flow, s_span, flowEqs0)


				#print(E0)
		#sol = odeint(flow, flowEqs0, s,tfirst=True)

		sol = solve_ivp(self.flow,s_span,flowEqs0)

		print("Solution found!!!")
		#print(E0)

		self.predictedEnergy = -g/4*2*N*(omega+1)+g/4*N**2+N/(2*omega)*np.sum(eps)
		self.imsrgEnergy = sol.y[0][-1]
		self.imsrgConverge = sol.y[0]
		self.s_vals = sol.t

		print("Initial energy:",self.predictedEnergy)
		print('IMSRG energy = ', self.imsrgEnergy)








	def Delta(self,a,b):

		return int(a==b)


	def Lamb1(self,a,b):

		 return self.N/(2*self.omega)*self.Delta(a,b)
	 

	def Lamb2(self,a,b,c,d):

		N = self.N
		omega = self.omega
		delta = self.Delta

		term1 = 1/(4*omega*(omega-1))*(N*(N-2)*(delta(-a,-c)*delta(-b,-d)-delta(-a,-d)*delta(-b,-c))+N*(N-2*omega)*(-1)**(-a-c)*(delta(-a,b)*delta(-d,c)))
		term2 = (N/(2*omega))**2*(delta(a,d)*delta(b,c)-delta(a,c)*delta(b,d))
		
		return term1+term2

	def AntiSym3(self,lambi,lambj,lambk):
		a,d = lambi
		b,e = lambj  
		c,f = lambk

		lamb1 = self.Lamb1
	
		

		return (lamb1(a,d)*lamb1(b,e)*lamb1(c,f)-
				lamb1(a,e)*lamb1(b,d)*lamb1(c,f)-
				lamb1(a,f)*lamb1(b,e)*lamb1(c,d)-
				lamb1(a,d)*lamb1(b,f)*lamb1(c,e)+
				lamb1(a,e)*lamb1(b,f)*lamb1(c,d)+
				lamb1(a,f)*lamb1(b,d)*lamb1(c,e))
	  
	def AntiSym2(self,lambi,lambj):

		omega = self.omega
		lamb1 = self.Lamb1
		lamb2 = self.Lamb2
	

		if omega==self.N/2:
			return 0

		a,d = lambi
		
		b,c,e,f = lambj


		term1 =   (lamb1(a,d)*(lamb2(b,c,e,f)-lamb2(c,b,e,f)+lamb2(c,b,f,e)-lamb2(b,c,f,e))
				  -lamb1(a,e)*(lamb2(b,c,d,f)-lamb2(c,b,d,f)+lamb2(c,b,f,d)-lamb2(b,c,f,d))
				  +lamb1(a,f)*(lamb2(b,c,d,e)-lamb2(c,b,d,e)+lamb2(c,b,e,d)-lamb2(b,c,e,d)))


		term2 = (lamb1(b,d)*(lamb2(a,c,e,f)-lamb2(c,a,e,f)+lamb2(c,a,f,e)-lamb2(a,c,f,e))
				-lamb1(b,e)*(lamb2(a,c,d,f)-lamb2(c,a,d,f)+lamb2(c,a,f,d)-lamb2(a,c,f,d))
				+lamb1(b,f)*(lamb2(a,c,d,e)-lamb2(c,a,d,e)+lamb2(c,a,e,d)-lamb2(a,c,e,d)))

		term3 = (lamb1(c,d)*(lamb2(a,b,e,f)-lamb2(b,a,e,f)+lamb2(b,a,f,e)-lamb2(a,b,f,e))
				-lamb1(c,e)*(lamb2(a,b,d,f)-lamb2(b,a,d,f)+lamb2(b,a,f,d)-lamb2(a,b,f,d))
				+lamb1(c,f)*(lamb2(a,b,d,e)-lamb2(b,a,d,e)+lamb2(b,a,e,d)-lamb2(a,b,e,d)))
				 


		return (term1-term2+term3)

	def P(self,a):
		return (-1)**(self.J-a)
		

	def Lamb3(self,a,b,c,i,j,k):

		N = self.N
		omega = self.omega

		Anti3 = self.AntiSym3
		Anti2 = self.AntiSym2
		delta = self.Delta
		p = self.P

		if N==2:
			return 0


		f = i
		e = j
		d = k
		   
		 

		if omega==0 or omega==1:
			alpha = 0
			beta = 0
		elif omega==2:
			alpha = 1/omega*1/(2*(omega-1))
			beta = 0
		else:
			alpha = 1/omega*1/(2*(omega-1))

			beta = 1/omega*1/(2*(omega-1))*1/(3*(omega-2))


		abar,bbar,cbar,dbar,ebar,fbar = -a,-b,-c,-d,-e,-f


		preFac = p(a)*N/2*(N/2-1)
							   
								
		term1 = 2*p(fbar)*(delta(fbar,e)*delta(dbar,cbar)*p(dbar)+p(ebar)*(delta(d,ebar)*delta(cbar,fbar)-delta(d,fbar)*delta(ebar,cbar)))
						
										 

		A1 = p(dbar)*delta(fbar,e)*(delta(bbar,c)*delta(abar,dbar)-delta(c,abar)*delta(bbar,dbar))
		A2 = delta(c,bbar)*(delta(d,ebar)*delta(abar,fbar)-delta(abar,ebar)*delta(fbar,d))
		A3 = delta(c,abar)*(delta(d,ebar)*delta(bbar,fbar)-delta(bbar,ebar)*delta(fbar,d))
		
		term2 = 2*p(fbar)*(A1+p(ebar)*(A2-A3))
					 
					
					
		A = delta(ebar,d)*(-p(b)*delta(cbar,fbar)*delta(abar,b)-p(c)*(delta(c,bbar)*delta(abar,fbar)-delta(fbar,bbar)*delta(abar,c)))
		B = p(b)*delta(d,fbar)*delta(cbar,ebar)*delta(abar,b)-delta(d,fbar)*p(c)*(delta(ebar,bbar)*delta(abar,c)-delta(c,bbar)*delta(abar,ebar))
		C = p(dbar)*((delta(cbar,ebar)*(delta(bbar,dbar)*delta(abar,fbar)-delta(dbar,abar)*delta(bbar,fbar))
			 -delta(bbar,ebar)*(delta(cbar,dbar)*delta(abar,fbar)-delta(dbar,abar)*delta(cbar,fbar))
			 +delta(abar,ebar)*(delta(cbar,dbar)*delta(bbar,fbar)-delta(dbar,bbar)*delta(cbar,fbar))))
		gamma = A+B+C
		
		term3 = 6*p(fbar)*(-p(dbar)*p(b)*delta(abar,b)*delta(cbar,dbar)-p(dbar)*p(c)*(delta(abar,dbar)*delta(bbar,c)-delta(bbar,dbar)*delta(abar,c))+p(ebar)*gamma)
		   

		#print(preFac*(alpha*(delta(abar,b)*p(c)*term1+p(b)*term2)+beta*(N/2-2)*p(b)*p(c)*term3))
		return preFac*(alpha*(delta(abar,b)*p(c)*term1+p(b)*term2)+beta*(N/2-2)*p(b)*p(c)*term3)-Anti3((a,i),(b,j),(c,k))-Anti2((a,i),(b,c,j,k))


		
		
	def Eta1(self,i,j,gam):
		J = self.J
		return (-1/2*np.sum([gam[(m+J).astype(int),(j+J).astype(int)]*self.Lamb2(i,-j,m,-m)
		 -gam[(i+J).astype(int),(m+J).astype(int)]*self.Lamb2(m,-m,j,-i) for m in self.ms]))



	def Eta2(self,i,j,k,l,gam,f):

		lamb2 = self.Lamb2
		lamb3 = self.Lamb3
		ms = self.ms
		J = self.J

		fi = f[(i+J).astype(int)]
		fj = f[(j+J).astype(int)]
		fk = f[(k+J).astype(int)]
		fl = f[(l+J).astype(int)]
		
	    
		term1 =  fi*lamb2(i,j,k,l)-fj*lamb2(j,i,k,l)-fk*lamb2(i,j,k,l)+fl*lamb2(i,j,l,k)
	   

		return (term1+(1/2)*np.sum([gam[(m+J).astype(int),(k+J).astype(int)]*lamb3(-k,i,j,m,-m,l)
								   -gam[(m+J).astype(int),(l+J).astype(int)]*lamb3(-l,i,j,m,-m,k)
								   -gam[(i+J).astype(int),(m+J).astype(int)]*lamb3(m,-m,j,-i,k,l)
								   +gam[(j+J).astype(int),(m+J).astype(int)]*lamb3(m,-m,i,-j,k,l) for m in ms]))


	def flow(self,s,flowEqs):
		dGds = self.dGds
		ms = self.ms
		N = self.N
		omega = self.omega
		mult = self.mult
		J = self.J

		lamb3 = self.Lamb3
		lamb2 = self.Lamb2
		eta1 = self.Eta1
		eta2 = self.Eta2		

		E = flowEqs[0]
		f = flowEqs[1:mult+1]
		
		gam = np.reshape(flowEqs[mult+1:],(mult,mult))


		

		dEds = 1/4*np.sum([[dGds[int(J+MPrime),int(J+M)]*lamb2(M,-M,MPrime,-MPrime) for M in ms] for MPrime in ms])
		dEds += 1/4*np.sum([gam[int(b),int(a)]*(eta2(b,c,d,e,gam,f)*lamb3(c,a,-a,d,e,-b)-eta2(c,d,a,e,gam,f)*lamb3(-a,c,d,b,-b,e)) for a in ms for b in ms for c in ms for d in ms for e in ms])

		
		
		self.dGds = (np.array([[(eta1(m,m,gam)+eta1(-m,-m,gam)-eta1(mPrime,mPrime,gam)-eta1(-mPrime,-mPrime,gam))*gam[int(mPrime+J),int(m+J)] for m in ms] for mPrime in ms])
			   -np.array([[(f[int(J+m)]+f[int(J-m)]-f[int(J+mPrime)]-f[int(J-mPrime)])*eta2(m,-m,mPrime,-mPrime,gam,f) for m in ms] for mPrime in ms])
				+1/(2)*(1-N/omega)*np.array([[np.sum([eta2(m,-m,a,-a,gam,f)*gam[int(mPrime+J),int(a+J)]-eta2(a,-a,mPrime,-mPrime,gam,f)*gam[int(a+J),int(m+J)] for a in ms]) for m in ms] for mPrime in ms]))
	  

	   
		dFds = (N/(4*omega)*(1-N/(2*omega))*np.array([np.sum([eta2(m,-m,a,-a,gam,f)*gam[int(a+J),int(m+J)]-eta2(a,-a,m,-m,gam,f)*gam[int(m+J),int(a+J)] for a in ms]) for m in ms])
				+np.array([np.sum([gam[int(m+J),int(a+J)]*(1/4*eta2(m,-m,b,c,gam,f)*lamb2(a,-a,b,c)+eta2(m,b,a,c,gam,f)*lamb2(b,-a,c,-m))
				   - gam[int(a+J),int(m+J)]*(1/4*eta2(b,-a,m,-m,gam,f)*lamb2(b,c,a,-a)+eta2(a,b,m,c,gam,f)*lamb2(-m,b,-a,c))
				  +1/2*gam[int(b+J),int(a+J)]*eta2(m,c,m,a,gam,f)*lamb2(c,-a,b,-b)
				   -1/2*gam[int(a+J),int(c+J)]*eta2(m,a,m,b,gam,f)*lamb2(c,-c,b,a)
				   +1/2*gam[int(m+J),int(m+J)]*(eta2(b,c,-m,a,gam,f)*lamb2(b,c,-m,a)-eta2(-m,a,b,c,gam,f)*lamb2(-m,a,b,c)) 
					for a in ms for b in ms for c in ms]) for m in ms]))
		
		return np.insert(np.concatenate((dFds,self.dGds.flatten())),0,dEds) 





########################################################################################################################
########################################################################################################################
######################################## Richardson ####################################################################
########################################################################################################################
########################################################################################################################

class Richardson:
	def __init__(self,N,g,singlePartEnergy):
		self.N = N
		self.singlePartEnergy = singlePartEnergy
		self.g = g

		self.J = (self.N-1)/2+1
		self.mult = int(2*self.J+1)
		self.omega = self.mult/2

		self.ms = np.array([m-self.J for m in range(self.mult)])
		self.eps = np.abs(self.ms)*self.singlePartEnergy


		self.E0 = np.sum(self.eps)*self.N/(2*self.omega)-(self.g*self.N/4*(2*self.omega-self.N+2))


		self.solve()


		print("Richardson energy",self.E)


	def solve(self):
		self.Es =  fsolve(self.RichardsonEqs, ([self.E0 for i in range(int(self.N/2))]))

		self.E = np.sum(self.Es)



	def RichardsonEqs(self,Es):


		term2 = []
		for i in range(len(Es)):
			E = Es[i]
	
			enSum = 0
	
			for j in range(len(Es)):
				if E==Es[j]:
					pass
				else:
					enSum -= 2*self.g*1/(Es[j]-E)
			term2.append(enSum)




		return [1-self.g*np.sum(1/(2*self.eps-Es[i]))/2+term2[i] for i in range(len(Es))]









############################################################################
############################################################################
############################################################################



N = int(input("Number of nucleons:"))



eps = np.linspace(0,1,20)

g = 1
#eps = [.4]

imsrgEn = []
richEn = []
initEn = []
#Ground state from the diagonalized 2x2 Hamiltonian 
diagEn = []

relErrors = []

imsrgConv = []
s_vals = []

ep = 0.1

imsrg = IMSRG(N,g,ep)





for i in range(len(eps)):

	ep = eps[i]

	imsrg = IMSRG(N,g,ep)

	imsrgConv.append(imsrg.imsrgConverge)

	imsrgEn.append(imsrg.imsrgEnergy)
	rich = Richardson(N,g,ep)
	richEn.append(rich.E)
	initEn.append(rich.E0)
	diagEn.append(imsrg.eps[1]+imsrg.eps[0]-g-np.sqrt((imsrg.eps[1]-imsrg.eps[0])**2+g**2))

	print("Diagonalized Hamiltonian energy:",diagEn[-1])


	relErrors.append(abs((imsrgEn[-1]-diagEn[-1])/diagEn[-1]))

	s_vals.append(imsrg.s_vals)

	print()




fig,ax  = plt.subplots(2, sharex='col',figsize=(10,8),gridspec_kw={'height_ratios': [2, 1]})

ax[0].scatter(eps,initEn,c="g",marker="^",label="Initial ground state")
ax[0].scatter(eps,imsrgEn,c="b",label="IMSRG ground state")
ax[0].scatter(eps,richEn,c="r",marker="s",label="Richardson ground state")
ax[0].scatter(eps,diagEn,c="y",marker="o",label="Diagonalized Hamiltonian ground state")
ax[0].legend()
ax[0].set_title("Ground state energy vs energy spacing $(g=$"+str(g)+"$)$")
ax[0].set_ylabel("Energy")


ax[1].plot(eps,relErrors,"--o",color="k")
ax[1].set_xlabel("Single particle energy spacing")
ax[1].set_ylabel("Relative error \n $\\left|(E_{Diag.}-E_{IMSRG})/E_{Diag.}\\right|$")
#ax[1].set_title("Relative error vs energy spacing")



plt.savefig(r"Figs/RichardsonVsIMSRG.png",bbox_inches='tight',dpi=200)
plt.close()


for i in range(len(imsrgConv)):
	plt.scatter(s_vals[i],imsrgConv[i],label="$\\epsilon = $"+str(eps[i]),c='k')
	plt.axhline(y=diagEn[i],linestyle="--")
	plt.xlabel("$s$")
	plt.ylabel("$Energy$")
	plt.title("$E_{IMSRG}$ vs $s$\n Single particle energy = "+str(eps[i])+", $g=$"+str(g))
	plt.savefig(r"Figs/Convergence/IMSRGConvergenceEps"+str(eps[i])+".png",bbox_inches='tight',dpi=200)
	plt.close()





