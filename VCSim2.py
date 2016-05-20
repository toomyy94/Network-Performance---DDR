import simpy
import numpy as np
from scipy.misc import factorial
import matplotlib.pyplot as plt

class NodeStats:

	def __init__(self,bw):
		self.vcs_total=0
		self.vcs_spec_total=0
		self.vcs_acc=0
		self.vcs_blk=0
		self.vcs_spec_blk=0
		self.bw=bw
		self.available_bw=bw
		self.lastloadchange=0
		self.loadint=0

def vc_generator(env,av_rate,av_duration,b,stats):

	while True:
		yield env.timeout(np.random.exponential(1.0/av_rate))
		stats.vcs_spec_total += 1
		if stats.available_bw>=b:
			stats.vcs_acc += 1
			stats.loadint += (1.0/stats.bw)*(stats.bw-stats.available_bw)*(env.now-stats.lastloadchange)
			stats.lastloadchange=env.now
			stats.available_bw -= b
		#	print("time %f: VC %d started"%(env.now,stats.vcs_total))
			env.process(vc(env,stats.vcs_total,av_duration,b,stats))
		else:
			stats.vcs_spec_blk += 1
		
		#	print("time %f: VC %d blocked"%(env.now,stats.vcs_total))
		
def vc_generator_standard(env,av_rate,av_duration,b,stats,reservated):

	while True:
		yield env.timeout(np.random.exponential(1.0/av_rate))
		stats.vcs_total += 1
		if stats.available_bw-reservated>=b:
			stats.vcs_acc += 1
			stats.loadint += (1.0/stats.bw)*(stats.bw-stats.available_bw)*(env.now-stats.lastloadchange)
			stats.lastloadchange=env.now
			stats.available_bw -= b
		#	print("time %f: VC %d started"%(env.now,stats.vcs_total))
			env.process(vc(env,stats.vcs_total,av_duration,b,stats))
		else:
			stats.vcs_blk += 1
		#	print("time %f: VC %d blocked"%(env.now,stats.vcs_total))

def vc(env,id,av_duration,b,stats):
	yield env.timeout(np.random.exponential(av_duration))
	stats.loadint += (1.0/stats.bw)*(stats.bw-stats.available_bw)*(env.now-stats.lastloadchange)
	stats.lastloadchange=env.now
	stats.available_bw += b
	#print("time %f: VC %d ended"%(env.now,id))

lamb=1
invmu=10
b1=2
b2=4
B=32
C=B/b1
simtime=3000
Reser = 0



x1=[1,1.5,2,2.5,3]
x2=[1,2,3,4,5]
y=[2,4,6,8,10]
z=[16,25,32]

res = np.zeros((5,5,3,4))


for j in range(0,len(x1)):
	for k in range(0,len(y)):
		for w in range(0,len(z)):
			stats=NodeStats(B)
			env = simpy.Environment()
			env.process(vc_generator(env,x1[j],y[k],b2,stats))
			env.process(vc_generator_standard(env,x2[j],y[k],b1,stats,Reser))
			env.run(simtime)
			C=1.0*z[w]/b1
			print("Simulated Block Probability=%f"%(1.0*stats.vcs_blk/stats.vcs_total))
			print("Simulated Average Link Load=%.2f%%"%(100.0*stats.loadint/simtime))
			rho=x1[j]*y[k]
			i=np.arange(0,C+1)
			blkp=(np.power(1.0*rho,C)/factorial(C))/np.sum(np.power(1.0*rho,i)/factorial(i))
			print("Theoretical Block Probability=%f"%(blkp))
			i1=np.arange(1,C+1)
			linkload=(1.0/C)*np.sum(np.power(1.0*rho,i1)/factorial(i1-1))/np.sum(np.power(1.0*rho,i)/factorial(i))
			print("Theoretical Average Link Load=%.2f%%"%(100*linkload))
			res[j,k,w,:] =np.array([1.0*stats.vcs_blk/stats.vcs_total,100.0*stats.loadint/simtime,blkp,100*linkload])

plt.title('BandWidth = '+str(B)+' Reservated= '+str(Reser))
for k in range(0,len(y)):
	#carga
	plt.plot(x2,res[:,k,0,1],label=str(y[k]))
	#plt.plot(x,res[:,k,0,2],ls='dotted')
	
	#descomentar para prob. de bloqueio
	#plt.plot(x2,res[:,k,0,0],label=str(y[k]))
	#plt.plot(x,res[:,k,0,3],ls='dotted')
plt.legend() 
plt.show()

