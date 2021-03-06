import simpy
import random 
import numpy as np

class Packet(object):
	"""
	Packet Object
	time: packet creation time/ID (float)
	size: packet size (integer)
	dst: packet destination (string) - pkt_Receiver ID
	"""
	def __init__(self,time,size,dst):
		self.time=time
		self.size=size
		self.dst=dst
	
	def __repr__(self):
		return 'Pkt %f [%d] to %s'%(self.time,self.size,self.dst)

class Node(object):
	"""
	Node Object
	env: SimPy environment
	id: Node ID (string)
	speed: Node routing speed (float, pkts/sec)
	qsize: Node input queue size (integer, number of packets, default inf)
	"""
	def __init__(self,env,id,speed,qsize=np.inf):
		self.env=env
		self.id=id
		self.speed=speed
		self.qsize=qsize
		self.queue = simpy.Store(env)
		self.lost_pkts=0
		self.out={} #list with obj {'dest1':[elem1,elem3],'dest2':[elem1,elem2],...}
		self.action = env.process(self.run())
		
	def add_conn(self,elem,dsts):
		"""
		Defines node output connections to other simulation elements
		elem: Next element (object)
		dsts: list with destination(s) ID(s) accessible via elem (string or list of strings)
		"""
		for d in dsts:
			if self.out.has_key(d):
				self.out[d].append(elem)
			else:
				self.out.update({d:[elem]})
	
	def run(self):
		while True:
			pkt = (yield self.queue.get())
			yield self.env.timeout(1.0/self.speed)
			if self.out.has_key(pkt.dst):
				#random routing over all possible paths to dst
				outobj=self.out[pkt.dst][random.randint(0,len(self.out[pkt.dst])-1)]
				#print(str(self.env.now)+': Packet out node '+self.id+' - '+str(pkt))
				outobj.put(pkt)
			#else:
				#print(str(self.env.now)+': Packet lost in node '+self.id+'- No routing path - '+str(pkt))
	
	def put(self,pkt):
		if len(self.queue.items)<self.qsize:
			self.queue.put(pkt)
		else:
			self.lost_pkts += 1
			#print(str(env.now)+': Packet lost in node '+self.id+' queue - '+str(pkt))

class Link(object):
	"""
	Link Object
	env: SimPy environment
	id: Link ID (string)
	speed: Link transmission speed (float, bits/sec)
	qsize: Node to Link output queue size (integer, number of packets, default inf)
	"""
	def __init__(self,env,id,speed,qsize=np.inf):
		self.env=env
		self.id=id
		self.speed=1.0*speed/8
		self.qsize=qsize
		self.queue = simpy.Store(env)
		self.lost_pkts=0
		self.out=None
		self.action = env.process(self.run())
		
	def run(self):
		while True:
			pkt = (yield self.queue.get())
			yield self.env.timeout(1.0*pkt.size/self.speed)
			#print(str(self.env.now)+': Packet out link '+self.id+' - '+str(pkt))
			self.out.put(pkt)
				
	def put(self,pkt):
		if len(self.queue.items)<self.qsize:
			self.queue.put(pkt)
		else:
			self.lost_pkts += 1
			#print(str(self.env.now)+': Packet lost in link '+self.id+' queue - '+str(pkt))
		
		
class pkt_Sender(object):
	"""
	Packet Sender
	env: SimPy environment
	id: Sender ID (string)
	rate: Packet generation rate (float, packets/sec)
	dst: List with packet destinations (list of strings, if size>1 destination is random among all possible destinations)
	"""
	def __init__(self,env,id,rate,dst):
		self.env=env
		self.id=id
		self.rate=rate
		self.out=None	
		self.dst=dst
		self.packets_sent=0	
		self.action = env.process(self.run())
		
	def run(self):
		while True:
			yield self.env.timeout(np.random.exponential(1.0/self.rate))
			self.packets_sent += 1
			#size=random.randint(64,1500)
			#size=int(np.random.exponential(500))
			size=int(np.random.choice([64,1500],1,[.5,.5]))
			if len(self.dst)==1:
				dst=self.dst[0]
			else:
				dst=self.dst[random.randint(0,len(self.dst)-1)]
			pkt = Packet(self.env.now,size,dst)
			#print(str(self.env.now)+': Packet sent by '+self.id+' - '+str(pkt))
			self.out.put(pkt)
		
class pkt_Receiver(object):
	"""
	Packet Receiver
	env: SimPy environment
	id: Sender ID (string)
	"""
	def __init__(self,env,id):
		self.env=env
		self.id=id
		self.queue = simpy.Store(env)
		self.packets_recv=0
		self.overalldelay=0
		self.overallbytes=0
		self.action = env.process(self.run())
		
	def run(self):
		while True:
			pkt = (yield self.queue.get())
			self.packets_recv += 1
			self.overalldelay += self.env.now-pkt.time
			self.overallbytes += pkt.size
			#print(str(self.env.now)+': Packet received by '+self.id+' - '+str(pkt))
	
	def put(self,pkt):
		self.queue.put(pkt)

for i in range(0,4):
	env = simpy.Environment()

	#Sender (tx) -> Node1 -> Link -> Receiver (rx)

<<<<<<< HEAD:Guia3/PktSim5.py
lamb=300
K=96
B=2e6
tmp=782
R=300
mu=B/(tmp*8)
#mu=350
=======
	lamb=150
	K=1000
	B=2e6
	tmp=782
	R=350
	mu=B/(tmp*8)

>>>>>>> origin/master:PktSim5.py


	rx=pkt_Receiver(env,'B')
	tx=pkt_Sender(env,'A',lamb,'B')
	node1=Node(env,'N1',R,K)
	link=Link(env,'L',B,K)
	link2 = Link(env,'L2',B,K)

	tx.out=link
	node1.add_conn(link,'B')
	node1.add_conn(link2,'B')

	link.out=node1
	link2.out = rx


	#print(node1.out)

	simtime=100
	env.run(simtime)

	print('%.2f%%'%(100.0*(link.lost_pkts+node1.lost_pkts+link2.lost_pkts)/tx.packets_sent))
	print('%f '%(1.0*rx.overalldelay/rx.packets_recv))
	#print('Transmitted bandwidth: %.1f Bytes/sec'%(1.0*rx.overallbytes/simtime))

<<<<<<< HEAD:Guia3/PktSim5.py
#... prints de delays das varias formulas e perda prob 
=======
	#... prints de delays das varias formulas e perda prob
>>>>>>> origin/master:PktSim5.py



	ro=1.0*lamb/mu
	PB=(ro**K)/(np.sum(ro**np.arange(0,K+1)))
	lambK=(1-PB)*lamb

	a=64*8
	b=1500*8

	#M/M/1
	Wmm1=1.0/(mu-lamb)
	#M/D/1
	Wmd1=(1.0/mu)+(lamb/(2.0*mu*(mu-lamb)))
	#M/M//1/K
	Wmm1k=(1.0/lambK)*((ro/(1.0-ro))-(K+1.0)*ro**(K+1.0)/(1.0-ro**K+1))
	#M/G/1
	ES=((a+b)/(2.0*B))
	ES2=(a**2.0+b**2.0)/(2.0*B**2.0)
	Wmg1=((lamb*ES2)/(2.0*(1.0-lamb*ES)))+ES

	print("Delay M/M/1: ")
	print(round(Wmm1, 4))
	print("Delay M/D/1: ")
	print(round(Wmd1, 4))
	print("Delay M/G/1: ")
	print(round(Wmg1, 4))
	print("Delay M/M/1/K: ")
	print(round(Wmm1k, 3))
	print("Probabilidade de Bloqueio: ")
	print(round(PB, 3)*100)



