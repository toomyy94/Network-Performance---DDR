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

env = simpy.Environment()

B = 10e6
K = 256
lamba = 300
lambb = 300
tmp = 782
mu = 350
mu = 1.0*B/(tmp*8) 
R1= 500
R2= 500
R3= 500
R4= 500


#Sender (tx) -> Node1 -> Link -> Receiver (rx)

rxa = pkt_Receiver(env,'A')
txa = pkt_Sender(env,'A',lamba,'I')
rxb = pkt_Receiver(env,'B')
txb = pkt_Sender(env,'B',lambb,'I') 
rxi = pkt_Receiver(env,'I')
#txi = pkt_Sender(env,'I',lamb,'I') 
 
node1 = Node(env,'N1',R1,K)
node2 = Node(env,'N2',R2,K)
node3 = Node(env,'N3',R3,K)
node4 = Node(env,'N4',R4,K) 

link12 = Link(env,'L12',B,K)
link23 = Link(env,'L23',B,K) 
link34 = Link(env,'L34',B,K)
link4I = Link(env,'L4I',B,K)

#link43 = Link(env,'L43',B,K)
#link31 = Link(env,'L31',B,K) 
#link32 = Link(env,'L32',B,K)



txa.out = node1
txb.out = node2


node1.add_conn(link12,'I')
link12.out=node2
#link21.out=node1


node2.add_conn(link23,'I')
link23.out=node3
#link21.out=rx


node3.add_conn(link34,'I')
link34.out=node4

node4.add_conn(link4I,'I')
link4I.out=rxi

#print(node1.out)
#print(node2.out) 

simtime=50
env.run(simtime)

print('total loss:%.2f%%'%(100.0*(link12.lost_pkts+link23.lost_pkts+link34.lost_pkts+link4I.lost_pkts+node1.lost_pkts+node2.lost_pkts+node3.lost_pkts+node4.lost_pkts)/(txa.packets_sent+txb.packets_sent)) ) 
#print('Loss probability 1: %.2f%%'%(100.0*link1.lost_pkts/tx.packets_sent))
#print('Loss probability 2: %.2f%%'%(100.0*link2.lost_pkts/tx.packets_sent))
#print('Average delay: %f sec'%(1.0*rx.overalldelay/rx.packets_recv))
#print('Transmitted bandwidth: %.1f Bytes/sec'%(1.0*rx.overallbytes/simtime))

#Wk = 2.0/(mu-lamb) 









