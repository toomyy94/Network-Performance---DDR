import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mygeo import getgeo, geodist 
import itertools
import random
import pickle
import argparse

mu=1e9/8000 #link speed in pkts/sec
lightspeed=300000.0 #Km/sec

def listStats(L):
	#Returns the mean and maximum values of a list of numbers with generic keys
	#	returns also the key of the maximum value
	V=L.values()
	K=L.keys()
	meanL=np.mean(V)
	maxL=np.max(V)
	p=np.where(V==maxL)[0][0]
	maxLK=K[p]
	return meanL, maxL, maxLK
	
def compute_average(allpairs,net,WsAll,sol):
	#print('sol '+ str(sol))
	for pair in allpairs:
		Ws = 0 
		path=sol[pair]
		for i in range(0,len(path)-1):
			Ws+=1e6/(mu-net[path[i]][path[i+1]]['load'])#+net[path[i]][path[i+1]]['distance']/lightspeed
			WsAll.update({pair:Ws})
			#print('#flow %s-%s: %.2f micro sec'%(pair[0],pair[1],Ws))

	meanWs, maxWs, maxWsK = listStats(WsAll)
	#print('Mean one-way delay: %.2f ms\nMaximum one-way delay: %.2f micro sec for flow %s-%s'%(meanWs,maxWs,maxWsK[0],maxWsK[1]))
	return meanWs

WsAll={}

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', nargs='?', help='input network file', default='networks.dat')
args=parser.parse_args()

filename=args.file 

with open(filename) as f:
	nodes, links, pos, tm = pickle.load(f)

print(tm)

net=nx.DiGraph()
for node in nodes:
	net.add_node(node)
	
for link in links:
	dist=geodist((pos[link[0]][1],pos[link[0]][0]),(pos[link[1]][1],pos[link[1]][0]))
	net.add_edge(link[0],link[1],distance=dist, load=0, delay=0)
	net.add_edge(link[1],link[0],distance=dist, load=0, delay=0)
	#print(link,dist,(pos[link[0]][1],pos[link[0]][0]),(pos[link[1]][1],pos[link[1]][0]))
net0 = net.copy()

nx.draw(net,pos,with_labels=True)
plt.show()

allpairs=list(itertools.permutations(nodes,2))
sol={}

mu = 1e9/8000

random.shuffle(allpairs)

best = np.inf 
best_path = {}

for i in range(0,1000): 
	sol={}
	net = net0.copy() 
	for pair in allpairs:
		path=nx.shortest_path(net,pair[0],pair[1],weight='delay')
		sol.update({pair:path})
		for i in range(0,len(path)-1):
			net[path[i]][path[i+1]]['load']+=tm[pair[0]][pair[1]]
			net[path[i]][path[i+1]]['delay']=1.0/(mu-net[path[i]][path[i+1]]['load'])
		#print('---')
		#print('Solution:'+str(i)+' :'+str(sol))
	current = compute_average(allpairs,net,WsAll,sol)
	if current<best:
		best = current
		best_path = sol 	
			 
print(' best solution '+str(best))
print(' best path '+str(best_path))
	
print('---')
loadAll={}
	
for link in links:
	print("#link %s-%s: %d pkts/sec"%(link[0],link[1],net[link[0]][link[1]]['load']))
#	loadAll.update({(link[0],link[1]):net[link[0]][link[1]]['load']})
	print("#link %s-%s: %d pkts/sec"%(link[1],link[0],net[link[1]][link[0]]['load']))
#	loadAll.update({(link[0],link[1]):net[link[1]][link[0]]['load']})
	
print(loadAll)

	
	
	
	
	
	
