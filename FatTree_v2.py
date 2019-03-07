from queue import Queue
import config as cfg
from DeployAlg import DummyAlgorithm


class Server:
    def __init__(self):
        self.availableCPU = cfg.CPU_MAX
        self.availableMEM = cfg.MEM_MAX
        self.availableBW = cfg.BW_MAX
        self.delayFactor = 0
    
    def addFunction(function):
        is_success = None
    
        # The server CANNOT deploy the function
        if self.availableCPU < function.cpu:
            is_success = False
        if self.availableCPU < function.cpu:
            is_success = False
        if self.availableCPU < function.cpu:
            is_success = False
        
        # The server CAN deploy the function
        self.availableCPU -= function.cpu
        self.availableMEM -= function.mem
        self.availableBW -= function.bw
        is_success = True        
        
        return is_success
        
    def removeFunction(function):
        self.availableCPU -= function.cpu
        self.availableMEM -= function.mem
        self.availableBW -= function.bw

class DataCenter:
    def __init__(self, k):
        self.k = k
        self.servers=[Server() for idx in range(k**3/4)]
        self.chains=[]
        
    def assignChain(self, chain):
        is_success = True
        
        # Initialize
        self.initDelayFactor()
    
        # Try to deploy the chain
        for function_idx in range(len(chain)):
            target_function = chain[function_idx]
        
            # Find the server for the function through the algorithm
            ### TODO: Construct RLstate
            RLstate = self.constructRLstate()
            server_idx = DummyAlgorithm(target_function, RLstate)
            target_server = self.servers[server_idx]
            
            # Try to deploy the function
            is_deployed = target_server.addFunction(target_function)
            
            
            if is_deployed:
                # Record the server idx on function if deployment succeed
                target_function.server_idx = server_idx
                # Update delay factor
                self.updateDelayFactor(server_idx)
                
            # Remove whole chain if the deployment fails
            else:
                for rm_idx in range(function_idx):
                    rm_function = chain[rm_idx]
                    rm_server = rm_function.server_idx
                    rm_server.removeFunction(rm_function)
                is_success = False
                break
            
        
        if is_success:
            # If deploy chain sucessfully
            self.chains.append(chain)
            return True
        
        else:
            # If not deploy chain sucessfully
            return False
    
    def initDelayFactor(self):
        for server in self.servers:
            server.delayFactor = 0
            
    def updateDelayFactor(self, server_idx):
        half_k = self.k/2
    
        for idx in range(len(self.servers)):
            # Same server
            if idx == server_idx:
                server.delayFactor = 0
            # Same edge switch
            elif idx//half_k == server_idx//half_k:
                server.delayFactor = 2
            # Same pod
            elif idx//(half_k**2) == server_idx//(half_k**2):
                server.delayFactor = 4
            else:
                server.delayFactor = 6
    
    def constructRLstate(self):
        # TODO: construct RL state (return with RL state)
        return None
        
    
        
        

class Node:
	def __init__(self,score):
		
        # score is a function of CPU, Memory and Bandwidth
		#self.score=score
        ### Mick: in this project, score will be CPU, MEM, BW independently
        
        
		# NodeList have all the references to its immediate children
		self.NodeList=None
		self.parent=None
		# level can be one of the 0-core, 1-aggr, 2-edge or 3-server levels
		self.level = None
		self.serviceFunction=None

class FatTree:
	# K- no of ports for a switch
	# Assume k as even
	def __init__(self,k):
		self.root  = None
		self.k = k
	def createTree(self):
		for level in range(0,4):
			if(level == 0):
				score = (self.k**3 / 4) * 100
				self.root = Node(score)
				self.root.level = level
			if(level == 1):
				score = (self.k**2)/4 * 100
				self.root.NodeList=[]
				for i in range(0,self.k):
					aggrNode = Node(score)
					aggrNode.level = level
					aggrNode.parent = self.root
					self.root.NodeList.append(aggrNode)
			if(level == 2):
				score = (self.k)/2 * 100
				for aggrNode in self.root.NodeList:
					aggrNode.NodeList=[]
					for i in range(0,int(self.k/2)):
						edgeNode = Node(score)
						edgeNode.level = level
						edgeNode.parent = aggrNode
						aggrNode.NodeList.append(edgeNode)
			if(level == 3):
				score = 100
				for aggrNode in self.root.NodeList:
					for edgeNode in aggrNode.NodeList:
						edgeNode.NodeList = []
						for i in range(0,int(self.k/2)):
							serverNode = Node(score)
							serverNode.level = level
							serverNode.parent = edgeNode
							edgeNode.NodeList.append(serverNode)
	def print_level_order(self):
		q=Queue()
		temp=self.root
		q.put(temp)
		while(not q.empty()):
			temp=q.get()
			print("Level:", temp.level)
			print("score:", temp.score)
			if(temp.NodeList is not None):
				for node in temp.NodeList:
					q.put(node)

if __name__ == "__main__":
	f_tree = FatTree(6)
	f_tree.createTree()
	f_tree.print_level_order()









