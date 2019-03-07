from queue import Queue

class Node:
	def __init__(self,score):
		# score is a function of CPU, Memory and Bandwidth
		self.score=score
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









