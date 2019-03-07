class ServiceChain:
	def __init__(self, latency_req, serviceTime):
		self.latency_req = None
		self.serviceFunctions = []
		self.serviceTime = None
		self.waitingTime = None
