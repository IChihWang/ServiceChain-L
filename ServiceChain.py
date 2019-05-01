import config as cfg
from random import randrange


class ServiceFunction:
    def __init__(self, cpu, mem, bw):
        self.cpu = cpu
        self.mem = mem
        self.bw = bw
        
        # Record which server this function is current on
        self.server_idx = None
        
    
    # O2AI copy
    def O2AI_copy(self):
        new_service_function = ServiceFunction(self.cpu, self.mem, self.bw)
        new_service_function.server_idx = self.server_idx
        
        return new_service_function
        
        
        
    def O2AI_aggregate_function(self, service_function):
        self.cpu += service_function.cpu
        self.mem += service_function.mem
        self.bw += service_function.bw
    def O2AI_rm_aggregate_function(self, service_function):
        self.cpu -= service_function.cpu
        self.mem -= service_function.mem
        self.bw -= service_function.bw
    
        
class ServiceChain:
    def __init__(self, latency_req, serviceTime, arrive_time, serviceFunctions):
        assert serviceTime > 0
        
        self.latency_req = latency_req
        self.serviceFunctions = serviceFunctions
        self.functionNum = len(serviceFunctions)
        self.serviceTime = serviceTime
        self.waitingTime = 0
        
        # Requirement after some functions are deployed
        self.current_latency_req = self.latency_req
        
        # Exact time when the chain is served
        self.arrive_time = arrive_time
        # Exact time when the chain is served
        self.finish_time = None
        
        
    
    def O2AI_copy(self, serviceFunctions):
        # O2AI: for sub service chain
        new_chain = ServiceChain(self.latency_req, self.serviceTime, self.arrive_time, serviceFunctions)
        new_chain.functionNum = len(serviceFunctions)
        
        return new_chain
        
    '''  
    # Copy instruction
    def __init__(self, chain, serviceFunctions):
        # O2AI: for sub service chain
        self.latency_req = chain.latency_req
        self.serviceFunctions = serviceFunctions
        self.serviceTime = chain.serviceTime
        self.functionNum = len(serviceFunctions)
        
    def __init__(self, chain):
        # O2AI: for sub service chain
        self.latency_req = chain.latency_req
        self.serviceFunctions = []
        self.serviceTime = chain.serviceTime
        self.functionNum = 0
    '''    
        
    def reset(self):
        self.current_latency_req = self.latency_req
        for function in self.serviceFunctions:
            function.server_idx = None

def DummyGenChains():
    # List of chains
    testSet = []
    
    num_chain = randrange(100)
    num_function_chain = randrange(1,10)
    
    for idx in range(num_chain):
        functs = []
        for jdx in range(num_function_chain):
            functs.append(ServiceFunction(1, 2, 3))
        chain = ServiceChain(100, 5, randrange(cfg.SIMU_TIME-10), functs)
        
        testSet.append(chain)
        
    return testSet
    
    
    
### TODO: generate chain in different ways