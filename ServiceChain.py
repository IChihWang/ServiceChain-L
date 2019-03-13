import config as cfg
from random import randrange


class ServiceFunction:
    def __init__(self, cpu, mem, bw):
        self.cpu = cpu
        self.mem = mem
        self.bw = bw
        
        # Record which server this function is current on
        self.server_idx = None

class ServiceChain:
    def __init__(self, latency_req, serviceTime, arrive_time, serviceFunctions):
        assert serviceTime > 0
        assert len(serviceFunctions) > 0
        
        self.latency_req = latency_req
        self.serviceFunctions = serviceFunctions
        self.serviceTime = serviceTime
        self.waitingTime = 0
        
        # Requirement after some functions are deployed
        self.current_latency_req = self.latency_req
        
        # Exact time when the chain is served
        self.arrive_time = arrive_time
        # Exact time when the chain is served
        self.finish_time = None
        
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