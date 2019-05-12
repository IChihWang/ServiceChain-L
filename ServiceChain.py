import config as cfg
from random import randrange, expovariate, uniform


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
        assert len(serviceFunctions) > 0
        
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
    #num_chain = 21
    print("num_chain", num_chain)
    num_function_chain = randrange(1,10)
    num_function_chain = 5
    #print("Waring !! Fix num_function_chain Now !!")

    #∂çprint("num_function_chain", num_function_chain)
    for idx in range(num_chain):
        functs = []
        for jdx in range(num_function_chain):
            functs.append(ServiceFunction(40, 35, 35))
        chain = ServiceChain(20, 5, randrange(cfg.SIMU_TIME-10), functs)
        
        testSet.append(chain)
        
    return testSet

def PoissonGenChains(k, arrival_rate, service_rate):
    # List of chains
    testSet = []

    time_idx = 0
    while time_idx < cfg.SIMU_TIME:

        num_function_chain = randrange(1,10)
        num_function_chain = 5

        functs = []
        for jdx in range(num_function_chain):
            CPU_req = int(uniform(1,50))
            MEM_req = int(uniform(1,50))
            BW_req = int(uniform(1,50))
            functs.append(ServiceFunction(CPU_req, MEM_req, BW_req))

        service_time = expovariate(service_rate)*(1/cfg.TIME_STEP_PER_SECOND) + 1

        min_delay = 0
        server_num = 1
        agg_function = ServiceFunction(0,0,0)
        for idx in range(num_function_chain):
            agg_function.O2AI_aggregate_function(functs[idx])
            if agg_function.cpu > 100 or agg_function.mem > 100 or agg_function.bw > 100:
                agg_function = ServiceFunction(0,0,0)
                server_num += 1
        min_delay += 6 * server_num // ((k/2)*(k/2))
        min_delay += 4 * ((server_num)%((k/2)*(k/2))) // (k/2)
        min_delay += 2 * ((server_num)%(k/2))


        max_delay = 6 * (num_function_chain-1)

        delay = uniform(min_delay, max_delay)

        chain = ServiceChain(delay, int(service_time), int(time_idx), functs)

        testSet.append(chain)

        next_time = expovariate(arrival_rate)*(1/cfg.TIME_STEP_PER_SECOND)
        time_idx += next_time


    return testSet
    
    
    
### TODO: generate chain in different ways