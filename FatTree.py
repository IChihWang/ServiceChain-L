import config as cfg
from DeployAlg import DummyAlgorithm


class Server:
    def __init__(self):
        self.availableCPU = cfg.CPU_MAX
        self.availableMEM = cfg.MEM_MAX
        self.availableBW = cfg.BW_MAX
        self.delayFactor = 0
    
    def addFunction(self, function):
        is_success = None
    
        # The server CANNOT deploy the function
        if self.availableCPU < function.cpu:
            is_success = False
        if self.availableMEM < function.mem:
            is_success = False
        if self.availableBW < function.bw:
            is_success = False
        
        if is_success == False:
            None
        else:
            # The server CAN deploy the function
            self.availableCPU -= function.cpu
            self.availableMEM -= function.mem
            self.availableBW -= function.bw
            is_success = True        
            return is_success
        return is_success
        
    def removeFunction(self, function):
        self.availableCPU += function.cpu
        self.availableMEM += function.mem
        self.availableBW += function.bw

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
        for function_idx in range(len(chain.serviceFunctions)):
            target_function = chain.serviceFunctions[function_idx]
        
            # Find the server for the function through the algorithm
            ### TODO: Construct RLstate
            RLstate = self.constructRLstate()
            server_idx = DummyAlgorithm(target_function, RLstate)
            target_server = self.servers[server_idx]
            
            
            # Try to deploy the function
            if (chain.current_latency_req >= target_server.delayFactor):
                is_deployed = target_server.addFunction(target_function)
                
            else:
                # Reject directly if latency requirement doesn't meet
                is_deployed = False
                
            
            
            if is_deployed:
                # Record the server idx on function if deployment succeed
                target_function.server_idx = server_idx
                chain.current_latency_req -= target_server.delayFactor
                # Update delay factor
                self.updateDelayFactor(server_idx)
                
            # Remove whole chain if the deployment fails
            else:
                for rm_idx in range(function_idx):
                    rm_function = chain.serviceFunctions[rm_idx]
                    rm_server = self.servers[rm_function.server_idx]
                    rm_server.removeFunction(rm_function)
                chain.reset()
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
            server = self.servers[idx]
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
    
    def removeChain(self, chain):
        for function in chain.serviceFunctions:
            server = self.servers[function.server_idx]
            server.removeFunction(function)
        chain.reset()
    
    
    def constructRLstate(self):
        # TODO: construct RL state (return with RL state)
        return None
        
    










