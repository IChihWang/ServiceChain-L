import config as cfg
import itertools
import copy
from DeployAlg import DummyAlgorithm


class Server:
    def __init__(self, idx):
        self.availableCPU = cfg.CPU_MAX
        self.availableMEM = cfg.MEM_MAX
        self.availableBW = cfg.BW_MAX
        self.delayFactor = 0
        self.index = idx
    
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
        
    def getScore(self):
        score = self.availableCPU**2+self.availableMEM**2+self.availableBW**2
        return score

class DataCenter:
    def __init__(self, k):
        self.k = k
        self.servers=[Server(idx) for idx in range(k**3/4)]
        self.chains=[]
        
    def assignChainRL(self, chain):
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
    
    
    # Brute Force Assignment
    def assignChainBF(self, chain):
        # Exhaustively list all the possibilities
        a = [server_idx for server_idx in range(len(self.servers))]
        b = len(chain.serviceFunctions)
        possibilities = [x for x in itertools.product(a, repeat=b)]
        
        # Remove the one with latency violations
        for possibility in list(possibilities):
            latency = self.computeLatency(possibility)
            if latency > chain.latency_req:
                possibilities.remove(possibility)
        # Remove the one with resource violations & find best result (maximize N2) (minimize reduced score)
        min_possibility = None
        min_score_diff = float("inf")
        for possibility in list(possibilities):
            target_servers = [self.servers[server_idx] for server_idx in possibility]
            
            # Record the original score
            old_score = sum([target_server.getScore() for target_server in target_servers])
            
            # Evaluate the possibility
            is_feasible = True
            last_success_fun_idx = len(chain.serviceFunctions)
            for fun_idx in range(len(chain.serviceFunctions)):
                target_server = target_servers[fun_idx]
                target_function = chain.serviceFunctions[fun_idx]
                is_deployed = target_server.addFunction(target_function)
                if (not is_deployed):
                    is_feasible = False
                    last_success_fun_idx = fun_idx
                    break
            
            # Resource violation or not
            if (is_feasible):
                # No Resource violation
                new_score = sum([target_server.getScore() for target_server in target_servers])
                if old_score-new_score < min_score_diff:
                    min_score_diff = old_score-new_score
                    min_possibility = possibility
            else:
                # Resource violation
                possibilities.remove(possibility)
            
            # Reset the settings
            for rm_idx in range(last_success_fun_idx):
                rm_function = chain.serviceFunctions[rm_idx]
                rm_server = target_servers[rm_idx]
                rm_server.removeFunction(rm_function)
        
        
        if min_possibility != None:
            # If deploy chain sucessfully
            target_servers = [self.servers[server_idx] for server_idx in min_possibility]
            for fun_idx in range(len(chain.serviceFunctions)):
                target_server = target_servers[fun_idx]
                target_server.addFunction(target_function)
                target_function.server_idx = target_server.index
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
                
    def computeLatency(self, server_idx_list):
        latency = 0
        half_k = self.k/2
        
        for idx in range(1, len(server_idx_list)):
            server1_idx = server_idx_list[idx-1]
            server2_idx = server_idx_list[idx]
            
            # Same server
            if server1_idx == server2_idx:
                latency += 0
            # Same edge switch
            elif server1_idx//half_k == server2_idx//half_k:
                latency += 2
            # Same pod
            elif server1_idx//(half_k**2) == server2_idx//(half_k**2):
                latency += 4
            else:
                latency += 6
                
        return latency
            
    
    def removeChain(self, chain):
        for function in chain.serviceFunctions:
            server = self.servers[function.server_idx]
            server.removeFunction(function)
        chain.reset()
    
    
    def constructRLstate(self):
        # TODO: construct RL state (return with RL state)
        return None
        
    










