import config as cfg
import itertools
import copy
from DeployAlg import DummyAlgorithm
from ServiceChain import ServiceFunction, ServiceChain
import math


class Server:
    def __init__(self, idx):
        self.availableCPU = cfg.CPU_MAX
        self.availableMEM = cfg.MEM_MAX
        self.availableBW = cfg.BW_MAX
        self.delayFactor = 0
        self.index = idx
        
        # For O2AI
        self.O2AI_child_servers = []
        self.O2AI_delay_cost = 0
        
    
    def addFunction(self, function):
        is_success = self.canFitFunction(function)
    
        if is_success == True:
            # The server CAN deploy the function
            self.availableCPU -= function.cpu
            self.availableMEM -= function.mem
            self.availableBW -= function.bw
            
        return is_success
        
    def removeFunction(self, function):
        self.availableCPU += function.cpu
        self.availableMEM += function.mem
        self.availableBW += function.bw
        
        # DEBUG
        assert self.availableCPU <= cfg.CPU_MAX
        assert self.availableMEM <= cfg.MEM_MAX
        assert self.availableBW <= cfg.BW_MAX
        
    def canFitFunction(self, function):
        is_success = True
    
        # The server CANNOT deploy the function
        if self.availableCPU < function.cpu:
            is_success = False
        if self.availableMEM < function.mem:
            is_success = False
        if self.availableBW < function.bw:
            is_success = False
        return is_success
        
    def getScore(self):
        score = self.availableCPU**2+self.availableMEM**2+self.availableBW**2
        return score
    
    def O2AI_fast_copy(self):
        new_server = Server(self.index)
        new_server.availableCPU = self.availableCPU
        new_server.availableMEM = self.availableMEM
        new_server.availableBW = self.availableBW
        return new_server
        
    def O2AI_copy(self):
        new_server = Server(self.index)
        new_server.availableCPU = self.availableCPU
        new_server.availableMEM = self.availableMEM
        new_server.availableBW = self.availableBW
        
        new_server.O2AI_child_servers = [server.O2AI_copy() for server in self.O2AI_child_servers]
        
        return new_server
    
    def O2AI_get_max_CPU(self):
        if len(self.O2AI_child_servers) == 0:
            return self
        else:
            return max(self.O2AI_child_servers, key=lambda x:x.O2AI_get_max_CPU().availableCPU)
    def O2AI_get_max_MEM(self):
        if len(self.O2AI_child_servers) == 0:
            return self
        else:
            return max(self.O2AI_child_servers, key=lambda x:x.O2AI_get_max_MEM().availableMEM)
    def O2AI_get_max_BW(self):
        if len(self.O2AI_child_servers) == 0:
            return self
        else:
            return max(self.O2AI_child_servers, key=lambda x:x.O2AI_get_max_BW().availableBW)
            
    def O2AI_get_height_cost(self):
        if len(self.O2AI_child_servers) == 0:
            return 0
        else:
            return self.O2AI_child_servers[0].O2AI_get_height_cost()+2
    
    
    
    def O2AI_deploy(self, chain, datacenter, whole_chain):
        
        # Handle leaves
        if len(self.O2AI_child_servers) == 0:
            deployed_num = 0
            func_list = chain.serviceFunctions
            for function in func_list:
                is_sucess = self.addFunction(function)
                if is_sucess:
                    function.server_idx = self.index
                    deployed_num += 1
                else:
                    break
            func_list = chain.serviceFunctions
            new_func_list = func_list[deployed_num:len(func_list)]
            new_chain = chain.O2AI_copy(new_func_list)
            
            return {"remaining_chain":new_chain, "deployed_num":deployed_num}
        
        # Handle nodes
        deployed_num = 0
        sub_chain = chain
        height_cost = self.O2AI_get_height_cost()
        
        
        while True:
            func_list = sub_chain.serviceFunctions
            acc_func_list = [func_list[0].O2AI_copy()]
            # Accumulate the functions
            for funct_idx in range(1, len(func_list)):
                acc_function = func_list[funct_idx].O2AI_copy()
                acc_function.O2AI_aggregate_function(acc_func_list[funct_idx-1])
                acc_func_list.append(acc_function)
             
            
            # Calculate the scores for each server
            bf_score = {"CPU":{"score":float("inf"), "lump_idx":-1}, "MEM":{"score":float("inf"), "lump_idx":-1}, "BW":{"score":float("inf"), "lump_idx":-1}} # CPU, MEM, BW, idx of lumpy function
            for server in self.O2AI_child_servers:
                # Best fit score
                fail_idx = None
                for function_idx in range(len(acc_func_list)):
                    if not server.canFitFunction(acc_func_list[function_idx]):
                        fail_idx = function_idx
                        break
                        
                if fail_idx == 0:
                    # This server is full, no function can be deployed
                    continue
                elif fail_idx != None:
                    # Some function can be deployed, some not, might have lumpy
                    # 1. Check if there is a lumpy function
                    lumpy_function = func_list[fail_idx]
                    sublist = func_list[0:fail_idx]
                    if lumpy_function.cpu > max(sublist, key=lambda x: x.cpu):
                    
                        # 2. Check if the tight CPU can handle
                        target_server = server.O2AI_get_max_CPU()
                        if not target_server.canFitFunction(lumpy_function):
                            continue
                        
                        # 3. Check if delay can be satisfied
                        predicted_delay = 0
                        pre_lump_delay = 0
                        aggr_cpu = acc_func_list[fail_idx].cpu
                        for rm_idx in range(fail_idx):
                            aggr_cpu -= func_list[rm_idx].cpu
                            pre_lump_delay += height_cost
                            if aggr_cpu < server.availableCPU:
                                break
                        predicted_delay += pre_lump_delay
                        predicted_delay += height_cost
                        
                        # Post lump delay
                        copy_server_list = []
                        for copy_server in self.O2AI_child_servers:
                            if server == copy_server:
                                continue
                            else:
                                copy_server_list.append(copy_server.O2AI_fast_copy())
                        
                        result = self.O2AI_get_FFit_delay(copy_server_list, func_list, height_cost)
                        post_lump_delay = result["delay"]
                        predicted_delay += post_lump_delay
                        if predicted_delay <= sub_chain.latency_req:
                            score = target_server.availableCPU - aggr_cpu
                            if score < bf_score["CPU"]["score"]:
                                bf_score["CPU"]["score"] = score
                                bf_score["CPU"]["lump_idx"] = fail_idx
                            
                    if lumpy_function.mem > max(sublist, key=lambda x: x.mem):
                        # 2. Check if the tight mem can handle
                        target_server = server.O2AI_get_max_MEM()
                        if not target_server.canFitFunction(lumpy_function):
                            continue
                        
                        # 3. Check if delay can be satisfied
                        predicted_delay = 0
                        pre_lump_delay = 0
                        aggr_mem = acc_func_list[fail_idx].mem
                        for rm_idx in range(fail_idx):
                            aggr_mem -= func_list[rm_idx].mem
                            pre_lump_delay += height_cost
                            if aggr_mem < server.availableMEM:
                                break
                        predicted_delay += pre_lump_delay
                        predicted_delay += height_cost
                        
                        # Post lump delay
                        copy_server_list = []
                        for copy_server in self.O2AI_child_servers:
                            if server == copy_server:
                                continue
                            else:
                                copy_server_list.append(copy_server.O2AI_fast_copy())
                        
                        result = self.O2AI_get_FFit_delay(copy_server_list, func_list, height_cost)
                        post_lump_delay = result["delay"]
                        predicted_delay += post_lump_delay
                        if predicted_delay <= sub_chain.latency_req:
                            score = target_server.availableMEM - aggr_mem
                            if score < bf_score["MEM"]["score"]:
                                bf_score["MEM"]["score"] = score
                                bf_score["MEM"]["lump_idx"] = fail_idx
                                
                    if lumpy_function.bw > max(sublist, key=lambda x: x.bw):
                        # 2. Check if the tight bw can handle
                        target_server = server.O2AI_get_max_BW()
                        if not target_server.canFitFunction(lumpy_function):
                            continue
                        
                        # 3. Check if delay can be satisfied
                        predicted_delay = 0
                        pre_lump_delay = 0
                        aggr_bw = acc_func_list[fail_idx].bw
                        for rm_idx in range(fail_idx):
                            aggr_bw -= func_list[rm_idx].bw
                            pre_lump_delay += height_cost
                            if aggr_bw < server.availableBW:
                                break
                        predicted_delay += pre_lump_delay
                        predicted_delay += height_cost
                        
                        # Post lump delay
                        copy_server_list = []
                        for copy_server in self.O2AI_child_servers:
                            if server == copy_server:
                                continue
                            else:
                                copy_server_list.append(copy_server.O2AI_fast_copy())
                        
                        result = self.O2AI_get_FFit_delay(copy_server_list, func_list, height_cost)
                        post_lump_delay = result["delay"]
                        predicted_delay += post_lump_delay
                        if predicted_delay <= sub_chain.latency_req:
                            score = target_server.availableBW - aggr_bw
                            if score < bf_score["BW"]["score"]:
                                bf_score["BW"]["score"] = score
                                bf_score["BW"]["lump_idx"] = fail_idx
            
            # Decide BFit or FFit
            bf_type = min(bf_score.keys(), key=(lambda k: bf_score[k]["score"]))
            min_score = bf_score[bf_type]["score"]
            lump_idx = bf_score[bf_type]["lump_idx"]
            
            
            is_re_predict = False
            if min_score != float('inf'):
                if bf_type == "CPU":
                    # Deploy with cpu
                    while True:
                        func_list = sub_chain.serviceFunctions
                        target_func = func_list[0]
                        server_list = []
                        for server in self.O2AI_child_servers:
                            if server.canFitFunction(target_func):
                                score = server.availableCPU-target_func.cpu
                                server_list.append({"server":server, "score":score})
                        server_list.sort(key=lambda x: x["score"])
                        
                        is_deployed = False
                        for server in server_list:
                            result = server["server"].O2AI_deploy(sub_chain, datacenter, whole_chain)
                            if result["deployed_num"] > 0:
                                deployed_num += result["deployed_num"]
                                sub_chain = result["remaining_chain"]
                                is_deployed = True
                                break

                        if not is_deployed:
                            # Cannot deploy anymore
                            break
                        if len(sub_chain.serviceFunctions) == 0:
                            # Done deploying
                            break
                        elif len(chain.serviceFunctions)-len(sub_chain.serviceFunctions) > lump_idx+1:
                            # Done BFit
                            is_re_predict = True
                            break
                        
                            
                    
                elif bf_type == "MEM":
                    # Deploy with mem
                    while True:
                        func_list = sub_chain.serviceFunctions
                        target_func = func_list[0]
                        server_list = []
                        for server in self.O2AI_child_servers:
                            if server.canFitFunction(target_func):
                                score = server.availableMEM-target_func.mem
                                server_list.append({"server":server, "score":score})
                        server_list.sort(key=lambda x: x["score"])
                        
                        is_deployed = False
                        for server in server_list:
                            result = server["server"].O2AI_deploy(sub_chain, datacenter, whole_chain)
                            if result["deployed_num"] > 0:
                                deployed_num += result["deployed_num"]
                                sub_chain = result["remaining_chain"]
                                is_deployed = True
                                break

                        if not is_deployed:
                            # Cannot deploy anymore
                            break
                        if len(sub_chain.serviceFunctions) == 0:
                            # Done deploying
                            break
                        elif len(chain.serviceFunctions)-len(sub_chain.serviceFunctions) > lump_idx+1:
                            # Done BFit
                            is_re_predict = True
                            break
                    
                    
                elif bf_type == "BW":
                    # Deploy with bw
                    while True:
                        func_list = sub_chain.serviceFunctions
                        target_func = func_list[0]
                        server_list = []
                        for server in self.O2AI_child_servers:
                            if server.canFitFunction(target_func):
                                score = server.availableBW-target_func.bw
                                server_list.append({"server":server, "score":score})
                        server_list.sort(key=lambda x: x["score"])
                        
                        is_deployed = False
                        for server in server_list:
                            result = server["server"].O2AI_deploy(sub_chain, datacenter, whole_chain)
                            if result["deployed_num"] > 0:
                                deployed_num += result["deployed_num"]
                                sub_chain = result["remaining_chain"]
                                is_deployed = True
                                break

                        if not is_deployed:
                            # Cannot deploy anymore
                            break
                        if len(sub_chain.serviceFunctions) == 0:
                            # Done deploying
                            break
                        elif len(chain.serviceFunctions)-len(sub_chain.serviceFunctions) > lump_idx+1:
                            # Done BFit
                            is_re_predict = True
                            break
            
            if not is_re_predict:
                break
                
        
        server_idx_list = [function.server_idx for function in whole_chain.serviceFunctions if function.server_idx != None]
        current_latency = datacenter.computeLatency(server_idx_list)
        
        # Check if latency violated by BFit
        if current_latency > whole_chain.latency_req:
            # Fail due to latency violation
            for function in chain.serviceFunctions:
                if function.server_idx == None:
                    break
                else:
                    server = datacenter.servers[function.server_idx]
                    server.removeFunction(function)
                    function.server_idx = None
            sub_chain = chain
            deployed_num = 0
                
        # Done deploying without latency violation
        elif len(sub_chain.serviceFunctions) == 0:
            return {"remaining_chain":sub_chain, "deployed_num":deployed_num}
            
        
        # Do Ordered-FFit
        server_list = self.O2AI_child_servers
        func_list = sub_chain.serviceFunctions
        if len(func_list) > 0:
            result = self.O2AI_get_FFit_delay(server_list, func_list, height_cost)
            
            if result["type"] == "CPU":
                server_list.sort(key=lambda x: x.availableCPU, reverse=True)
            elif result["type"] == "MEM":
                server_list.sort(key=lambda x: x.availableMEM, reverse=True)
            elif result["type"] == "BW":
                server_list.sort(key=lambda x: x.availableBW, reverse=True)
            
            for server in server_list:
                result = server.O2AI_deploy(sub_chain, datacenter, whole_chain)
                deployed_num += result["deployed_num"]
                sub_chain = result["remaining_chain"]
                
                if len(sub_chain.serviceFunctions) == 0:
                    break
            
        
        server_idx_list = [function.server_idx for function in whole_chain.serviceFunctions if function.server_idx != None]
        current_latency = datacenter.computeLatency(server_idx_list)
        if current_latency > whole_chain.latency_req:
            # Fail after BFit and FFit (latency violation)
            for function in chain.serviceFunctions:
                if function.server_idx != None:
                    server = datacenter.servers[function.server_idx]
                    server.removeFunction(function)
                    function.server_idx = None
                else:
                    break
                    
            sub_chain = chain
            deployed_num = 0
            return {"remaining_chain":sub_chain, "deployed_num":deployed_num}
        
        else:
            return {"remaining_chain":sub_chain, "deployed_num":deployed_num}
        
        
        # (Done): Should I loop back and check
        #       1. if reach lump, then check
        
        # return {"remaining_chain", "deployed_num" chain}
    
    
    def O2AI_get_FFit_delay(self, server_list, func_list, height_cost):
        server_list.sort(key=lambda x: x.availableCPU, reverse=True)
        delay_cpu = self.O2AI_get_FFit_delay_sub(server_list, func_list, height_cost)
        
        server_list.sort(key=lambda x: x.availableMEM, reverse=True)
        delay_mem = self.O2AI_get_FFit_delay_sub(server_list, func_list, height_cost)
        
        server_list.sort(key=lambda x: x.availableBW, reverse=True)
        delay_bw = self.O2AI_get_FFit_delay_sub(server_list, func_list, height_cost)
        
        delay_list = [delay_cpu, delay_mem, delay_bw]
        min_delay = min(delay_list)
        min_idx = delay_list.index(min_delay)
        
        min_type = None
        if min_idx == 0:
            min_type = "CPU"
        elif min_idx == 1:
            min_type = "MEM"
        elif min_idx == 2:
            min_type = "BW"
        else:
            print("ERROR: cannot find min latency for FFit")
        
        return {"delay": min_delay, "type": min_type}
        
    def O2AI_get_FFit_delay_sub(self, sorted_server_list, func_list, height_cost):
        if len(func_list) == 0:
            return 0
        
        predicted_delay = -height_cost
        
        acc_func_list = [copy.deepcopy(func_list[0])]
        # Accumulate the functions
        for funct_idx in range(1, len(func_list)):
            acc_function = func_list[funct_idx].O2AI_copy()
            acc_function.O2AI_aggregate_function(acc_func_list[funct_idx-1])
            acc_func_list.append(acc_function)
        
        
        fail_idx = 0
        for server in sorted_server_list:
            is_fail = False
            is_any_feasible = False
            for acc_func_idx in range(fail_idx, len(acc_func_list)):
                if not server.canFitFunction(acc_func_list[acc_func_idx]):
                    is_fail = True
                    fail_idx = acc_func_idx
                    predicted_delay += height_cost
                    break
                is_any_feasible = True
                    
            if not is_fail:
                fail_idx = None
                break
            elif is_any_feasible:
                for acc_func_idx in range(fail_idx, len(acc_func_list)):
                    target_agg_function = acc_func_list[fail_idx-1]
                    acc_func_list[acc_func_idx].O2AI_rm_aggregate_function(target_agg_function)
        
        if fail_idx != None:
            predicted_delay += (len(acc_func_list)-fail_idx)*height_cost+2
            
        return predicted_delay
    
    
    
    def O2AI_aggregate(self, server):
        self.availableCPU += server.availableCPU
        self.availableMEM += server.availableMEM
        self.availableBW += server.availableBW
        
        
        
        
        
    def O2AI_BFit_deploy_score(self, function):
        is_success = self.can_fit_function(function)
    
        if is_success == True:
            # The server CAN deploy the function
            self.availableCPU -= function.cpu
            self.availableMEM -= function.mem
            self.availableBW -= function.bw
            return min(self.availableCPU, self.availableMEM, self.availableBW)
        else:
            return False
    

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
            
    
    
    # Heuristic Assignment with O2AI
    def assignChainO2AI(self, chain):
        # 1. Construct layer information
        idx_jump_edge = (self.k**3)/4
        idx_jump_agg = idx_jump_edge+(self.k**2)/2
        agg_servers = [ Server(idx+idx_jump_agg) for idx in range(self.k)]
        edge_servers = [ Server(idx+idx_jump_edge) for idx in range((self.k**2)/2)]
        
        # Construct edge level
        for idx in range(len(self.servers)):
            server = self.servers[idx]
            edge_idx = idx//(self.k/2)
            edge_servers[edge_idx].O2AI_child_servers.append(server)
        # Construct aggregate level
        for idx in range(len(edge_servers)):
            edge_server = edge_servers[idx]
            edge_server.O2AI_delay_cost = 2
            agg_idx = idx//(self.k/2)
            agg_servers[agg_idx].O2AI_child_servers.append(edge_server)
        # Construct root
        root_resource_tree = Server(idx_jump_agg+self.k)
        root_resource_tree.O2AI_delay_cost = 6
        for agg_server in agg_servers:
            root_resource_tree.O2AI_child_servers.append(agg_server)
            agg_server.O2AI_delay_cost = 4
            
        
        
        # Aggregate resource and construct edge level
        for idx in range(len(self.servers)):
            server = self.servers[idx]
            agg_idx = idx//((self.k**2)/4)
            edge_idx = idx//(self.k/2)
            agg_servers[agg_idx].O2AI_aggregate(server)
            edge_servers[edge_idx].O2AI_aggregate(server)
            edge_servers[edge_idx].O2AI_child_servers.append(server)
        # Construct aggregate level
        for idx in range(len(edge_servers)):
            edge_server = edge_servers[idx]
            agg_idx = idx//(self.k/2)
            agg_servers[agg_idx].O2AI_child_servers.append(edge_server)
    
        result = root_resource_tree.O2AI_deploy(chain, self, chain)
        
        if len(result['remaining_chain'].serviceFunctions) > 0:
            for function in chain.serviceFunctions:
                if function.server_idx != None:
                    server = self.servers[function.server_idx]
                    server.removeFunction(function)
                    function.server_idx = None
                else:
                    break
                    
            return False
        
        else:
            server_idx_list = [function.server_idx for function in chain.serviceFunctions]
            current_latency = self.computeLatency(server_idx_list)
            if current_latency > chain.latency_req:
                for function in chain.serviceFunctions:
                    server = self.servers[function.server_idx]
                    server.removeFunction(function)
                    function.server_idx = None
                sub_chain = chain
                deployed_num = 0
                
                return False
        
            
            # Deploy sucessfully
        
            # ============ For Debug =======================
            if result['deployed_num'] != len(chain.serviceFunctions):
                print ("Warning: deploy num not match")
            
            for function in chain.serviceFunctions:
                if function.server_idx == None:
                    print("Error: empty server index")
            
            server_idx_list = [function.server_idx for function in chain.serviceFunctions]
            
            current_latency = self.computeLatency(server_idx_list)
            if current_latency > chain.latency_req:
                # This should be handled during deployment
                print ("Error: latency requirement fail")
            #==================================================
            
            
            return True
        
    '''
    # BFit: optimizing-resource-oriented deployment
    def O2AI_BFit(self, chain, servers, delay_bound_server_num):
        # Formatting for return value
        best_result = None
        sub_chain_list = [] # (sub-chains on each server)
        func_list = [[] for i in range(len(servers))]
        jump_server_num = 0
        is_success = True
    
        previous_server = None # record server of last function to calculate hops
        for func in chain.serviceFunctions:
            # Find most-fit server for the function
            min_score = float("inf")
            min_server_idx = None
            for server_idx in range(len(servers)):
                server = servers[server_idx]
                result = server.O2AI_BFit_deploy_score(func)
                if result != False:
                     if result < min_score:
                        min_server_idx = server_idx
                        min_score = result
            
            if min_server_idx != None:
                func_list[min_server_idx].append(func)
                
                if (previous_server != None) and (previous_server != min_server_idx):
                    jump_server_num += 1
                    if jump_server_num > delay_bound_server_num:
                        is_success = False
                        break
                previous_server = min_server_idx
                
            else:
                break
        
        # TODO: return remain chain and results
                
            
    
    # FFit: optimizing-delay-oriented deployment
    def O2AI_FFit(self, chain, servers, delay_bound_server_num):
        sorted_server_idxs = []
        best_result = False
        
        # "CPU":
        sorted_server_idxs = sorted(range(len(servers)), key=lambda x: servers[x].availableCPU, reverse=True)
        result = self.O2AI_FFit_try_deploy(chain, sorted_server_idxs, servers, delay_bound_server_num)
        best_result = result
        
        # "Mem":
        sorted_server_idxs = sorted(range(len(servers)), key=lambda x: servers[x].availableMEM, reverse=True)
        result = self.O2AI_FFit_try_deploy(chain, sorted_server_idxs, servers, delay_bound_server_num)
        if result != False:
            if best_result == False:
                best_result = result
            elif result["jump_server_num"] < best_result["jump_server_num"]:
                best_result = result
        
        # "BW":
        sorted_server_idxs = sorted(range(len(servers)), key=lambda x: servers[x].availableBW, reverse=True)
        result = self.O2AI_FFit_try_deploy(chain, sorted_server_idxs, servers, delay_bound_server_num)
        if result != False:
            if best_result == False:
                best_result = result
            elif result["jump_server_num"] < best_result["jump_server_num"]:
                best_result = result
        
        
        # TODO: decide best result with remain chain
        return best_result
        
        
            
    def O2AI_FFit_try_deploy(self, chain, sorted_server_idxs, servers, delay_bound_server_num):
        function_success_num = 0
        sf = chain.serviceFunctions
        sub_chain_list = [ServiceChain() for i in range(len(servers))] # (sub-chains on each server)
        jump_server_num = -1
        ServiceChain(100, 5, randrange(cfg.SIMU_TIME-10), functs)
        
        
        for idx in sorted_server_idxs:
            server = servers[idx]
            func_list = []
            is_success = False
            
            agg_function = ServiceFunction(0,0,0)
            for func_idx in range(function_success_num, len(sf)):
                agg_function.O2AI_aggregate_function(sf[func_idx])
                if server.canFitFunction(agg_function):
                    is_success = True
                    function_success_num += 1
                    func_list.append(sf[func_idx])
                else:
                    break
            
            sub_chain = ServiceChain(chain, func_list)
            sub_chain_list[idx] = sub_chain
            
            if is_success:
                jump_server_num += 1
                if function_success_num == len(sf):
                    # Finish try deploying
                    break
                elif jump_server_num > delay_bound_server_num:
                    # Terminated because of exceeding delay bound
                    break
            
            # in this case, allow empty element. eg. [ [1], [], [2], [] ]
        
        if (jump_server_num > delay_bound_server_num):
            return False
        elif (function_success_num == len(sf)):
            return {"jump_server_num": jump_server_num, "sub_chain_list": sub_chain_list, "sorted_server_idxs": sorted_server_idxs, "remain_chain": []}
        else:
            # TODO: constuct remain chain
            return {"jump_server_num": jump_server_num, "sub_chain_list": sub_chain_list, "sorted_server_idxs": sorted_server_idxs, "remain_chain": }
    '''
    
    
    # Heuristic Assignment with SOVWin
    def assignChainSOVWin(self, chain):
        window = 1
        
        func_list = chain.serviceFunctions
        acc_func_list = [func_list[0].O2AI_copy()]
        # Accumulate the functions
        for funct_idx in range(1, len(func_list)):
            acc_function = func_list[funct_idx].O2AI_copy()
            acc_function.O2AI_aggregate_function(acc_func_list[funct_idx-1])
            acc_func_list.append(acc_function)
        
        is_success = False
        while True:
            # Try several windows
            
            min_server_idx_list = None
            min_latency = float("inf")
            for server_idx in range(len(self.servers)-window+1):
                server_list = self.servers[server_idx:server_idx+window]
                server_list.sort(key=lambda x: x.availableCPU+x.availableMEM+x.availableBW, reverse=True)
            
                function_ptr = 0
                server_idx_list = []
                for server in server_list:
                    is_feasible = True
                    for function_idx in range(function_ptr, len(acc_func_list)):
                        if not server.canFitFunction(acc_func_list[function_idx]):
                            function_ptr = function_idx
                            is_feasible = False
                            break
                        else:
                            server_idx_list.append(server.index)
                            
                    if is_feasible:
                        for function_idx in range(function_ptr, len(acc_func_list)):
                            target_agg_function = acc_func_list[function_ptr-1]
                            acc_func_list[function_idx].O2AI_rm_aggregate_function(target_agg_function)
                
                if len(server_idx_list) == len(func_list):
                    current_latency=self.computeLatency(server_idx_list)
                    if current_latency < min_latency:
                        min_latency = current_latency
                        min_server_idx_list = server_idx_list
            
            if min_server_idx_list != None:
                # Found solution
                for idx in range(len(min_server_idx_list)):
                    function = func_list[idx]
                    server = self.servers[min_server_idx_list[idx]]
                    server.addFunction(function)
                    function.server_idx = min_server_idx_list[idx]
                is_success = True
                break
            
            window += 1
            if window > len(self.servers):
                # Fail deployment
                is_success = False
                break
            
            if window > math.ceil(chain.latency_req/6) * ((self.k)**2)/4:
                break
        
        
        return is_success
        
        
    def getUtilization(self):
        #print([cfg.CPU_MAX-server.availableCPU for server in self.servers if cfg.CPU_MAX-server.availableCPU < 0])
        cpu_uti = sum([cfg.CPU_MAX-server.availableCPU for server in self.servers]) / float(len(self.servers)*cfg.CPU_MAX)
        mem_uti = sum([cfg.CPU_MAX-server.availableMEM for server in self.servers]) / float(len(self.servers)*cfg.MEM_MAX)
        bw_uti = sum([cfg.CPU_MAX-server.availableBW for server in self.servers]) / float(len(self.servers)*cfg.BW_MAX)
        return [cpu_uti, mem_uti, bw_uti]
            
    
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
        
    










