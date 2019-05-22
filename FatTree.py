import config as cfg
import itertools
import copy
from DeployAlg import DummyAlgorithm
import numpy as np
from utils import logger
from ServiceChain import ServiceFunction, ServiceChain
import math

class State:
    def __init__(self, state_dim):
        self.s0 = 0
        self.s1 = 0
        self.server_state = 0
        self.chain_state = 0
        self.state_dim = state_dim
        self.counter = 0
        self.ep_ret = 0
        self.wall_time = 0
        self.episode_counter = 0
        self.finishtimes = 0
        self.chain_cnt = 0
        self.last_state = None
        self.is_last_in_time = False

    def get_obs(self):

        tmp = np.vstack((self.server_state, self.chain_state))
        tmp = tmp.reshape(1,-1)
        tmp = np.concatenate((tmp, np.array([[self.chain_waiting_time]])), axis=1)


        assert self.state_dim == tmp.shape[1]
        self.last_state = tmp
        return tmp   #[self.server_state, self.chain_state]

    def construct_chain_state(self, chain_state, waiting_time=0):
        # state of One chain's function
        mu = np.mean(chain_state)
        var = np.var(chain_state)
        #chain_state = (chain_state-mu)/var
        self.chain_state = chain_state
        self.chain_waiting_time = waiting_time
        self.chain_state_ptr = 0

    def construct_normalize_reward(self, servers):
        state = [(a.availableCPU, a.availableMEM, a.availableBW) for a in servers]
        state = np.asarray(state)
        state = state.astype(float)
        state = state/100.0
        state = np.square(state)
        self.reward_norm_factor = np.sum(state)
        #print("norm_factor:", self.reward_norm_factor)




    def construct_server_state(self, servers):
        state = [(a.availableCPU, a.availableMEM, a.availableBW, a.delayFactor) for a in servers]
        state = np.asarray(state)
        state = state.astype(float)
        '''
        mu = 100 #np.mean(state[:,0:-1])
        var = np.var(state[:,0:-1]) + 0.0001
        #print(mu,var)
        #exit()
        state[:,0:-1] = (state[:,0:-1]-mu)/50 #/var
        #state[:, 0:-1] = state[:, 0:-1]/100.0
        '''
        # scaling to [0-1]
        state[:, 0:-1] = state[:,0:-1]/ 100.0 - 0.5
        #print("serverstate:", state)

        ##
        self.server_state = state

    def norm2_reward(self, servers):
        state = [(a.availableCPU, a.availableMEM, a.availableBW) for a in servers]
        state = np.asarray(state)
        state = state.astype(float)

        #state = state / 100.0

        state[:,0] = state[:,0]/self.server_max[0]
        state[:, 1] = state[:, 1] / self.server_max[1]
        state[:, 2] = state[:, 2] / self.server_max[2]
        state = np.square(state)

        #return np.sum(state)/self.reward_norm_factor
        #return np.sum(state)/100

        return np.sum(state) / 10

    def wait_reward(self):

        return 1.0/(self.chain_waiting_time+1)



    def update_server_max(self, servers):
        state = [(a.availableCPU, a.availableMEM, a.availableBW) for a in servers]
        #max(state)
        state = np.asarray(state)
        state = state.astype(float)
        self.server_max = (max(state[:,0]), max(state[:,1]),max(state[:,2]))
        #print(self.server_max)


    def step(self):
        self.counter += 1
        self.chain_state[self.chain_state_ptr] = 0
        self.chain_state_ptr += 1
        #done =agent.buf.is_full()
        #if self.counter == 1999:
        #    done = True
        #return agent.buf.is_full()
        #print(self.chain_state)
        #exit()


    def update_wall_time(self):
        self.wall_time += 1

    def reset(self):
        self.ep_ret = 0
        self.counter = 0
        self.finishtimes += 1
        self.chain_cnt = 0
        self.last_state = None
        self.is_last_in_time = False

    def update_chain_cnt(self):
        self.chain_cnt += 1



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
        self.servers=[Server(idx) for idx in range(k**3//4)]
        self.chains=[]
        self.counter = 0


    def assignChain(self, chain, env, agent, use_dummy=False):
        is_success = True

        # Initialize
        self.initDelayFactor()

        # Try to deploy the chain
        env.update_server_max(self.servers)
        env.construct_normalize_reward(self.servers)
        #env.norm2_reward(self.servers)
        env.update_chain_cnt()
        for function_idx in range(len(chain.serviceFunctions)):

            target_function = chain.serviceFunctions[function_idx]

            #input("dsfsd")

            # Find the server for the function through the algorithm
            ### TODO: Construct RLstate
            #RLstate = self.constructRLstate(self.servers)
            env.construct_server_state(self.servers)
            rl_state = env.get_obs()

            #print("get_obs", rl_state)
            #exit()



            # Agent predict  action
            a, v_t, logp_t = agent.sess.run(agent.get_action_ops, feed_dict={agent.x_ph: rl_state.reshape(1, -1)})
            #print("a:", a, logp_t)
            server_idx = a[0]
            #terminal = agent.buf.is_full() #env.step()
            env.step()
            #logger.info("counter:{}".format(env.counter))

            # Dummy
            if use_dummy:
                server_idx = DummyAlgorithm(target_function, rl_state)
            target_server = self.servers[server_idx]
            #print(target_server.index)
            #exit()

            # Try to deploy the function
            if (chain.current_latency_req >= target_server.delayFactor):
                is_deployed = target_server.addFunction(target_function)

            else:
                # Reject directly if latency requirement doesn't meet
                print("latency Fail:", function_idx, server_idx)
                is_deployed = False

            # store
            if is_deployed:
                reward = 0
            else:
                print("resource Fail:", function_idx, server_idx)
                #print("agent ptr", agent.buf.ptr)
                reward = -0.5


            agent.buf.store(rl_state, a, reward, v_t, logp_t)
            terminal = agent.buf.is_full()

            if is_deployed:
                # Record the server idx on function if deployment succeed
                target_function.server_idx = server_idx
                chain.current_latency_req -= target_server.delayFactor
                # Update delay factorz
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
            #print("success")
            # If deploy chain sucessfully
            self.chains.append(chain)

            #reward = env.norm2_reward(self.servers)
            reward = env.wait_reward()

            #logger.info("success reward: {}".format(reward))
            agent.buf.overwrite_last_rew(reward)
            env.ep_ret += reward

            #return True

        else:
            # If not deploy chain sucessfully
            #print("Fail")
            env.ep_ret += reward
            #return False

        # assume infinte horizon now
        magic = 100#np.random.randint(20, 66)

        #if env.counter > magic :
        #if env.chain_cnt >=5:
        if env.is_last_in_time or terminal:

            if terminal:
                logger.info("Buffer full")
                print("Buffer Full")

            #last_val = agent.sess.run(agent.v, feed_dict={agent.x_ph: rl_state.reshape(1, -1)})
            last_val = env.ep_ret

            agent.buf.finish_path(last_val)
            env.episode_counter += 1
            agent.log_tf(env.ep_ret, 'Return', env.episode_counter)
            print('Return:', env.ep_ret)
            logger.info("Return:{}, PTR:{}".format(env.ep_ret, agent.buf.ptr))
            if env.finishtimes % 200 == 0 or agent.buf.is_full():
                logger.info("update")
                print("!!!!!!!!!!!!Update")
                agent.update()
            #print("Doing update:", env.episode_counter)
            env.reset()

        return is_success

    def assignChain_eval(self, chain, env, agent, use_dummy=False):
        is_success = True

        # Initialize
        self.initDelayFactor()

        # Try to deploy the chain
        env.update_server_max(self.servers)

        #env.norm2_reward(self.servers)
        env.update_chain_cnt()
        for function_idx in range(len(chain.serviceFunctions)):

            target_function = chain.serviceFunctions[function_idx]


            # Find the server for the function through the algorithm
            ### TODO: Construct RLstate
            #RLstate = self.constructRLstate(self.servers)
            env.construct_server_state(self.servers)
            rl_state = env.get_obs()


            # Agent predict  action
            a, v_t, logp_t = agent.sess.run(agent.get_action_ops, feed_dict={agent.x_ph: rl_state.reshape(1, -1)})
            #print("a:", a, logp_t)
            server_idx = a[0]
            env.step()

            # Dummy
            if use_dummy:
                server_idx = DummyAlgorithm(target_function, rl_state)
            target_server = self.servers[server_idx]

            # Try to deploy the function
            if (chain.current_latency_req >= target_server.delayFactor):
                is_deployed = target_server.addFunction(target_function)

            else:
                # Reject directly if latency requirement doesn't meet
                #print("Fail")
                print("latency Fail:", function_idx, server_idx)
                is_deployed = False

            # store
            if is_deployed:
                reward = 0
            else:
                #print("Fail")
                print("resource Fail:", function_idx, server_idx)
                reward = -2

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
            #print("success")
            # If deploy chain sucessfully
            self.chains.append(chain)
            reward = 10
            reward = env.norm2_reward(self.servers)
            agent.buf.overwrite_last_rew(reward)
            env.ep_ret += reward

            #return True

        else:
            # If not deploy chain sucessfully
            env.ep_ret += reward
            #return False

        if env.is_last_in_time:

            env.episode_counter += 1

            #print('Return:', env.ep_ret)

            env.reset()

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


    def constructRLstate(self, servers):
        # TODO: construct RL state (return with RL state)
        state = [(a.availableCPU, a.availableMEM, a.availableBW, a.delayFactor) for a in servers]
        state = np.asarray(state)
        #state_mem = [a.availableMEM for a in servers]
        #state_mem = [a.availableBW for a in servers]

        print(state)
        #input("skjdflsd")




        return None












