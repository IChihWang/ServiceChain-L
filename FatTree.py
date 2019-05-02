import config as cfg
from DeployAlg import DummyAlgorithm
import numpy as np


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

    def get_obs(self):

        tmp = np.vstack((self.server_state, self.chain_state))
        tmp = tmp.reshape(1,-1)
        assert self.state_dim == tmp.shape[1]
        return tmp   #[self.server_state, self.chain_state]

    def construct_chain_state(self, chain_state):
        # state of One chain's function
        mu = np.mean(chain_state)
        var = np.var(chain_state)
        #chain_state = (chain_state-mu)/var
        self.chain_state = chain_state
        self.chain_state_ptr = 0

    def construct_server_state(self, servers):
        state = [(a.availableCPU, a.availableMEM, a.availableBW, a.delayFactor) for a in servers]
        state = np.asarray(state)
        state = state.astype(float)
        mu = 100 #np.mean(state[:,0:-1])
        var = np.var(state[:,0:-1]) + 0.0001
        #print(mu,var)
        #exit()
        state[:,0:-1] = (state[:,0:-1]-mu)/var
        #state[:, 0:-1] = state[:, 0:-1]/100.0
        #print(type(state))
        #print("serverstate:", state)
        self.server_state = state

    def norm2_reward(self, servers):
        state = [(a.availableCPU, a.availableMEM, a.availableBW) for a in servers]
        state = np.asarray(state)
        state = state.astype(float)
        state[:,0] = state[:,0]/self.server_max[0]
        state[:, 1] = state[:, 1] / self.server_max[1]
        state[:, 2] = state[:, 2] / self.server_max[2]
        state = np.square(state)

        return np.sum(state)/10.0


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
        #print(self.chain_state)
        #exit()


    def update_wall_time(self):
        self.wall_time += 1

    def reset(self):
        self.ep_ret = 0
        self.counter = 0
        self.finishtimes += 1
        self.chain_cnt = 0

    def update_chain_cnt(self):
        self.chain_cnt += 1



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
        self.servers=[Server() for idx in range(k**3//4)]
        self.chains=[]
        self.counter = 0


    def assignChain(self, chain, env, agent, use_dummy=False):
        is_success = True

        # Initialize
        self.initDelayFactor()

        # Try to deploy the chain
        env.update_server_max(self.servers)

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
            #print("a:", a)
            server_idx = a[0]
            env.step()

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
                print("Fail")
                is_deployed = False

            # store
            if is_deployed:
                reward = 0
            else:
                reward = -10
            agent.buf.store(rl_state, a, reward, v_t, logp_t)


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
            reward = 10
            reward = env.norm2_reward(self.servers)
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
        if env.chain_cnt >5:
            terminal = True
            last_val = agent.sess.run(agent.v, feed_dict={agent.x_ph: rl_state.reshape(1, -1)})
            agent.buf.finish_path(last_val)
            env.episode_counter += 1
            agent.log_tf(env.ep_ret, 'Return', env.episode_counter)
            print('Return:', env.ep_ret)

            if env.finishtimes % 5 == 0:
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
            #print("a:", a)
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
                print("Fail")
                is_deployed = False

            # store
            if is_deployed:
                reward = 0
            else:
                print("Fail")
                reward = -10

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

        if env.chain_cnt >5:

            env.episode_counter += 1

            print('Return:', env.ep_ret)

            env.reset()

        return is_success







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


    def constructRLstate(self, servers):
        # TODO: construct RL state (return with RL state)
        state = [(a.availableCPU, a.availableMEM, a.availableBW, a.delayFactor) for a in servers]
        state = np.asarray(state)
        #state_mem = [a.availableMEM for a in servers]
        #state_mem = [a.availableBW for a in servers]

        print(state)
        #input("skjdflsd")




        return None












