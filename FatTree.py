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

    def get_obs(self):

        tmp = np.vstack((self.server_state, self.chain_state))
        tmp = tmp.reshape(1,-1)
        assert self.state_dim == tmp.shape[1]
        return tmp   #[self.server_state, self.chain_state]

    def construct_chain_state(self, chain_state):
        # state of One chain's function
        self.chain_state = chain_state

    def construct_server_state(self, servers):
        state = [(a.availableCPU, a.availableMEM, a.availableBW, a.delayFactor) for a in servers]
        state = np.asarray(state)
        self.server_state = state

    def step(self):
        self.counter += 1


    def update_wall_time(self):
        self.wall_time += 1

    def reset(self):
        self.ep_ret = 0
        self.counter = 0



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
        for function_idx in range(len(chain.serviceFunctions)):

            target_function = chain.serviceFunctions[function_idx]

            #input("dsfsd")

            # Find the server for the function through the algorithm
            ### TODO: Construct RLstate
            #RLstate = self.constructRLstate(self.servers)
            env.construct_server_state(self.servers)
            rl_state = env.get_obs()
            #print("get_obs", rl_state.shape)
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
                reward = -1
            agent.buf.store(rl_state, a, reward, v_t, logp_t)


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
            reward = 1
            agent.buf.overwrite_last_rew(reward)
            env.ep_ret += reward

            #return True

        else:
            # If not deploy chain sucessfully
            #print("Fail")
            env.ep_ret += reward
            #return False

        # assume infinte horizon now
        if env.counter > 400:

            terminal = True
            last_val = agent.sess.run(agent.v, feed_dict={agent.x_ph: rl_state.reshape(1, -1)})
            agent.buf.finish_path(last_val)
            env.episode_counter += 1
            agent.log_tf(env.ep_ret, 'Return', env.episode_counter)
            print('Return:', env.ep_ret)
            agent.update()
            #print("Doing update:", env.episode_counter)
            env.reset()

        return is_success

    def assignChain_eval(self, chain, env, agent, use_dummy=False):
        is_success = True

        # Initialize
        self.initDelayFactor()

        # Try to deploy the chain
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
                reward = -1

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
            print("success")
            # If deploy chain sucessfully
            self.chains.append(chain)
            reward = 1
            agent.buf.overwrite_last_rew(reward)
            env.ep_ret += reward
            #return True

        else:
            # If not deploy chain sucessfully
            env.ep_ret += reward
            #return False

        # assume infinte horizon now
        if env.counter > 200:
            env.episode_counter += 1
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












