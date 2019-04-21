from FatTree import DataCenter, State
from ServiceChain import DummyGenChains
from agent import Agent
import config as cfg
import time
import numpy as np
import random
from utils import logger
import argparse

def get_all_func(envlist):
    print("get_all_func")
    chain_states = []
    # Assume every chain has the same number of functions
    for chain in envlist:
        chain_state = [(func.cpu, func.mem, func.bw) for func in chain.serviceFunctions]
        chain_states.extend(chain_state)
        #for i, func in enumerate(chain.serviceFunctions):

        #    print(i, func.mem)
    chain_states = np.asarray(chain_states)
    print(chain_states.shape)
    print("chain_states:", chain_states)
    return chain_states


def get_func_in_chain(chain):
    #print("get_func_in_chain")
    latency_req = chain.latency_req
    chain_state = [(func.cpu, func.mem, func.bw, latency_req) for func in chain.serviceFunctions]
    chain_state = np.asarray(chain_state)
    #print("chain_State:", chain_state)
    return chain_state

'''
def train_():

    event_list = [{'arrive': [], 'finish': [], 'queue': []} for idx in range(cfg.SIMU_TIME)]

    for chain in chains:
        t_idx = chain.arrive_time
        event_list[t_idx]['arrive'].append(chain)

    # Start simulation
    for t_idx in range(cfg.SIMU_TIME - 1):

        print("t_idx:", t_idx)
        env.update_wall_time()

        # 1. handle finished chain
        for chain in (event_list[t_idx]['finish']):
            data_center.removeChain(chain)

        # 2. handle queued chain
        # start = time.time()

        # get_all_func(event_list[t_idx]['queue'])
        # get_all_func(event_list[t_idx]['arrive'])

        # end = time.time()
        # print("time:", end-start)
        for chain in (event_list[t_idx]['queue']):
            chain_state = get_func_in_chain(chain)
            env.construct_chain_state(chain_state)

            # step(a)
            is_success = data_center.assignChain(chain, env, agent)

            if is_success:
                # Insert finish event
                chain.finish_time = chain.serviceTime + t_idx

                event_list[chain.finish_time]['finish'].append(chain)
            else:
                # Record waiting, and queue to next time (at the first of the queue)
                chain.waitingTime += 1
                event_list[t_idx + 1]['queue'].insert(0, chain)

        # 3. handle arrival chain

        for chain in (event_list[t_idx]['arrive']):
            chain_state = get_func_in_chain(chain)
            env.construct_chain_state(chain_state)

            is_success = data_center.assignChain(chain, env, agent)

            if is_success:
                # Insert finish event
                chain.finish_time = chain.serviceTime + t_idx
                # print("sT", chain.serviceTime)
                # input("sdfsd")
                event_list[chain.finish_time]['finish'].append(chain)
            else:
                # Record waiting, and queue to next time (at the first of the queue)
                chain.waitingTime += 1
                event_list[t_idx + 1]['queue'].insert(0, chain)
'''

def train():
    #random.seed(7)
    #data_center = DataCenter(6)
    #chains = DummyGenChains()
    #print(len(chains))

    epochs = 3000

    # list of queue of event of chain arrival and finishes
    #event_list = [{'arrive': [], 'finish': [], 'queue': []} for idx in range(cfg.SIMU_TIME)]

    for epoch in range(epochs):

        print("epoch:", epoch)
        random.seed(7)
        data_center = DataCenter(6)
        chains = DummyGenChains()
        #print(len(chains))
        event_list = [{'arrive': [], 'finish': [], 'queue': []} for idx in range(cfg.SIMU_TIME_TRAIN)]

        for chain in chains:
            t_idx = chain.arrive_time
            event_list[t_idx]['arrive'].append(chain)

        # Start simulation
        for t_idx in range(cfg.SIMU_TIME_TRAIN - 1):

            #print("t_idx:", t_idx)
            env.update_wall_time()

            # 1. handle finished chain
            for chain in (event_list[t_idx]['finish']):
                data_center.removeChain(chain)

            # 2. handle queued chain
            # start = time.time()
            # get_all_func(event_list[t_idx]['queue'])
            # get_all_func(event_list[t_idx]['arrive'])

            # end = time.time()
            # print("time:", end-start)
            for chain in (event_list[t_idx]['queue']):
                chain_state = get_func_in_chain(chain)
                env.construct_chain_state(chain_state)

                # step(a)
                is_success = data_center.assignChain(chain, env, agent)

                if is_success:
                    # Insert finish event
                    chain.finish_time = chain.serviceTime + t_idx

                    event_list[chain.finish_time]['finish'].append(chain)
                else:
                    # Record waiting, and queue to next time (at the first of the queue)
                    chain.waitingTime += 1
                    event_list[t_idx + 1]['queue'].insert(0, chain)

            # 3. handle arrival chain

            for chain in (event_list[t_idx]['arrive']):
                chain_state = get_func_in_chain(chain)
                env.construct_chain_state(chain_state)

                is_success = data_center.assignChain(chain, env, agent)

                if is_success:
                    # Insert finish event
                    chain.finish_time = chain.serviceTime + t_idx
                    # print("sT", chain.serviceTime)
                    # input("sdfsd")
                    event_list[chain.finish_time]['finish'].append(chain)
                else:
                    # Record waiting, and queue to next time (at the first of the queue)
                    chain.waitingTime += 1
                    event_list[t_idx + 1]['queue'].insert(0, chain)

        if (epoch % cfg.VAL_FREQENCY) == 0:
            #val_performance()
            waiting_list = [chain.waitingTime if chain.finish_time is not None else 1000 for chain in chains]
            agent.log_tf(sum(waiting_list) / len(waiting_list), tag='Avg Waiting Time', step_counter= epoch)
            logger.info("Epoch{}, Avg waiting time{}".format(epoch, sum(waiting_list) / len(waiting_list)))
            print("Average waiting time is ", len(waiting_list), sum(waiting_list) / len(waiting_list))

        if epoch % cfg.Save_Model_Epoch == 0:
            logger.info("Save Model")
            agent.save_model(epoch)
    return chains




def eval():
    #random.seed(7)
    #data_center = DataCenter(6)
    #chains = DummyGenChains()
    #print(len(chains))

    epochs = 1


    # list of queue of event of chain arrival and finishes
    #event_list = [{'arrive': [], 'finish': [], 'queue': []} for idx in range(cfg.SIMU_TIME)]

    for epoch in range(epochs):

        print("epoch:", epoch)
        random.seed(7)
        data_center = DataCenter(6)
        chains = DummyGenChains()
        print(len(chains))
        event_list = [{'arrive': [], 'finish': [], 'queue': []} for idx in range(cfg.SIMU_TIME)]


        for chain in chains:
            t_idx = chain.arrive_time
            event_list[t_idx]['arrive'].append(chain)
        #print(event_list)
        # Start simulation
        for t_idx in range(cfg.SIMU_TIME - 1):

            print("t_idx:", t_idx)
            env.update_wall_time()

            # 1. handle finished chain
            for chain in (event_list[t_idx]['finish']):
                data_center.removeChain(chain)

            # 2. handle queued chain
            # start = time.time()
            # get_all_func(event_list[t_idx]['queue'])
            # get_all_func(event_list[t_idx]['arrive'])

            # end = time.time()
            # print("time:", end-start)
            for chain in (event_list[t_idx]['queue']):
                chain_state = get_func_in_chain(chain)
                env.construct_chain_state(chain_state)

                # step(a)
                is_success = data_center.assignChain_eval(chain, env, agent, True)

                if is_success:
                    # Insert finish event
                    chain.finish_time = chain.serviceTime + t_idx

                    event_list[chain.finish_time]['finish'].append(chain)
                else:
                    # Record waiting, and queue to next time (at the first of the queue)
                    chain.waitingTime += 1
                    event_list[t_idx + 1]['queue'].insert(0, chain)

            # 3. handle arrival chain

            for chain in (event_list[t_idx]['arrive']):
                chain_state = get_func_in_chain(chain)
                env.construct_chain_state(chain_state)

                is_success = data_center.assignChain_eval(chain, env, agent, True)

                if is_success:
                    # Insert finish event
                    chain.finish_time = chain.serviceTime + t_idx
                    # print("sT", chain.serviceTime)
                    # input("sdfsd")
                    event_list[chain.finish_time]['finish'].append(chain)
                else:
                    # Record waiting, and queue to next time (at the first of the queue)
                    chain.waitingTime += 1
                    event_list[t_idx + 1]['queue'].insert(0, chain)

    return chains



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--pi_lr', type=float, default=1e-3)
    parser.add_argument('--v_lr', type=float, default=1e-3)
    config_rl = parser.parse_args()

    local_steps_per_epoch = 1000
    state_dim = int(6 ** 3 // 4 * 4 + 4 * 3)
    print("state_dim:", state_dim)
    env = State(state_dim)
    # exit()
    action_dim = 54
    agent = Agent(state_dim, action_dim, config_rl,steps_per_epoch=local_steps_per_epoch)
    if config_rl.load:
        agent.load_model()
        print("load")
    chains = train()
    #chains = eval()

    # Results
    waiting_list = [chain.waitingTime if chain.finish_time is not None else 1000 for chain in chains]
    print(waiting_list)
    print("Average waiting time is ", len(waiting_list), sum(waiting_list) / len(waiting_list))
