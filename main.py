from FatTree import DataCenter
from ServiceChain import DummyGenChains, PoissonGenChains
import config as cfg
import sys
import random


if __name__ == "__main__":
    seed = sys.argv[1]
    random.seed(seed)
    
    
    # Simulation time is in 0.1ms
    arrival_rate = 1000
    service_rate = 10
    
    k = 6

    data_center = DataCenter(k)
    #chains = DummyGenChains()
    
    chains = PoissonGenChains(k, arrival_rate, service_rate)
    
    # list of queue of event of chain arrival and finishes
    event_list = [  {'arrive':[], 'finish':[], 'queue':[]} for idx in range(cfg.SIMU_TIME)]
    
    for chain in chains:
        t_idx = chain.arrive_time
        event_list[t_idx]['arrive'].append(chain)
        
    
    # Statistics
    resource_utilization = [[], [], []]
    
    # Start simulation
    for t_idx in range(len(event_list)):
        print(t_idx)
        # 1. handle finished chain
        for chain in (event_list[t_idx]['finish']):
            data_center.removeChain(chain)
            
        # 2. handle queued chain
        for chain in (event_list[t_idx]['queue']):
            #is_success = data_center.assignChainRL(chain)
            is_success = data_center.assignChainO2AI(chain)
            #is_success = data_center.assignChainSOVWin(chain)
            
            if is_success:
                # Insert finish event
                chain.finish_time = chain.serviceTime + t_idx
                if (chain.finish_time < len(event_list)):
                    event_list[chain.finish_time]['finish'].append(chain)
            else:
                # Record waiting, and queue to next time (at the first of the queue)
                chain.waitingTime += 1
                if (t_idx+1 < len(event_list)):
                    event_list[t_idx+1]['queue'].insert(0, chain)
        
        # 3. handle arrival chain
        for chain in (event_list[t_idx]['arrive']):
            #is_success = data_center.assignChainRL(chain)
            is_success = data_center.assignChainO2AI(chain)
            #is_success = data_center.assignChainSOVWin(chain)
            
            if is_success:
                # Insert finish event
                chain.finish_time = chain.serviceTime + t_idx
                if (chain.finish_time < len(event_list)):
                    event_list[chain.finish_time]['finish'].append(chain)
            else:
                # Record waiting, and queue to next time (at the first of the queue)
                chain.waitingTime += 1
                if (t_idx+1 < len(event_list)):
                    event_list[t_idx+1]['queue'].insert(0, chain)
          
          
        utilization = data_center.getUtilization()
        #print(utilization)
        for idx in range(3):
            resource_utilization[idx].append(utilization[idx])
        
    
    # Results
    waiting_list = [chain.waitingTime for chain in chains if chain.finish_time != None]
    print("Average waiting time is ", len(waiting_list), sum(waiting_list)/len(waiting_list))
    
    for idx in range(3):
        resource_utilization[idx] = sum(resource_utilization[idx])/len(resource_utilization[idx])
    print("DataCenter utilization (CPU, MEM, BW): ", resource_utilization)