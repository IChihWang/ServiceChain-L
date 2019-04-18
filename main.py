from FatTree import DataCenter
from ServiceChain import DummyGenChains
import config as cfg


if __name__ == "__main__":
    data_center = DataCenter(6)
    chains = DummyGenChains()
    
    # list of queue of event of chain arrival and finishes
    event_list = [  {'arrive':[], 'finish':[], 'queue':[]} for idx in range(cfg.SIMU_TIME)]
    
    for chain in chains:
        t_idx = chain.arrive_time
        event_list[t_idx]['arrive'].append(chain)
        
    
    # Start simulation
    for t_idx in range(cfg.SIMU_TIME-1):
        print(t_idx)
        # 1. handle finished chain
        for chain in (event_list[t_idx]['finish']):
            data_center.removeChain(chain)
            
        # 2. handle queued chain
        for chain in (event_list[t_idx]['queue']):
            is_success = data_center.assignChain(chain)
            
            if is_success:
                # Insert finish event
                chain.finish_time = chain.serviceTime + t_idx
                event_list[chain.finish_time]['finish'].append(chain)
            else:
                # Record waiting, and queue to next time (at the first of the queue)
                chain.waitingTime += 1
                event_list[t_idx+1]['queue'].insert(0, chain)
        
        # 3. handle arrival chain
        for chain in (event_list[t_idx]['arrive']):
            #is_success = data_center.assignChainRL(chain)
            is_success = data_center.assignChainBF(chain)
            
            if is_success:
                # Insert finish event
                chain.finish_time = chain.serviceTime + t_idx
                event_list[chain.finish_time]['finish'].append(chain)
            else:
                # Record waiting, and queue to next time (at the first of the queue)
                chain.waitingTime += 1
                event_list[t_idx+1]['queue'].insert(0, chain)
    
    # Results
    waiting_list = [chain.waitingTime for chain in chains if chain.finish_time != None]
    print("Average waiting time is ", len(waiting_list), sum(waiting_list)/len(waiting_list))