from queue import Queue

from FatTree_v2 import DataCenter
from ServiceChain import DummyGenChains
import config as cfg




if __name__ == "__main__":
	data_center = DataCenter(6)
    chains = DummyGenChains()
    
    ### TODO: event-based simulation
    # list of queue of event of chain arrival and finishes
    event_list = [  []  ]*SIMU_TIME
    
    for chain in chains:
        # TODO: insert arrival time of a chain  
        #(finish event is inserted after the chain is deployed sucessfully)
        None