import FatTree
import ServiceFunction
import ServiceChain

if __name__ == "__main__":
	sChain = ServiceChain(4,8)
	sChain.ServiceFunctions(ServiceFunction(50))