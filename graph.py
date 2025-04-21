# TODO:
# 1. Shortest Path
# 2. Maximum Flow
from logger import logger


class Graph:
    WEIGHT_UPPER_BOUND=None # No constranint by default
    def __init__(self, data=None, directed:bool=False):
        logger.debug(f"initialize with {data}")
        self.data = data if data else {}
        self.directed = directed
        """
        data = {
            0 : { (2, w1), (3, w2)},
            1 : { (...)}
        }
        """
    def weightValidation(self, weight:float)->bool:
        # Validate the weight
        # You can re-write this constraint function
        if not self.WEIGHT_UPPER_BOUND:
            return True # No constraint for weight
        else:
            return weight>=self.WEIGHT_UPPER_BOUND # Check the upper bound
        
    def set(self, start_node, end_node, weight):
        assert start_node is not None, "start_node should not be None"
        assert self.weightValidation(weight), f"The input weight {weight} is not validated"

        def _set(start_node, end_node, weight):
            if start_node in self.data:
                nodes_conn_with_start = self.data[start_node]
                logger.debug(nodes_conn_with_start)
                for idx, (p, p_weight) in enumerate(nodes_conn_with_start):
                    if p == end_node: # the `end_node` is existent
                        logger.debug(f"Set {(p, p_weight)} to {(p, p_weight)}")
                        self.data[start_node][idx] = (p, weight)
                else: # the `end_node` doesnt exist
                    logger.debug(f"Set new end node for {start_node}")
                    self.data[start_node].add(  (end_node, weight)  )
            else:
                # The start node doesnt exist
                logger.debug(f"the start node {start_node} does not exist, set {start_node}: {(end_node, weight)}")
                self.data[start_node] = { (end_node, weight) }

            # log the updated data
            logger.debug(f"Updated data: {self.data}")


        _set(start_node, end_node, weight)

        if not self.directed: # undirected graph
            logger.debug(f"set the inverse path from {end_node} to  {start_node} due to undirected")
            _set(end_node, start_node, weight)
        
    def __str__(self): 
        s = ""
        for start_node, v in self.data.items():
            s += f"{start_node}: {v}\n"
        return s

if __name__ == "__main__":
    g = Graph()
    g.set(0, 2, 1.6)
    logger.debug(g)
    g.set(0, 1, 3.14)

