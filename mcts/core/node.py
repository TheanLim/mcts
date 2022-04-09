from mcts.core.state import State

'''
A Prototype of a Node
'''
class Node:
  def __init__(self, state:State, parent=None):
    self.state = state
    self.parent = parent
    self.children = {} # {action:Node(stateAfterAction, self)}
    self.numVisits = 0
    self.utilities = None
  
  def isLeaf(self)->bool:
    # A terminal state is considered a leaf node
    return len(self.children)==0 or self.state.isTerminal()
  
  def __str__(self)->str:
    s=[]
    s.append("rewards: "+str(self.utilities))
    s.append("numVisits: "+str(self.numVisits))
    s.append("children/actions: " + str(list(self.children.keys())))
    return str(self.__class__.__name__)+": {"+", ".join(s)+"}"