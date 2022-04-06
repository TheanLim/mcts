from mcts.core.action import Action
from typing import List, Any

'''
A Prototype of a State
'''
class State:
  def getActions(self)->List[Action]:
    pass
  def takeAction(self, action:Action, preserveState:bool=True)->'State':
    '''
    preserveState: if True, make a copy of the current state, act and return the copied state.
    '''
    pass
  def isTerminal(self)->bool:
    pass
  def getReward(self)->Any:
    pass