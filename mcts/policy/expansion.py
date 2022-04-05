from mcts.core import State, Action
from typing import List

def linearExpansion(state:State)->List[Action]:
  '''
  Returns a list of actions in a sequence 
  that are encoded by the state.
  '''
  return state.getActions()