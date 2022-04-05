from mcts.core import State, Action
from typing import List

def linearExpansion(state:State)->List[Action]:
  return state.getActions()