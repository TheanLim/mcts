from copy import deepcopy
from mcts.core import State
from typing import Any
import random

def randomRollout(state:State)->Any:
  '''
  Starting from the provided state, randomly take actions
  until the terminal state. 
  Returns the terminal state reward.
  '''
  # Deep copy to perform takeAction that doesnt preserve state
  # This is done to speed up rollout
  state = deepcopy(state)
  while not state.isTerminal():
    action = random.choice(state.getActions())
    state = state.takeAction(action, preserveState=False)
  return state.getReward()