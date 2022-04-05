from mcts.core import State
from typing import Any
import random

def randomRollout(state:State)->Any:
  '''
  Starting from the provided state, randomly take actions
  until the terminal state. 
  Returns the terminal state reward.
  '''
  while not state.isTerminal():
      action = random.choice(state.getActions())
      state = state.takeAction(action)
  return state.getReward()