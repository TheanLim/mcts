from mcts.core import State
from typing import Any
import random

def randomRollout(state:State)->Any:
  while not state.isTerminal():
      action = random.choice(state.getActions())
      state = state.takeAction(action)
  return state.getReward()