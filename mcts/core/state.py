from mcts.core.action import Action
from typing import List, Any

class State:
  def getActions(self)->List[Action]:
    pass
  def takeAction(self, action:Action)->'State':
    pass
  def isTerminal(self)->bool:
    pass
  def getReward(self)->Any:
    pass