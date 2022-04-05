from mcts import MCTS
from mcts.core import Search, State, Action
from mcts.policy import UCB, linearExpansion, randomRollout
from mcts.utils import sumTuple, Random, gamePlay

from typing import List, Any, Tuple, Dict, Optional, Any
from copy import deepcopy

class MNKAction(Action):
  def __init__(self, playerSign:Any, m:int, n:int):
    self.playerSign = playerSign
    self.m = m
    self.n = n
  
  def __str__(self):
    return str((self.m, self.n))
  
  def __repr__(self):
    return str(self)


class MNK(State):
  def __init__(self, m:int, n:int, k:int, playerSigns:List, emptySign:Any="-") -> None:
      if k > min(m, n):
        raise Exception("k has to be smaller or equal to min (m, n)")
      if emptySign in playerSigns:
        raise Exception("Use different signs for players and default empty cell")
      self.m = m
      self.n = n
      self.k = k
      self.playerSigns = playerSigns[:]
      self.playerSignsRotation = playerSigns[:]
      self.emptySign = emptySign
      self.lastAction = None
      self._isTerminal = False
      self.remainingMoves = m*n
      self.reward = tuple([0 for i in range(len(playerSigns))]) # draw by default
      self.board = [[emptySign for j in range(n)] for i in range(m)]
  
  def getBoard(self)->None: 
    return deepcopy(self.board)

  def getPlayersSigns(self)->List[Any]: 
    return self.playerSigns

  def getCurrentPlayerSign(self)->Any: 
    return self.playerSignsRotation[0]

  def getActions(self)->List[MNKAction]:
    curPlayer = self.getCurrentPlayerSign()
    return [MNKAction(curPlayer, i, j) for i in range(self.m) for j in range(self.n) if self.board[i][j] == self.emptySign]

  def takeAction(self, action:MNKAction)->'State':
    if self.isTerminal():
      raise Exception("Cannot take actions in a terminal state.")
    if (
        action.m<0 or action.m>=self.m or
        action.n<0 or action.n>=self.n or
        action.playerSign!=self.getCurrentPlayerSign() or
        self.board[action.m][action.n] != self.emptySign
        ):
      raise Exception("Illegal Action.")
    stateCopy = deepcopy(self)
    stateCopy.board[action.m][action.n] = action.playerSign
    ## Rotate the Players Sign
    stateCopy.playerSignsRotation.pop(0)
    stateCopy.playerSignsRotation.append(action.playerSign)
    ## End Rotate the Players Sign
    stateCopy.remainingMoves-=1
    stateCopy.lastAction = action
    return stateCopy
  
  def isTerminal(self)->bool:
    if self._isTerminal:
      return True
    if not self.lastAction: # No action taken yet
      return False
    # No moves remaning-> any winner?
    if self.remainingMoves<=0:
      self._isTerminal = True
      self.checkWinner()
      return True
    else:
      # MOves remaining -> any winner?
      return self.checkWinner() 
  
  def checkWinner(self)->bool:
    def encodeReward()->None:
      reward = []
      for sign in self.playerSigns:
        if sign == self.lastAction.playerSign:
          reward.append(1)
        else:
          reward.append(0)
      self.reward = tuple(reward)
    
    lastAction = self.lastAction
    lastPlayerSign = lastAction.playerSign
    ### Check if there's a winner
    leftMost = max(0, lastAction.n-(self.k-1))
    rightMost = min(self.n-1, lastAction.n+(self.k-1))
    topMost = max(0, lastAction.m-(self.k-1))
    bottomMost = min(self.m-1, lastAction.m+(self.k-1))

    #### Check row ####
    l, r  = leftMost, rightMost
    runningK = 0
    while l<=r:
      if self.board[lastAction.m][l]==lastPlayerSign:
        runningK+=1
      else:
        runningK = 0
      l+=1
    if runningK==self.k: 
      encodeReward()
      self._isTerminal = True
      return True
    #### Check col ####
    top, bottom = topMost, bottomMost
    runningK = 0
    while top<=bottom:
      if self.board[top][lastAction.n]==lastPlayerSign:
        runningK+=1
      else:
        runningK = 0
      top+=1
    if runningK==self.k: 
      encodeReward() 
      self._isTerminal = True
      return True
    #### Check Diag1 ####
    top, bottom, l, r = topMost, bottomMost, leftMost, rightMost
    runningK = 0
    while top<=bottom and l<=r:
      if self.board[top][l]==lastPlayerSign:
        runningK+=1
      else:
        runningK = 0
      top+=1
      l+=1
    if runningK==self.k: 
      encodeReward()
      self._isTerminal = True
      return True
    #### Check Diag2 ####
    top, bottom, l, r = topMost, bottomMost, leftMost, rightMost
    while top<=bottom and l<=r:
      if self.board[top][r]==lastPlayerSign:
        runningK+=1
      else:
        runningK = 0
      top+=1
      r-=1
    if runningK==self.k: 
      encodeReward()
      self._isTerminal = True
      return True
    return False
  
  def getReward(self)->Tuple:
    return self.reward

  def __str__(self):
    return '\n'.join(map(str, self.board))

def main():
  TTT = MNK(3,3, 3, ["X", "O"])
  agent1 = MCTS(UCB, linearExpansion, randomRollout, sumTuple)
  agent2 = Random()
  agents = [agent1, agent2]
  agentKwargs=[{'maxIteration':100, 'rewardIdx':[0]}, {}]

  winStatistics = gamePlay( rounds=100, initialState=TTT, agentList = agents, 
                            agentSigns = ["X", "O"], agentKwargList=agentKwargs, 
                            printDetails=False)
  print(winStatistics)

if __name__ == "__main__":
    main()