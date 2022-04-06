from mcts.core import State, Action

from typing import List, Any, Tuple, Any
from copy import deepcopy

'''
An MNK Action.
An action is characterize by placing a player's sign (such as "X")
at (m,n) coordinate of the board.
'''
class MNKAction(Action):
  def __init__(self, playerSign:Any, m:int, n:int):
    self.playerSign = playerSign
    self.m = m
    self.n = n
  
  def __str__(self):
    return str((self.m, self.n))
  
  def __repr__(self):
    return str(self)

'''
An MNK Game State.
The Tic-Tac-Toe game is an example of m=n=k=3
'''
class MNK(State):
  def __init__(self, m:int, n:int, k:int, playerSigns:List, emptySign:Any="-") -> None:
    '''
    m: number of rows of the board
    n: number of cols of the board
    k: number of connected sign (such as "X") to win. k has to be smaller or equal to min (m, n)"
    playerSigns: a list of signs, each representing a player. Ex: ["X","O"]
    emptySign: represents that a particular cell is empty, and available for taking action. 
                emptySign cannot be the same as any of the playerSigns
    '''
    if k > min(m, n):
      raise Exception("k has to be smaller or equal to min (m, n)")
    if emptySign in playerSigns:
      raise Exception("Use different signs for players and default empty cell")
    self.m = m
    self.n = n
    self.k = k
    self.playerSigns = playerSigns[:]
    # The first element self.playerSignsRotation represents the currentPlayer
    # self.playerSignsRotation is a circular queue: the player that took an action
    # is placed as the end of the queue.
    self.playerSignsRotation = playerSigns[:]
    self.emptySign = emptySign
    self.lastAction = None
    # Whether it is a terminal state. This is stored to prevent recomputation.
    self._isTerminal = False
    self.remainingMoves = m*n
    # Store rewards in the form of tuple(rewardPlayer1, rewardPlayer2,...)
    self.reward = tuple([0 for i in range(len(playerSigns))]) # draw by default
    # Board of m*n filled with emptySign
    self.board = [[emptySign for j in range(n)] for i in range(m)]
  
  def getBoard(self)->None: 
    '''
    Returns a deep-copied board to prevent accidental modification
    '''
    return deepcopy(self.board)

  def getPlayersSigns(self)->List[Any]: 
    '''
    Returns a list of player signs
    '''
    return self.playerSigns

  def getCurrentPlayerSign(self)->Any:
    '''
    The first element self.playerSignsRotation represents the currentPlayer
    ''' 
    return self.playerSignsRotation[0]

  def getActions(self)->List[MNKAction]:
    '''
    Returns available actions to take.
    Returns all the cells with emptySign
    '''
    curPlayer = self.getCurrentPlayerSign()
    return [MNKAction(curPlayer, i, j) for i in range(self.m) for j in range(self.n) if self.board[i][j] == self.emptySign]

  def takeAction(self, action:MNKAction, preserveState:bool=True)->'State':
    '''
    Take an action and returns the resulting state.
    If preserveState, The original state is NOT altered.
    Only allows to take action on/if:
    a) Non terminal state
    b) Empty Cell
    c) It is the current player's turn
    After taking an action, the playerSignsRotation is rotated, 
    remainingMoves, and lastAction are updated accordingly.
    '''
    if self.isTerminal():
      raise Exception("Cannot take actions in a terminal state.")
    if (
        action.m<0 or action.m>=self.m or
        action.n<0 or action.n>=self.n or
        action.playerSign!=self.getCurrentPlayerSign() or
        self.board[action.m][action.n] != self.emptySign
        ):
      raise Exception("Illegal Action.")
    stateCopy = deepcopy(self) if preserveState else self
    stateCopy.board[action.m][action.n] = action.playerSign
    ## Rotate the Players Sign
    stateCopy.playerSignsRotation.pop(0)
    stateCopy.playerSignsRotation.append(action.playerSign)
    ## End Rotate the Players Sign
    stateCopy.remainingMoves-=1
    stateCopy.lastAction = action
    return stateCopy
  
  def isTerminal(self)->bool:
    '''
    Whether it is a terminal state.
    Returns True if there's a winner or there's no remaining moves
    '''
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
    '''
    Whether there's a winner.
    There's a winner if a player could connect self.k signs
    in a row, col, or diagonally.
    Updates self.reward, and self._isTerminal if needed.
    '''
    def encodeReward()->None:
      '''
      Encodes reward properly
      (-1,-1,1,-1) means the third player wins and get reward of 1
      and the other lose (-1 reward)
      '''
      reward = []
      for sign in self.playerSigns:
        if sign == self.lastAction.playerSign:
          reward.append(1)
        else:
          reward.append(-1)
      self.reward = tuple(reward)
    
    lastAction = self.lastAction
    lastPlayerSign = lastAction.playerSign
    ######## Check if there's a winner ########
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
    minDist = min(lastAction.m-topMost, lastAction.n-leftMost)
    top, l = lastAction.m-minDist, lastAction.n-minDist
    bottom, r = bottomMost, rightMost
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
    minDist = min(lastAction.m-topMost, rightMost-lastAction.n)
    top, r = lastAction.m-minDist, lastAction.n+minDist
    bottom, l = bottomMost, leftMost
    runningK = 0
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
    '''
    Get the reward from this game state.
    The defult reward is a tuple of zeroes
    '''
    # Reward comes from winning.
    self.checkWinner()
    return self.reward

  def __str__(self)->str:
    '''
    Prints the 2D board properly so that each
    row is separated (newline) from each other
    '''
    return '\n'.join(map(lambda x: ' '.join(map(str, x)), self.board))