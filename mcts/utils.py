from mcts.core import State, Search, Action
from typing import Tuple, List, Optional, Dict, Any, Callable
from tqdm import tqdm
from copy import deepcopy
import random

class Random(Search):
  '''
  A Random agents. It randomly returns an action given a state.
  '''
  def search(self, state: State)->Action:
    return random.choice(state.getActions())

class Schedule:
  '''
  Returns a schedule of values
  '''
  def __init__(self, initialValue, step, maxBound:int=10000, minBound:int=0):
    self.iteration = 0
    self.curVal = initialValue
    self.step = step
    self.maxBound = maxBound
    self.minBound = minBound
  def __call__(self):
    self.iteration+=1
    curVal = self.curVal + self.step
    self.curVal = max(self.minBound, curVal)
    self.curVal = min(self.maxBound, self.curVal)
    return self.curVal

def sumTuple(a:Tuple, b:Tuple)->Tuple:
  '''
  Sum up each element of two tuples.
  Useful for cases where rewards are encoded in tuples.
  Ex:
  a = (1,1,1)
  b = (2,2,2)
  sumTuple(a,b) gives (3,3,3)
  '''
  return tuple(map(sum,zip(a,b)))

def gamePlay( rounds:int, 
              initialState:State, 
              agentList:List[Search], 
              agentKwargList:Optional[List[Dict]]=None,
              rewardSumFunc:Callable=sum,
              printDetails:bool=False, 
              )->Any:
  '''
  Play/simulate a game for one/multiple rounds
  Args:
    rounds: number of simulation
    initialState: the initial game state. Each round starts with the same initialState
    agentList: list of agents. The agent should have a `search(state)` method (inherits from the class `Search`.)
    agentKwargList: kwargs to be passed to `agent.search()`. Optional.
    rewardSumFunc: function used to sum two rewards.
    printDetails: whether to print the `state` or not.
  '''
  maxIterPerRound = len(initialState.getActions())
  numPlayers = len(agentList)
  if not agentKwargList: 
    agentKwargList = [{} for _ in range(numPlayers)]
  else:
    # Check if Kwarg and the agentList have the same length
    if len(agentList)!=len(agentKwargList):
      raise Exception("Agents and Kwarg Lists have different length.")

  rewards = None
  initialKwargs = deepcopy(agentKwargList)
  for _ in tqdm(range(rounds)):
    state = initialState
    agentKwargList = deepcopy(initialKwargs)
    if printDetails: print(state)

    for i in range(maxIterPerRound):
      agent = agentList[i%numPlayers]
      kwargs = agentKwargList[i%numPlayers]
      action = agent.search(state,**kwargs)
      state = state.takeAction(action)

      if printDetails:
        print("\n")
        print(state)

      # Sum rewards
      if state.getReward():
        rewards = rewardSumFunc(rewards, state.getReward()) if rewards else state.getReward()
      
      if(state.isTerminal()): 
        break
  return rewards