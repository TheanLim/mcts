from mcts.core import State, Search, Action
from typing import Tuple, List, Optional, Dict, Any, Callable
from tqdm import tqdm
import random

class Random(Search):
  '''
  A Random agents. It randomly returns an action given a state.
  '''
  def search(self, state: State)->Action:
    return random.choice(state.getActions())

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
              agentSigns:Optional[List[Any]]=None,
              agentKwargList:Optional[List[Dict]]=None,
              rewardSumFunc:Callable=sum,
              printDetails:bool=False, 
              )->Dict[Any, Any]:
  '''
  Play/simulate a game for one/multiple rounds
  Args:
    rounds: number of simulation
    initialState: the initial game state. Each round starts with the same initialState
    agentList: list of agents. The agent should have a `search(state)` method (inherits from the class `Search`.)
    agentSigns: List of agent signs. For example: ["X", "O"] means that the first player is represented as "X". Optional.
                By default, the agentSigns are "0", "1",...
    agentKwargList: kwargs to be passed to `agent.search()`. Optional.
    rewardSumFunc: function used to sum two rewards.
    printDetails: whether to print the `state` or not.
  Returns:
    A dictionary of {agentSign: cumulativeRewards}. 
      Assume the current reward is applicable to the agent that's taking action only.
  '''
  maxIterPerRound = len(initialState.getActions())
  numPlayers = len(agentList)
  if not agentKwargList: 
    agentKwargList = [{} for _ in range(numPlayers)]
  else:
    # Check if Kwarg and the agentList have the same length
    if len(agentList)!=len(agentKwargList):
      raise Exception("Agents and Kwarg Lists have different length.")

  rewards = {}
  if not agentSigns: agentSigns = [str(i) for i in range(numPlayers)] # default signs
  for sign in agentSigns: rewards[sign]=None

  for _ in tqdm(range(rounds)):
    state = initialState
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
        agentSign = agentSigns[i%numPlayers]
        if rewards[agentSign]:
          rewards[agentSign] = rewardSumFunc(rewards[agentSign], state.getReward())
        else:
          rewards[agentSign]=state.getReward()
      
      if(state.isTerminal()): 
        break
  return rewards