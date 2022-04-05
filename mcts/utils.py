from mcts.core import State, Search, Action
from typing import Tuple, List, Optional, Dict, Any
from tqdm import tqdm
import random

def sumTuple(a:Tuple, b:Tuple)->Tuple:
  return tuple(map(sum,zip(a,b)))

class Random(Search):
  def search(self, state: State)->Action:
    return random.choice(state.getActions())

def gamePlay( rounds:int, 
              initialState:State, 
              agentList:List[Search], 
              agentSigns:Optional[List[Any]]=None,
              agentKwargList:Optional[List[Dict]]=None,
              printDetails:bool=False, 
              )->Optional[Dict[Any, int]]:
  
  maxIterPerRound = len(initialState.getActions())
  numPlayers = len(agentList)
  if not agentKwargList: 
    agentKwargList = [{} for _ in range(numPlayers)]
  else:
    # Check if Kwarg and the agentList have the same length
    if len(agentList)!=len(agentKwargList):
      raise Exception("Agents and Kwarg Lists have different length.")

  if agentSigns:
    win = {}
    for sign in agentSigns:
      win[sign]=0

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

      if(state.isTerminal()):
        if state.reward:
          if agentSigns: win[state.lastAction.playerSign]+=1
          if printDetails: print("Winner is", state.lastAction.playerSign)
        else:
          if printDetails: print("Draw")
        break
  if printDetails and agentSigns: print(win)
  return win