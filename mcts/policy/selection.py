from mcts.core import Node, Action
from typing import List, Optional, Callable, Union
import math, random

def UCB(node:Node, 
        explorationConstant:Union[float, int] = math.sqrt(2), 
        rewardIdx:Optional[List[int]]=None,
        breakTies:Callable[[List[Action]],Action]=random.choice
        )->Node:
  '''
  Given a parent node, returns a child node according to UCB1 quantity.
  rewardIdx: Applicable it the rewards are encoded with multiple elements, each representing different agents' reward
            For example reward =(0,1,1). rewardIdx:=2 means that only reward[rewardIdx] is considered.
  breakTies: Function used to choose an node from multiple equally good node.
  '''
  bestUCB, bestActions = float("-inf"), []
  epsilon = 0.00001
  
  # The sequence of action follows the expansion policy used
  for action, child in node.children.items():
    if not child.rewards:
        childRewards=0
    else:
      childRewards = sum([child.rewards[idx] for idx in rewardIdx]) if rewardIdx else sum(child.rewards)
    
    childExpectedReward = childRewards / (child.numVisits+epsilon)
    ucb = childExpectedReward + explorationConstant * math.sqrt(math.log(node.numVisits)/(child.numVisits+epsilon))
    if ucb>bestUCB:
      bestActions = [child]
      bestUCB = ucb
    elif ucb==bestUCB:
      bestActions.append(child)
  
  return breakTies(bestActions)