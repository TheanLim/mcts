from mcts.core import Node, Action
from typing import List, Optional, Callable, Union
import math, random

def UCB(node:Node, 
        explorationConstant:Union[float, int] = math.sqrt(2), 
        rewardIdx:Optional[List[int]]=None,
        breakTies:Callable[[List[Action]],Action]=random.choice
        )->Node:
  
  bestUCB, bestActions = float("-inf"), []
  epsilon = 0.00001
  
  # The sequence of action follows the expansion policy used
  for action, child in node.children.items():
    if not child.rewards:
        childRewards=0
    else:
      childRewards = sum([child.rewards[idx] for idx in rewardIdx]) if rewardIdx else child.rewards
    
    childExpectedReward = childRewards / (child.numVisits+epsilon)
    ucb = childExpectedReward + explorationConstant * math.sqrt(math.log(node.numVisits)/(child.numVisits+epsilon))
    if ucb>bestUCB:
      bestActions = [child]
      bestUCB = ucb
    elif ucb==bestUCB:
      bestActions.append(child)
  
  return breakTies(bestActions)