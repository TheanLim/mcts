from mcts.core import Node, Action
from typing import List, Optional, Callable, Union
import math, random

class Selection:
  def __call__(self, node:Node)->Node:
    pass

class UCB(Selection):
  '''
  Given a parent node, returns a child node according to UCB1 quantity.
  utilityIdx: Applicable it the utilities are encoded with multiple elements, each representing different agents' utility
            For example utility =(0,1,1). utilityIdx:=2 means that only utility[utilityIdx] is considered.
  breakTies: Function used to choose an node from multiple equally good node.
  '''
  def __init__( self, 
                utilityIdx:Optional[List[int]]=None,
                explorationConstant:Union[float, int] = math.sqrt(2), 
                breakTies:Callable[[List[Action]],Action]=random.choice
                )->Node:
    self.utilityIdx = utilityIdx
    self.explorationConstant = explorationConstant
    self.breakTies =breakTies
  
  def __call__(self, node:Node)->Node:
    bestUCB, bestChildNodes = float("-inf"), []
    epsilon = 0.00001
    
    # The sequence of action follows the expansion policy used
    for _, child in node.children.items():
      if not child.utilities:
          childUtilities=0
      else:
        childUtilities = sum([child.utilities[idx] for idx in self.utilityIdx]) if self.utilityIdx else sum(child.utilities)
      
      childExpectedUtility = childUtilities / (child.numVisits+epsilon)
      ucb = childExpectedUtility + self.explorationConstant * math.sqrt(math.log(node.numVisits)/(child.numVisits+epsilon))
      if ucb>bestUCB:
        bestChildNodes = [child]
        bestUCB = ucb
      elif ucb==bestUCB:
        bestChildNodes.append(child)
    
    return self.breakTies(bestChildNodes)