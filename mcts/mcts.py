from mcts.core import Search, Node, State, Action
from typing import Union, Callable, Optional, Any, List
import math
from copy import deepcopy
import time
import random

class MCTS(Search):
  def __init__(self, 
               selectionPolicy:Callable, 
               expansionPolicy:Callable[[State], List[Action]], 
               rollOutPolicy:Callable[[State],Any],  
               rewardSumFunc:Callable[[Any, Any], Any]=sum, 
               explorationConstant:Union[float, int] = math.sqrt(2), 
               simPerIter:int=1):
    self.explorationConstant = explorationConstant
    self.selectionPolicy = selectionPolicy
    self.expansionPolicy = expansionPolicy # function that returns a seq of actions
    self.rewardSumFunc = rewardSumFunc
    self.rollOutPolicy = rollOutPolicy
    self.simPerIter = simPerIter
  
  def search(self, 
             state:State, 
             maxIteration:int=1000,
             maxTimeSec:int=1000,
             rewardIdx:Optional[List[int]]=None,
             breakTies:Callable[[List[Action]],Action]=random.choice
             )->Action:
    self.root = Node(state, None)
    self.rewardIdx = rewardIdx
    iterCnt = 0
    now = time.time()
    timePassed, timeMax = now, now+maxTimeSec
    while iterCnt<maxIteration and timePassed<timeMax:
      self.oneIteration()
      iterCnt+=1
      timePassed = time.time()

    bestExpectedRewards, bestActions = float("-inf"), []
    epsilon = 0.00001
    # The sequence of action follows the expansion policy used
    for action, child in self.root.children.items():
      if not child.rewards:
        childRewards=0
      else:
        childRewards = sum([child.rewards[idx] for idx in self.rewardIdx]) if rewardIdx else child.rewards

      expectedRewards = childRewards/(child.numVisits+epsilon)
      if expectedRewards>bestExpectedRewards:
        bestActions = [action]
        bestExpectedRewards = expectedRewards
      elif expectedRewards==bestExpectedRewards:
        bestActions.append(action)
      #print("Action: ", action, "Child: ", child, " expectedRewards: ", expectedRewards, "Bestsofar:", bestExpectedRewards)
    return breakTies(bestActions)
  
  def oneIteration (self)->None:
    node = self.selection()
    # If the node was visited, and expandable (not terminal)
    if node.numVisits>0 and not node.state.isTerminal():
      node = self.expansion(node)
    for i in range(self.simPerIter):
      reward = self.simulation(node)
      self.backpropagation(node, reward, self.rewardSumFunc)
  
  def selection(self)->Node:
    # Select a leaf node starting from the root node
    node = self.root
    while not node.isLeaf():
      node = self.selectionPolicy(node, self.explorationConstant, self.rewardIdx)
    return node
  
  def expansion(self, node:Node)->Node:
    # Fully expand the tree ahead of time
    actions = self.expansionPolicy(node.state)
    for action in actions:
      # Add a new state to the tree
      stateAfterAction = node.state.takeAction(action)
      newNode = Node(stateAfterAction, node)
      node.children[action] = newNode
    # Choose the firstAction newNode to return
    return node.children[actions[0]]
  
  def simulation(self, node:Node)->Any:
    '''
    Returns the rewards received from this simulation
    '''
    return self.rollOutPolicy(node.state)
  
  def backpropagation(self, node:Node, reward:Any, rewardSumFunc:Callable=sum)->None:
    while node:
      node.numVisits+=1
      if node.rewards:
        node.rewards = rewardSumFunc(node.rewards,reward)
      else:
        node.rewards = reward
      node = node.parent