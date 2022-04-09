from mcts.core import Search, Node, State, Action
from typing import Union, Callable, Optional, Any, List
import math
from copy import deepcopy
import time
import random

'''
A Monte Carlo Tree Search Object.
It samples the search space and expands the search tree according to promising nodes.
Less promising nodes are visited from time to time.
'''
class MCTS(Search):
  def __init__(self, 
               selectionPolicy:Callable, 
               expansionPolicy:Callable[[State], List[Action]], 
               rollOutPolicy:Callable[[State],Any],  
               utilitySumFunc:Callable[[Any, Any], Any]=sum, 
               explorationConstant:Union[float, int] = math.sqrt(2)
               ):
    '''
    selectionPolicy: Given the current node, which child node should be selected to traverse to?
    expansionPolicy: Given the current (leaf) node, which child node should be expanded (grown) first?
    rollOutPolicy: Given the current node/state, how should a playout be completed? What's the sequence of action to take?
    utilitySumFunc: function used to sum two rewards. The default is sum()
    '''
    self.explorationConstant = explorationConstant
    self.selectionPolicy = selectionPolicy
    self.expansionPolicy = expansionPolicy # function that returns a seq of actions
    self.utilitySumFunc = utilitySumFunc
    self.rollOutPolicy = rollOutPolicy
  
  def search(self, 
             state:State, 
             maxIteration:Callable=(lambda: 1000),
             maxTimeSec:Callable=(lambda: 1000),
             simPerIter:Callable=(lambda:1),
             utilityIdx:Optional[List[int]]=None,
             breakTies:Callable[[List[Action]],Action]=random.choice
             )->Action:
    '''
    Search for the best action to take given a state.
    The search is stopped when the maxIteration or maxTimeSec is hitted. 
    Args:
      simPerIter: number of simulation(rollouts) from the chosen node.
      utilityIdx: Applicable if the utilities are encoded with multiple elements, each representing different agents' utility
                  For example utility =(0,1,1). utilityIdx:=2 means that only utility[utilityIdx] is considered.
      breakTies: Function used to choose an node from multiple equally good node.
    '''
    self.root = Node(state, None)
    self.simPerIter = simPerIter()
    self.utilityIdx = utilityIdx
    iterCnt = 0
    now = time.time()
    timePassed, timeMax = now, now+maxTimeSec()
    maxIter = maxIteration()
    #print(maxIter, self.simPerIter)
    # Loop while have remaining iterations or time
    while iterCnt<maxIter and timePassed<timeMax:
      self.oneIteration()
      iterCnt+=1
      timePassed = time.time()

    ########## Select the best action based on its expected utilities ##########
    bestExpectedUtilities, bestActions = float("-inf"), []
    epsilon = 0.00001 # Prevent numeric overflow
    # The sequence of action follows the expansion policy used
    for action, child in self.root.children.items():
      if not child.utilities:
        childUtilities=0
      else:
        childUtilities = sum([child.utilities[idx] for idx in self.utilityIdx]) if utilityIdx else sum(child.utilities)

      expectedUtilities = childUtilities/(child.numVisits+epsilon)
      if expectedUtilities>bestExpectedUtilities:
        bestActions = [action]
        bestExpectedUtilities = expectedUtilities
      elif expectedUtilities==bestExpectedUtilities:
        bestActions.append(action)
    return breakTies(bestActions)
  
  def oneIteration (self)->None:
    '''
    Perform one iteration of leaf node selection, expansion (if applicable), simulation, and backpropagation.
    Only expand a node if it was visited before. Otherwise, perform simulation on the node that wasn't visited.
    Simulation is performed `self.simPerIter` times
    '''
    node = self.selection()
    # If the node was visited, and expandable (not terminal)
    if node.numVisits>0 and not node.state.isTerminal():
      node = self.expansion(node)
    for i in range(self.simPerIter):
      utility = self.simulation(node)
      self.backpropagation(node, utility, self.utilitySumFunc)
  
  def selection(self)->Node:
    '''
    Select and returns a leaf node.
    Traverse from the root node to the leaf node, following self.selectionPolicy
    '''
    # Select a leaf node starting from the root node
    node = self.root
    while not node.isLeaf():
      node = self.selectionPolicy(node, self.explorationConstant, self.utilityIdx)
    return node
  
  def expansion(self, node:Node)->Node:
    '''
    Fully expands a node and return one of its child node.
    Expands a node following self.expansionPolicy. 
    Returns the first children node.
    '''
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
  
  def backpropagation(self, node:Node, utility:Any, utilitySumFunc:Callable=sum)->None:
    '''
    BackPropagate results to parent nodes.
    Update a node's Utility and Number of being visited.

    utilitySumFunc: function used to sum two utilities. The default is sum()
    '''
    while node:
      node.numVisits+=1
      if node.utilities:
        node.utilities = utilitySumFunc(node.utilities,utility)
      else:
        node.utilities = utility
      node = node.parent