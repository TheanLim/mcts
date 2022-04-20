import time
from collections import defaultdict
from multiprocessing import Process, Manager
from typing import Callable, Dict, Optional, List, Tuple
from mcts.core import Search, State, Action

'''
Minimax Search with AlphaBeta Pruning, and Memoization
'''
class Minimax(Search):
  def __init__( self, 
                depth:int, 
                evaluationFunction:Callable[[State, int], float],
                expansionPolicy:Callable[[State, int, Dict], List[Action]]=lambda state, depth, cache: state.getActions(),
                toCache:bool=False,
                toAlphaBetaPrune:bool=True,
                ):
    '''
    depth: maximum depth that the minimax tree should evaluate. At depth `depth` the `evaluationFunction` is called to evaluate the state.
    evaluationFunction: evaluate a given state and returns the state's value. The second argument is `depth` which could be used whether 
                        the evaluation function is called from a minimizer or maximizer
    expansionPolicy: returns a list of possible actions given a state. The sequence of actions returned will affect the efficiency of
                     alpha-beta pruning. The second argument is `depth` which could be used whether the evaluation function is called 
                     from a minimizer or maximizer. The third argument is `cache`, which could be used in sequencing actions.
    toCache: whether to cache state values. The cached value might be an approximate only if alpha-beta pruning is used.
    toAlphaBetaPrune: whether to use alpha-beta pruning. Pruning will likely be faster if use together with a cache.
    '''
    self.depth = depth
    self.evaluationFunction = evaluationFunction
    self.expansionPolicy = expansionPolicy
    self.toCache = toCache
    self.cache = defaultdict(lambda:("__eq__", 0)) if toCache else None
    self.toAlphaBetaPrune = toAlphaBetaPrune

  def search(self, state:State, resetCache:bool=True)->Action:
    '''
    If applicable, reset the cache to empty whenever a new search is called.
    This is done to save memory
    '''
    if resetCache and self.toCache: self.cache = defaultdict(lambda:("__eq__", 0))
    if self.toAlphaBetaPrune: rootAlpha, rootBeta = float('-inf'), float('inf')
    if not self.toAlphaBetaPrune: rootAlpha, rootBeta = None, None
    
    values = []

    value = float('-inf')
    actions = self.expansionPolicy(state, 0, self.cache)
    for action in actions:
      tempValue = self.minValue(state.takeAction(action), 1, rootAlpha, rootBeta)
      values.append(tempValue)
      value = max(value, tempValue)
      if rootAlpha: rootAlpha = max(rootAlpha, value)
    
    bestIndices=[index for index in range(len(values)) if values[index] == value]
    return actions[bestIndices[0]] # The first action

  def maxValue(self, state:State, depth:int, alpha:Optional[float], beta:Optional[float])->float:
    if state.isTerminal() or depth==self.depth: 
      return self.evaluationFunction(state, depth)

    if self.toCache:
      alpha, beta, value = self.readCache(state, depth, alpha, beta)
      alphaCopy, betaCopy = alpha, beta
      if value: return value

    value = float("-inf")
    actions = self.expansionPolicy(state, depth, self.cache)
    for action in actions:
      value = max(value, self.minValue(state.takeAction(action), depth+1, alpha, beta))
      if alpha and beta:
        if value >= beta: break
        alpha = max(alpha, value)
    
    # Store or Update a state's value
    if self.toCache: self.storeCache(state, depth, alphaCopy, betaCopy, value)
    return value
      
  def minValue(self, state:State, depth:int, alpha:Optional[float], beta:Optional[float])->float:
    if state.isTerminal() or depth==self.depth: 
      return self.evaluationFunction(state, depth)

    if self.toCache:
      alpha, beta, value = self.readCache(state, depth, alpha, beta)
      alphaCopy, betaCopy = alpha, beta
      if value: return value

    value = float("inf")
    actions = self.expansionPolicy(state, depth, self.cache)
    for action in actions:
      value = min(value, self.maxValue(state.takeAction(action), depth+1, alpha, beta))
      if alpha and beta:
        if value <= alpha: break
        beta = min(beta, value)
    
    # Store or Update a state's value
    if self.toCache: self.storeCache(state, depth, alphaCopy, betaCopy, value)
    return value

  def storeCache(self, state:State, depth:int, alpha:Optional[float], beta:Optional[float], value:float)->None:
    '''
    Store/Update the value of a state into a cache.
    A cache is in a form of dictionary with key of (state, depth).
    If no pruning happened (alpha<value<beta), it means the value of the state is exact (not approximate)
      Thus, we store a tuple of ("__eq__", value). The first element serves as a flag that the `value` is exact
    If value<=alpha, then we know that the actual stateValue is less than or equal the approximated `value`.
      Thus, we store ("__leq__", value)
    If value >= beta, then we know that the actual stateValue is greater than or equal the approximated `value`.
      Thus, we store ("__geq__", value)
    '''
    #depth = 0
    if not alpha or not beta:
      # No pruning happened
      self.cache[(state, depth)] = ("__eq__", value)
      return
    
    # Pruning might happen -> inaccurate value
    if value<=alpha:
      self.cache[(state, depth)] = ("__leq__", value)
    elif alpha<value<beta:
      self.cache[(state, depth)] = ("__eq__", value)
    elif beta<=value:
      self.cache[(state, depth)] = ("__geq__", value)
    else:
      raise Exception("Shouldn't get here.")
  
  def readCache(self, state:State, depth:int, alpha:Optional[float], beta:Optional[float])->Tuple[Optional[float], Optional[float], Optional[float]]:
    '''
    Read values from the cache.
    Returns a tuple of (alpha, beta, value of a state). If the `value` is NOT None, 
    it means that there's no need to further searching down the tree. 
    The returned alpha and beta might be modified to help narrow the search.

    Generally, Minimax with AlphaBeta Pruning fulfills:  
            alpha <= actual state value <= beta.
    So,
    1. If alpha or beta is none -> No pruning -> cached results must be exact state value. No need to search further
    2. If flag = __eq__, returns state value. No need to search further
    3. If flag = __leq__, it means that the actual stateValue is less than the cached value.
        a. cachedValue <= alpha -> actual stateValue <= alpha -> return cachedValue and no need to search further
        b. alpha < cachedValue < beta -> shrink the search to [alpha, cachedValue] -> set beta:=cachedValue
        c. beta<=cachedValue -> cachedValue has no help. Proceed to search as usual
    4. Similar thing for flag = __geq__
    '''
    #depth = 0
    if not alpha or not beta:
      if (state, depth) in self.cache: 
        return alpha, beta, self.cache[(state, depth)][1] # second element is the value
    
    # Pruning might happen
    # If alpha and beta are provided, return (possibly updated) alpha, beta, and value (if applicable)
    returnValue = None
    if (state, depth) in self.cache:
      flag, value = self.cache[(state, depth)]
      if flag == "__eq__": returnValue = value
      elif flag == "__leq__":
        if value <= alpha: returnValue = value
        elif alpha<value<beta: beta = value
      elif flag =="__geq__":
        if beta <= value: returnValue =  value
        elif alpha<value<beta: alpha = value
    return alpha, beta, returnValue

'''
Iterative Deepening Search with Minimax
Search while there is time left and the maxDepth has not reached.
'''
class MinimaxIDS(Minimax):
  def __init__(self, 
              time:int,
              maxDepth:int,
              evaluationFunction: Callable[[State, int], float], 
              expansionPolicy: Callable[[State, int, Dict], List[Action]]=lambda state, depth, cache: state.getActions(), 
              toCache: bool=False,
              toAlphaBetaPrune:bool=True,):
    # Start Searching until cutoff depth 1
    super().__init__(1, evaluationFunction, expansionPolicy, toCache, toAlphaBetaPrune)
    self.time = time
    self.maxDepth = maxDepth
  
  def search(self, state: State, resetCache:bool=True)->Action:
    '''
    Search and return for the best action to take given a state.
    To return an action after searching for self.time seconds,
    it spawns a process to IDS for an action, and kills it when time is up.
    The parent and the only child process communicate/share object (action)
    via a Queue(). 
    The latest action returned from the IDS is stored at the
    end of the queue, and the latest action is returned.
    If no action is returned from IDS, hence not stored in the queue,
    the first possible action given the state is returned.

    If applicable, reset the cache to empty whenever a new search is called.
    This is done to save memory
    '''
    if resetCache and self.toCache: self.cache = defaultdict(lambda:("__eq__", 0))

    # Spawn a process to IDS for an action
    # Kill the process when time is up and return the latest action found
    with Manager() as manager:
      # Using a queue to share objects
      q = manager.Queue()
      p = Process(target=self._search, args=(state, q))
      p.start()
      # Usage: join([timeout in seconds])
      p.join(self.time)
      if p.is_alive():
          p.terminate()
          p.join()
      # Get the latest chosen action
      action = None
      while not q.empty(): action = q.get()
    
    # If the search doesn't give any action, choose the first available action as the default
    if not action:
      action = self.expansionPolicy(state, 0, self.cache)[0]
      print("Fail to search for an action - return the first possible action found.")
    #print("Player take", state.getCurrentPlayerSign(), " action ", action)
    return action

  def _search(self, state:State, queueOfActions):
    '''
    Iterative Deepening Search while there is time left
    and depth search deeper.

    queueOfActions: multiprocessing.Manager().Queue()
    It is used to share the action searched to the parent process.
    '''
    # Start Searching until cutoff depth 1
    self.depth = 1
    endTime = time.time() + self.time
    while time.time() < endTime and self.depth<=self.maxDepth:
      action= super().search(state, resetCache=False) # maintain the cache over iterations
      queueOfActions.put(action)
      #print("Finish Depth: ", self.depth, " action: ", action)
      self.depth+=1
    return queueOfActions


def linearExpansion(state:State, depth:int, cache:Dict)->List[Action]:
  '''
  Returns a list of actions in a sequence that are encoded by the state.
  '''
  return state.getActions()

def cacheExpansion(state:State, depth:int, cache:Dict)->List[Action]:
  '''
  Return a list of actions sorted based on the cache.
  The actions is sorted such that it increases the chance of pruning in AlphaBeta Pruning.
  It assumes that if `depth` is an odd number, then this function is called from a minimizer,
  thus actions are sorted ascendingly using the resulting states value as the key.
  The reverse is true for an even number `depth`.
  If the cache is empty (None), returns the list of actions in a sequence as encoded by the state.
  '''
  if not cache: return state.getActions()

  def getActionValue(action):
    '''
    Get a resulting state value after taking an action
    '''
    #flag, value = cache.get((state.takeAction(action), 0), ("__eq_", 0))
    flag, value = cache.get((state.takeAction(action), depth+1), ("__eq_", 0))
    return value
  
  actions = state.getActions()
  if depth%2==0:
    sortedActions = sorted(actions, key=lambda action: getActionValue(action), reverse =True)
  else:
    sortedActions = sorted(actions, key=lambda action: getActionValue(action), reverse=False)
  return sortedActions