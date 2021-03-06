from mcts import MCTS, UCB, linearExpansion, randomRollout
from mcts.applications import MNK
from mcts.utils import sumTuple, Random, gamePlay, Schedule

def main():
  # Create a Tic Tac Toe Game State
  TTT = MNK(3,3, 3, ["O", "X"])
  # Initialize Agents: MCTS agent and a Random agent
  agent1 = Random() # Randomly choose an action. It is "O"
  utilIdx = [1]
  agent2 = MCTS(UCB(utilIdx), linearExpansion, randomRollout, sumTuple, utilityIdx=utilIdx) # It is "X"
  agents = [agent1, agent2]
  # Specify kwarg for MCTS agent.search()
  # Schedules the maximumIteration per search, and numSimulatinon per iteration
  # Setting schedules up such that more simulations are done early in the game, and
  # counter that time comsumption with lower maxIteration
  # Can use (lambda: constant) for constant `maxIteration` / `simPerIter`
  iterSchedule = Schedule(5,10, maxBound=25)
  simPerIterSchedule = Schedule(100, -30, minBound=1)
  agentKwargs=[{}, {'maxIteration':iterSchedule, 'simPerIter':simPerIterSchedule}]

  winStatistics = gamePlay( rounds=1, initialState=TTT, agentList = agents, 
                            agentKwargList=agentKwargs, 
                            utilitySumFunc=sumTuple, printDetails=True)
  print(winStatistics)
  # Run 1000 rounds
  # 891 Win, 0 Lose using (maxIter=25, simPerIter=100), took 11 mins
  # 848 Win, 0 Lose using Schedule, took 4 mins

if __name__ == "__main__":
    main()