from mcts import MCTS, UCB, linearExpansion, randomRollout
from mcts.applications import MNK
from mcts.utils import sumTuple, Random, gamePlay

def main():
  # Create a Tic Tac Toe Game State
  TTT = MNK(3,3, 3, ["O", "X"])
  # Initialize Agents: MCTS agent and a Random agent
  agent1 = Random() # Randomly choose an action. It is "O"
  agent2 = MCTS(UCB, linearExpansion, randomRollout, sumTuple, simPerIter=100) # It is "X"
  agents = [agent1, agent2]
  # Specify kwarg for MCTS agent.search()
  agentKwargs=[{}, {'maxIteration':25, 'rewardIdx':[1]}]

  winStatistics = gamePlay( rounds=1, initialState=TTT, agentList = agents, 
                            agentKwargList=agentKwargs, 
                            rewardSumFunc=sumTuple, printDetails=True)
  print(winStatistics)

if __name__ == "__main__":
    main()