# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        #seful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        

        currFood = currentGameState.getFood().asList()

       
       
	addScore = 0.0
        for i in newGhostStates:
	    if i.scaredTimer == 0 and i.getPosition() == newPos:
		addScore = float('-inf')
			
	    if i.scaredTimer >= 1 and i.getPosition() == newPos:
		addScore += 2.0
	
	    if Directions.STOP in action:
		addScore = float('-inf')  	
	
	
	minDist = float("inf")
       

	for i in currFood:
	    minDist = min(minDist,manhattanDistance(i,newPos))
            
	if minDist!= 0:
	    foodScore = 1.0/minDist
	else:
	    foodScore = 1.0
	
	
	addScore += foodScore
	return addScore
   

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

	
    def minMaxValue(self,gameState, index, depth):
		
	  if index >= gameState.getNumAgents():
            	index = 0
            	depth += 1

          if depth == self.depth:
            	return self.evaluationFunction(gameState)

          if index == self.index:
            	return self.maxValue(gameState, index, depth)
          else:
            	return self.minValue(gameState, index, depth)

	
    def minValue(self,gameState, index, depth):
		
	  min_val = float('inf')
	
	  legalActions = gameState.getLegalActions(index)
	  
	  if gameState.isWin() or gameState.isLose():
        	return self.evaluationFunction(gameState)
		
	 
	  for l in legalActions:
		
		successor = gameState.generateSuccessor(index,l)
		val = self.minMaxValue(successor,index+1, depth)
			
	
		if val < min_val:
  			min_val = val
  	
		
	  return min_val

  
    def maxValue(self,gameState, index, depth):

	  
	  if gameState.isWin() or gameState.isLose():
		return self.evaluationFunction(gameState)

	  max_val = float("-inf")
	  rootValue = ""

	  legalActions = gameState.getLegalActions(index)

	  for actions in legalActions:
		successor = gameState.generateSuccessor(index,actions)
		succValue = self.minMaxValue(successor,index+1,depth)

		if succValue > max_val:
		  max_val = succValue
		  rootValue = actions

	  if depth > 0:
		return max_val
	  else:
                return rootValue
	
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(index):
            Returns a list of legal actions for an agent
            index=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(index, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
	initialindex = 0
	initialDepth = 0
	return self.minMaxValue(gameState,initialindex,initialDepth)




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphaBetaValue(self,gameState, index, depth,alpha,beta):
		
	  if index >= gameState.getNumAgents():
            	index = 0
            	depth += 1

          if depth == self.depth:
            	return self.evaluationFunction(gameState)

          if index == self.index:
            	return self.alphaValue(gameState, index, depth,alpha,beta)
          else:
            	return self.betaValue(gameState, index, depth,alpha,beta)

    
    def betaValue(self,gameState, index, depth,alpha,beta):
		
	  min_val = float('inf')
	  
	  legalActions = gameState.getLegalActions(index)
	  
	  if gameState.isWin() or gameState.isLose():
        	return self.evaluationFunction(gameState)
		
	 
	  for l in legalActions:
		
		successor = gameState.generateSuccessor(index,l)
		val = self.alphaBetaValue(successor,index+1, depth,alpha,beta)
		if l  == Directions.STOP:
          		continue		
	
		if val < min_val:
  			min_val = val
  	
		if min_val< alpha:
			return min_val
		
		beta = min(min_val, beta)
		
	  return min_val

    def alphaValue(self,gameState, index, depth,alpha,beta):

	  
	  if gameState.isWin() or gameState.isLose():
		return self.evaluationFunction(gameState)

	  max_val = float("-inf")
	  rootValue= ""

	  legalActions = gameState.getLegalActions(index)

	  for actions in legalActions:
		if actions == Directions.STOP:
         	  continue

		successor = gameState.generateSuccessor(index,actions)
		succValue = self.alphaBetaValue(successor,index+1,depth,alpha,beta)

		if succValue > max_val:
		  max_val = succValue
		  rootValue = actions

		if max_val > beta:
		  return max_val

		alpha = max(alpha, max_val)

	  if depth > 0:
		return max_val
	  else:
                return rootValue

    def getAction(self, gameState):
	  index = 0
	  Depth = 0
	  alpha = float('-inf')
	  beta = float('inf')
	  return self.alphaBetaValue(gameState,index, Depth, alpha,beta)
	


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """


	
    def expectiMaxValue(self,gameState, index, depth):
		
	  if index >= gameState.getNumAgents():
            	index = 0
            	depth += 1

          if depth == self.depth:
            	return self.evaluationFunction(gameState)

          if index == self.index:
            	return self.max_Value(gameState, index, depth)
          else:
            	return self.expectValue(gameState, index, depth)

    def max_Value(self,gameState, index, depth):

	  
	  if gameState.isWin() or gameState.isLose():
		return self.evaluationFunction(gameState)

	  max_val = float("-inf")
	  rootValue = " "

	  legalActions = gameState.getLegalActions(index)

	  for actions in legalActions:

		successor = gameState.generateSuccessor(index,actions)
		succValue = self.expectiMaxValue(successor,index+1,depth)

		if succValue > max_val:
		  max_val = succValue
		  rootValue = actions

	  if depth > 0:
		return max_val
	  else:
                return rootValue

    def expectValue(self,gameState, index, depth):
		
	  min_val = 0
	
	  legalActions = gameState.getLegalActions(index)
	  probability = 0.0
          if len(legalActions)> 0 :
	  	probability = 1.0 / len(legalActions)
		

	  if gameState.isWin() or gameState.isLose():
        	return self.evaluationFunction(gameState)
		
	 
	  for l in legalActions:
		
		successor = gameState.generateSuccessor(index,l)
		val = self.expectiMaxValue(successor,index+1, depth)
			
	
		min_val += val*probability
		
	  return min_val


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        initialindex = 0
	initialDepth = 0
	return self.expectiMaxValue(gameState,initialindex,initialDepth)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
     
        
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood().asList()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [scaredTime.scaredTimer for scaredTime in currGhostStates]
    currCapsules = currentGameState.getCapsules()   
       
    addScore = 0.0
    scaredGhosts = 0
    GhostDist = float("inf")
    for i in currGhostStates:
	if i.scaredTimer == 0 and i.getPosition() == currPos:
		addScore = float('-inf')
			
	if i.scaredTimer >= 1 and i.getPosition() == currPos:
		addScore += 2.0
		scaredGhosts +=1
	
	GhostDist = min(manhattanDistance(currPos, i.getPosition()),GhostDist)
	
	    	
    score = currentGameState.getScore()
	
    minDist = float("inf")
    a = 1.0
    for i in currFood:
	minDist = min(minDist,manhattanDistance(i,currPos))
        
	if minDist!= 0:
	        a = 1.0/minDist
	
		
    addScore += a
    minCapDist = float ("inf")
    b = 1.0
    for i in currCapsules:
	minDist = min(minCapDist,manhattanDistance(i,currPos))
            
	if minCapDist!= 0:
	        b = 1.0/minDist

    addScore += b
    GhostDist = -5.0 * (1.0 /1.0 + GhostDist)

    return addScore + score + scaredGhosts + GhostDist
   


# Abbreviation
better = betterEvaluationFunction



