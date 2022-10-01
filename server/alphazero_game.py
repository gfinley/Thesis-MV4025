import numpy as np
import map
import unit
import math
import alphazero.Game
import game

# Alpha Star General game implementation for Atlatl's city-inf-5

def strengthUnitFeature(unt, faction):
    if not unt.ineffective and unt.faction == faction:
        return unt.currentStrength/100.0
    else:
        return 0.0
        
def blueUnitFeature(unt):
    return strengthUnitFeature(unt,"blue")
    
def redUnitFeature(unt):
    return strengthUnitFeature(unt,"red")

def canMoveFeature(unt):
    if not unt.ineffective and unt.canMove:
        return 1.0
    else:
        return 0.0
    
# def moverFeatureFactory(moving_unit):
#     def inner(unt):
#         if unt==moving_unit:
#             return 1.0
#         return 0.0
#     return inner

def unitTypeFeatureFactory(type):
    def inner(unt):
        if unt.type == type:
            return 1.0
        return 0.0
    return inner

def terrainFeatureFactory(terrain):
    def inner(hex):
        if hex.terrain == terrain:
            return 1.0
        return 0.0
    return inner 

# def legalMoveFeatureFactory(legal_move_hexes):
#     def inner(hex):
#         if hex.id in legal_move_hexes:
#             return 1.0
#         return 0.0
#     return inner

def constantFeatureFactory(value):
    def inner(hex):
        return value
    return inner

# def legalMoveHexes(self, mover):
#     result = {}
#     if mover:
#         fireTargets = mover.findFireTargets(self.unitData)
#         for unt in fireTargets:
#             result[unt.hex.id] = True
#         moveTargets = mover.findMoveTargets(self.mapData, self.unitData)
#         for hex in moveTargets:
#             result[hex.id] = True
#     return result

def feature(fn, type, mapData, unitData):
    count = 0
    arrayIndex = {}
    inverseIndex = []
    for hexId in mapData.hexIndex:
        arrayIndex[hexId] = count
        inverseIndex.append(hexId)
        count += 1
    dim = mapData.getDimensions()
    mat = np.zeros((dim['height'],dim['width']))
    if type=="hex":
        for hexId in mapData.hexIndex:
            hex = mapData.hexIndex[hexId]
            mat[hex.y_offset, hex.x_offset] = fn(hex)
    else: # type=="unit"
        for unitId in unitData.unitIndex:
            unt = unitData.unitIndex[unitId]
            hex = unt.hex
            if hex:
                mat[hex.y_offset, hex.x_offset] = fn(unt)
    return mat

def stateToNNInput(paramPo, statePo):
    mapData = map.MapData()
    unitData = unit.UnitData()
    map.fromPortable(paramPo['map'], mapData)
    unit.fromPortable(paramPo['units'], unitData, mapData)
    currentPhase = statePo['status']['currentPhase']
    maxPhases = paramPo['score']['maxPhases']
    phase_indicator = 0.9**(maxPhases-currentPhase)

    movingFaction = statePo['status']['onMove']

    if movingFaction=="blue":

        nnInput = np.stack( [
                            feature(blueUnitFeature,"unit",mapData,unitData), 
                            feature(redUnitFeature,"unit",mapData,unitData),
                            feature(canMoveFeature,"unit",mapData,unitData),
                            feature(unitTypeFeatureFactory("infantry"),"unit",mapData,unitData),
                            feature(unitTypeFeatureFactory("mechinf"),"unit",mapData,unitData),
                            feature(unitTypeFeatureFactory("armor"),"unit",mapData,unitData),
                            feature(unitTypeFeatureFactory("artillery"),"unit",mapData,unitData),
                            feature(terrainFeatureFactory("clear"),"hex",mapData,unitData),
                            feature(terrainFeatureFactory("water"),"hex",mapData,unitData),
                            feature(terrainFeatureFactory("rough"),"hex",mapData,unitData),
                            feature(terrainFeatureFactory("urban"),"hex",mapData,unitData),
                            feature(constantFeatureFactory(phase_indicator),"hex",mapData,unitData)
                         ] )

    else:

        nnInput = np.stack( [
                            feature(redUnitFeature,"unit",mapData,unitData), 
                            feature(blueUnitFeature,"unit",mapData,unitData),
                            feature(canMoveFeature,"unit",mapData,unitData),
                            feature(unitTypeFeatureFactory("infantry"),"unit",mapData,unitData),
                            feature(unitTypeFeatureFactory("mechinf"),"unit",mapData,unitData),
                            feature(unitTypeFeatureFactory("armor"),"unit",mapData,unitData),
                            feature(unitTypeFeatureFactory("artillery"),"unit",mapData,unitData),
                            feature(terrainFeatureFactory("clear"),"hex",mapData,unitData),
                            feature(terrainFeatureFactory("water"),"hex",mapData,unitData),
                            feature(terrainFeatureFactory("rough"),"hex",mapData,unitData),
                            feature(terrainFeatureFactory("urban"),"hex",mapData,unitData),
                            feature(constantFeatureFactory(phase_indicator),"hex",mapData,unitData)
                         ] )
    return (movingFaction, nnInput)

# The portable game board will be portable (JSON convertible) Python objects with param and state properties
# The neural network input board format will be a pair with the phasing faction plus a numpy state
class AlphaStarGame(alphazero.Game.Game):

    def __init__(self, game):
        self.game = game

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network, but actually
                        this will be a portable, i.e. JSON compatible, Python object)
        """
        param = game.parameters()
        state = game.initial_state()
        return {"param":param, "state":state}

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions.
        """
        return self.game.mapData.getDimensions()

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions. Will be based on four units per faction.
        """
        return 5*5*6+1 # 5x5 grid, 6 directions, plus end turn (pass)

    def atlatlActionToVectorIndex(self, actionPo, board):
        paramPo = board["param"]
        statePo = board["state"]
        mapData = map.MapData()
        unitData = unit.UnitData()
        map.fromPortable(paramPo["map"], mapData)
        unit.fromPortable(statePo["units"], unitData, mapData)
        # Get hex of actor
        # Get the actor's id
        if actionPo["type"]=="pass":
            return 150
        elif actionPo["type"]=="move":
            actorId = actionPo["mover"]
            targetHexId = actionPo["destination"]
            targetHexObj = mapData.hexIndex[targetHexId]
        else: # "fire"
            actorId = actionPo["source"]
            targetUnitId = actionPo["target"]
            targetHexObj = unitData.unitIndex[targetUnitId].hex
        actorObj = unitData.unitIndex[actorId]
        actorHexObj = actorObj.hex
        row = actorHexObj.x_offset
        col = actorHexObj.y_offset
        direction = map.directionFrom(actorHexObj, targetHexObj)
        index = row*30 + col*6 + direction
        return index

    def vectorIndexActionToAtlatl(self, actionOffset, board):
        paramPo = board["param"]
        statePo = board["state"]
        mapData = map.MapData()
        unitData = unit.UnitData()
        map.fromPortable(paramPo["map"], mapData)
        unit.fromPortable(statePo["units"], unitData, mapData)
        if actionOffset==150:
            return {"type":"pass"}
        row = math.floor(actionOffset/(5*6))
        rest = actionOffset-row*5*6
        col = math.floor(rest/6)
        direction = rest - col*6
        origin = f'hex-{row}-{col}'
        if not origin in unitData.occupancy:
            return None
        actorObj = unitData.occupancy[origin][0]
        originObj = mapData.hexIndex[origin]
        targetHex = map.getNeighborHex(originObj, mapData, direction)
        if not targetHex:
            return None
        targetHexId = targetHex.id
        targetOccupants = unitData.occupancy.get(targetHexId,[])
        if targetOccupants: # shoot
            targetId = targetOccupants[0].uniqueId
            return {"type":"fire","source":actorObj.uniqueId,"target":targetId}
        else: # move
            return {"type":"move","mover":actorObj.uniqueId,"destination":targetHexId}
            
    def getNextState(self, board, _player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action as portable Python
            nextPlayer: player who plays in the next turn (should be -player)
        """
        actionPo = self.vectorIndexActionToAtlatl(action, board)
        nextState = self.game.transition(board["state"], actionPo)
        nextBoard = {"param":board["param"], "state":nextState}
        nextPlayer = 0
        return nextBoard, nextPlayer

    def getValidMoves(self, board, _player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        result = [0]*self.getActionSize()
        statePo = board["state"]
        actionsJSON = self.game.legal_actions(statePo)
        for actionPo in actionsJSON:
            index = self.atlatlActionToVectorIndex(actionPo, board)
            result[index] = 1
        return result


    def getGameEnded(self, board, _player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            Was: r: 0 if game has not ended. 1 if player won, -1 if player lost,
            small non-zero value for draw. 
            Now (isTerminal, score), where 0<=score<=1
        """
        statePo = board["state"]
        isTerminal = statePo["status"]["isTerminal"]
        maxScoreCI5 = 240+400 # own city always and kill all opposing force
        score = statePo["status"]["score"]
        score_11 = float(score)/maxScoreCI5
        return (isTerminal, score_11)

    def getCanonicalForm(self, board, _player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                    form of the board and the corresponding pi vector. This
                    is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                        Required by MCTS for hashing.
        """
        pass

if __name__=="__main__":
    import json
    import game_dispenser
    import scenario
    import random
    # scenarioName = "atomic.scn"
    # scenarioPo = json.load( open("scenarios/"+scenarioName) )
    scenarioGen = scenario.clear_square_factory(**{'size':5, 'min_units':2, 'max_units':4, 'num_cities':1})
    scenarioPo = scenarioGen()
    game = game.Game(scenarioPo)
    ASgame = AlphaStarGame(game)
    board = ASgame.getInitBoard()
    action2index = {}
    print("Move Index, JSON Action")
    for i in range(151):
        result = ASgame.vectorIndexActionToAtlatl(i, board)
        if result:
            print(f'{i} {result}')
            action2index[json.dumps(result)] = i
    print("Verifying Result")
    for actionJSON in action2index:
        index1 = action2index[actionJSON]
        actionPo = json.loads(actionJSON)
        index2 = ASgame.atlatlActionToVectorIndex(actionPo, board)
        status = ""
        if index1!=index2:
            status = "FAIL"
        print(f'{index1} {index2} {status}')
    print("\n\nRandom Play Test")
    _player = 0
    done = False
    score = None
    while not done:
        validVec = ASgame.getValidMoves(board, _player)
        validIndices = []
        for i in range(len(validVec)):
            if validVec[i]:
                validIndices.append(i)
        action = random.choice(validIndices)
        print(f'action {action}')
        board, _nextPlayer = ASgame.getNextState(board, _player, action)
        done, score = ASgame.getGameEnded(board, _player)
    print(f'score {score}')
    
