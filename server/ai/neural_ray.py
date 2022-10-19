import asyncio
import websockets
import json
import argparse
import numpy as np

import torch

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import dqn

# This AI has a representation of the map and units, and updates the unit representation as it changes
import map
import unit


default_neural_net_file = "PPO_save_test_1_20221012-122916_state"
action_count = 0

#lab4 utils import
import Lab4_util

global util_red_units 
global util_blue_units

global NEXT_SPREAD_RED_PHASE
global NEXT_SPREAD_BLUE_PHASE

def debugPrint(str):
    condition = True
    if condition:
        print(str)

class AI:
    evenXOffsets18 = ((0,-1), (1,-1), (1,0), (0,1), (-1,0), (-1,-1),
                        (0,-2), (1,-2), (2,-1), (2,0), (2,1), (1,1), 
                        (0,2), (-1,1), (-2,1), (-2,0), (-2,-1), (-1,-2))
    oddXOffsets18 = ((0,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0),
                        (0,-2), (1,-1), (2,-1), (2,0), (2,1), (1,2), 
                        (0,2), (-1,2), (-2,1), (-2,0), (-2,-1), (-1,-1))
    def __init__(self, role, kwargs):
        self.role = role
        if "neuralNet" in kwargs:
            neural_net_file = kwargs["neuralNet"]
        else:
            neural_net_file = default_neural_net_file
        self.dqn = kwargs["dqn"]
        if self.dqn:
            self.model = torch.load(neural_net_file)
        else:
            self.model = torch.load(neural_net_file)
            self.model.eval()
        self.doubledCoordinates = kwargs["doubledCoordinates"]
        self.mapData = None
        self.unitData = None
        self.reset()
    def reset(self):
        self.attempted_moveD = {}
        self.phaseCount = 0
    def nextMover(self):
        for un in self.unitData.units():
            if un.faction==self.role and not self.attempted_moveD[un.uniqueId] and not un.ineffective:
                return un
        return None
    def moveMessage(self):
        while self.nextMover():
            debugPrint(f"processing next mover {self.nextMover().uniqueId}")
            action, _states = self.model(self.model, self.observation())
            msg = self.actionMessageDiscrete(action)
            if msg != None:
                return msg
        # No next mover
        return { "type":"action", "action":{"type":"pass"} }
    def process(self, message, response_fn=None):
        msgD = json.loads(message)
        if msgD['type'] == "parameters":
            paramD = msgD['parameters']
            # reset state variables
            self.mapData = map.MapData()
            self.unitData = unit.UnitData()
            self.phase = None
            for unt in self.unitData.units():
                if unt.faction == self.role:
                    self.attempted_moveD[unt.uniqueId] = False           
            map.fromPortable(paramD['map'], self.mapData)
            unit.fromPortable(paramD['units'], self.unitData, self.mapData)
            responseD = { "type":"role-request", "role":self.role}
        elif msgD['type'] == 'observation':
            obs = msgD['observation']
            if not obs['status']['isTerminal'] and obs['status']['onMove'] == self.role:
                if obs['status']['setupMode']:
                    self.phase = "setup"
                    responseD = { "type":"action", "action":{"type":"pass"} }
                else:
                    if self.phase != "move":
                        self.phase = "move"
                        self.phaseCount = obs['status']['phaseCount']
                        for unt in self.unitData.units():
                            if unt.faction == self.role:
                                self.attempted_moveD[unt.uniqueId] = False
                    for unitObs in obs['units']:
                        uniqueId = unitObs['faction'] + " " + unitObs['longName']
                        un = self.unitData.unitIndex[ uniqueId ]
                        un.partialObsUpdate( unitObs, self.unitData, self.mapData )
                    responseD = self.moveMessage() # Might be a pass
            else:
                self.phase = "wait"
                responseD = None       
        elif msgD['type'] == 'reset':
            self.reset()
            responseD = None
        else:
            raise Exception(f'Unknown message type {msgD["type"]}')
        if responseD:
            return json.dumps(responseD)
    #original Atlatl function, Navy version variation below 
    def feature(self, fn, type):
        count = 0
        self.arrayIndex = {}
        self.inverseIndex = []
        for hexId in self.mapData.hexIndex:
            self.arrayIndex[hexId] = count
            self.inverseIndex.append(hexId)
            count += 1
        dim = self.mapData.getDimensions()
        if self.doubledCoordinates:
            mat = np.zeros((2*dim['height']+1,dim['width']))
        else:
            mat = np.zeros((dim['height'],dim['width']))
        if type=="hex":
            for hexId in self.mapData.hexIndex:
                hex = self.mapData.hexIndex[hexId]
                x_mat, y_mat = hex.x_offset, hex.y_offset
                if self.doubledCoordinates:
                    y_mat = 2*y_mat + x_mat%2
                mat[y_mat, x_mat] = fn(hex)
        else: # type=="unit"
            for unitId in self.unitData.unitIndex:
                unt = self.unitData.unitIndex[unitId]
                hex = unt.hex
                if hex:
                    x_mat, y_mat = hex.x_offset, hex.y_offset
                    if self.doubledCoordinates:
                        y_mat = 2*y_mat + x_mat%2
                    mat[y_mat, x_mat] = fn(unt)
        return mat
    
    def feature_navy_goal(self, fn, type):
            count = 0
            self.arrayIndex = {}
            self.inverseIndex = []
            for hexId in self.mapData.hexIndex:
                self.arrayIndex[hexId] = count
                self.inverseIndex.append(hexId)
                count += 1
            dim = self.mapData.getDimensions()
            if self.doubledCoordinates:
                mat = np.zeros((2*dim['height']+1,dim['width']))
            else:
                mat = np.zeros((dim['height'],dim['width']))
                
            #above unhanged from original Atlatl function, below modified to take input from map goal data and create feature currently all cases will flow into
            #"else" part of function, structure maintained for furure use
            if type=="hex":
                for hexId in self.mapData.hexIndex:
                    hex = self.mapData.hexIndex[hexId]
                    x_mat, y_mat = hex.x_offset, hex.y_offset
                    if self.doubledCoordinates:
                        y_mat = 2*y_mat + x_mat%2
                    mat[y_mat, x_mat] = fn(hex)
            else: # type=="unit"
                for hex in self.mapData.goal:
                    if hex:
                        x_mat, y_mat = hex.x_offset, hex.y_offset
                        if self.doubledCoordinates:
                            y_mat = 2*y_mat + x_mat%2
                        mat[y_mat, x_mat] = 1
            return mat
    def observation(self):
        next_mover = self.nextMover()
        return np.stack( [
                            self.feature(moverFeatureFactory(next_mover),"unit"),
                            self.feature(blueUnitFeature,"unit"), 
                            self.feature(redUnitFeature,"unit")
                         ] )
    def action_result(self):
        done = self.last_terminal
        reward = self.accumulated_reward
        self.last_terminal = None # Make False???
        self.accumulated_reward = 0
        debugPrint(f'action_result returning done {done}')
        return (self.observation(), reward, done, {})
    def noneOrEndMove(self):
        if self.nextMover():
            return None
        else:
            return { "type":"action", "action":{"type":"pass"} }
    def actionMessageDiscrete(self, action):
        # Action represents the six or 18 most adjacent hexes plus wait,
        # either in range 0..6 or 0..18
        global action_count
        action_count += 1
        mover = self.nextMover()
        self.attempted_moveD[mover.uniqueId] = True
        
        debugPrint(f'gym_ai_surrogate:actionMessageDiscrete mover {mover.uniqueId} hex {mover.hex.id} action {action}')
            
        if action==0:
            debugPrint("Action was wait")
            return self.noneOrEndMove() # wait
        hex = mover.hex
        moveTargets = mover.findMoveTargets(self.mapData, self.unitData)
        fireTargets = mover.findFireTargets(self.unitData)
        if not moveTargets and not fireTargets:
            # No legal moves
            debugPrint("No legal moves")
            return self.noneOrEndMove()
        if hex.x_offset%2:
            delta = AI.oddXOffsets18[action-1]
        else:
            delta = AI.evenXOffsets18[action-1]
        to_hex_id = f'hex-{hex.x_offset+delta[0]}-{hex.y_offset+delta[1]}'
        if not to_hex_id in self.mapData.hexIndex:
            # Off-map move.
            debugPrint("Off-map move")
            return self.noneOrEndMove()
        to_hex = self.mapData.hexIndex[to_hex_id]
        if to_hex in moveTargets:
            return {"type":"action", "action":{"type":"move", "mover":mover.uniqueId, "destination":to_hex_id}}
        for fireTarget in fireTargets:
            if to_hex == fireTarget.hex:
                return {"type":"action", "action":{"type":"fire", "source":mover.uniqueId, "target":fireTarget.uniqueId}}
        # Illegal move request
        debugPrint("Illegal move")
        return self.noneOrEndMove()

class AIx2(AI):
    def __init__(self, role, kwargs):
        AI.__init__(self, role, kwargs)
    def observation(self):
        obs = AI.observation(self)
        shp = obs.shape
        new_shape = (shp[0], shp[1]*2+1, shp[2])
        mat = np.zeros(new_shape)
        for z in range(shp[0]):
            for y in range(shp[1]):
                for x in range(shp[2]):
                    mat[z, 2*y+(x%2), x] = obs[z,y,x]
        return mat

class AITwelve(AI):
    def __init__(self, role, kwargs):
        AI.__init__(self, role, kwargs)
    def observation(self):
        next_mover = self.nextMover()
        legal_move_hexes = self.legalMoveHexes(next_mover)
        return np.stack( [
                            self.feature(moverFeatureFactory(next_mover),"unit"),
                            self.feature(legalMoveFeatureFactory(legal_move_hexes),"hex"),
                            self.feature(blueUnitFeature,"unit"), 
                            self.feature(redUnitFeature,"unit"),
                            self.feature(unitTypeFeatureFactory("infantry"),"unit"),
                            self.feature(unitTypeFeatureFactory("mechinf"),"unit"),
                            self.feature(unitTypeFeatureFactory("armor"),"unit"),
                            self.feature(unitTypeFeatureFactory("artillery"),"unit"),
                            self.feature(terrainFeatureFactory("clear"),"hex"),
                            self.feature(terrainFeatureFactory("water"),"hex"),
                            self.feature(terrainFeatureFactory("rough"),"hex"),
                            self.feature(terrainFeatureFactory("urban"),"hex")
                         ] )
    def legalMoveHexes(self, mover):
        result = {}
        if mover:
            fireTargets = mover.findFireTargets(self.unitData)
            for unt in fireTargets:
                result[unt.hex.id] = True
            moveTargets = mover.findMoveTargets(self.mapData, self.unitData)
            for hex in moveTargets:
                result[hex.id] = True
        return result
    def getNFeatures(self):
        return 12


class AI13(AI):
    def __init__(self, role, kwargs):
        AI.__init__(self, role, kwargs)
    def observation(self):
        next_mover = self.nextMover()
        legal_move_hexes = self.legalMoveHexes(next_mover)
        phase_indicator = 0.9**self.phaseCount
        return np.stack( [
                            self.feature(moverFeatureFactory(next_mover),"unit"),
                            self.feature(legalMoveFeatureFactory(legal_move_hexes),"hex"),
                            self.feature(blueUnitFeature,"unit"), 
                            self.feature(redUnitFeature,"unit"),
                            self.feature(unitTypeFeatureFactory("infantry"),"unit"),
                            self.feature(unitTypeFeatureFactory("mechinf"),"unit"),
                            self.feature(unitTypeFeatureFactory("armor"),"unit"),
                            self.feature(unitTypeFeatureFactory("artillery"),"unit"),
                            self.feature(terrainFeatureFactory("clear"),"hex"),
                            self.feature(terrainFeatureFactory("water"),"hex"),
                            self.feature(terrainFeatureFactory("rough"),"hex"),
                            self.feature(terrainFeatureFactory("urban"),"hex"),
                            self.feature(constantFeatureFactory(phase_indicator),"hex")
                         ] )
    def legalMoveHexes(self, mover):
        result = {}
        if mover:
            fireTargets = mover.findFireTargets(self.unitData)
            for unt in fireTargets:
                result[unt.hex.id] = True
            moveTargets = mover.findMoveTargets(self.mapData, self.unitData)
            for hex in moveTargets:
                result[hex.id] = True
        return result
    def getNFeatures(self):
        return 13

class AI14(AI):
    def __init__(self, role, kwargs):
        AI.__init__(self, role, kwargs)
    def observation(self):
        next_mover = self.nextMover()
        legal_move_hexes = self.legalMoveHexes(next_mover)
        phase_indicator = 0.9**self.phaseCount
        return np.stack( [
                            self.feature(moverFeatureFactory(next_mover),"unit"),
                            self.feature(canMoveFeature,"unit"),
                            self.feature(legalMoveFeatureFactory(legal_move_hexes),"hex"),
                            self.feature(blueUnitFeature,"unit"), 
                            self.feature(redUnitFeature,"unit"),
                            self.feature(unitTypeFeatureFactory("infantry"),"unit"),
                            self.feature(unitTypeFeatureFactory("mechinf"),"unit"),
                            self.feature(unitTypeFeatureFactory("armor"),"unit"),
                            self.feature(unitTypeFeatureFactory("artillery"),"unit"),
                            self.feature(terrainFeatureFactory("clear"),"hex"),
                            self.feature(terrainFeatureFactory("water"),"hex"),
                            self.feature(terrainFeatureFactory("rough"),"hex"),
                            self.feature(terrainFeatureFactory("urban"),"hex"),
                            self.feature(constantFeatureFactory(phase_indicator),"hex")
                         ] )
    def legalMoveHexes(self, mover):
        result = {}
        if mover:
            fireTargets = mover.findFireTargets(self.unitData)
            for unt in fireTargets:
                result[unt.hex.id] = True
            moveTargets = mover.findMoveTargets(self.mapData, self.unitData)
            for hex in moveTargets:
                result[hex.id] = True
        return result
    def getNFeatures(self):
        return 14


class AI15(AI):
    def __init__(self, role, kwargs):
        AI.__init__(self, role, kwargs)
    def observation(self):
        

        if self.role == "blue":
            #only want to spread uncertinty for the next phase when red could have moved
            if self.phaseCount == Lab4_util.NEXT_SPREAD_RED_PHASE:
                Lab4_util.spread_uncertinty(self, self.mapData, self.unitData)
                Lab4_util.NEXT_SPREAD_RED_PHASE = Lab4_util.NEXT_SPREAD_RED_PHASE + 2
                #print("spread uncertinty phase:", self.phaseCount)
                #print(Lab4_util.get_uncertinty(self,self.mapData, self.unitData))
            mat = Lab4_util.get_uncertinty(self,self.mapData, self.unitData)
            mat_to_pass = mat
        if self.role == "red":
            if self.phaseCount == Lab4_util.NEXT_SPREAD_BLUE_PHASE:
                Lab4_util.spread_uncertinty(self,self.mapData, self.unitData)
                Lab4_util.NEXT_SPREAD_BLUE_PHASE = Lab4_util.NEXT_SPREAD_BLUE_PHASE + 2
            red_mat = Lab4_util.get_uncertinty(self,self.mapData, self.unitData)
            mat_to_pass = red_mat

        
        #print(mat_to_pass)
        next_mover = self.nextMover()
        legal_move_hexes = self.legalMoveHexes(next_mover)
        phase_indicator = 0.9**self.phaseCount
        return np.stack( [
                            self.feature(moverFeatureFactory(next_mover),"unit"),
                            self.feature(canMoveFeature,"unit"),
                            self.feature(legalMoveFeatureFactory(legal_move_hexes),"hex"),
                            self.feature(blueUnitFeature,"unit"), 
                            self.feature(redUnitFeature,"unit"),
                            self.feature(unitTypeFeatureFactory("infantry"),"unit"),
                            self.feature(unitTypeFeatureFactory("mechinf"),"unit"),
                            self.feature(unitTypeFeatureFactory("armor"),"unit"),
                            self.feature(unitTypeFeatureFactory("artillery"),"unit"),
                            self.feature(terrainFeatureFactory("clear"),"hex"),
                            self.feature(terrainFeatureFactory("water"),"hex"),
                            self.feature(terrainFeatureFactory("rough"),"hex"),
                            self.feature(terrainFeatureFactory("urban"),"hex"),
                            self.feature(constantFeatureFactory(phase_indicator),"hex"),
                            mat_to_pass
                         ] )
    def legalMoveHexes(self, mover):
        result = {}
        if mover:
            fireTargets = mover.findFireTargets(self.unitData)
            for unt in fireTargets:
                result[unt.hex.id] = True
            moveTargets = mover.findMoveTargets(self.mapData, self.unitData)
            for hex in moveTargets:
                result[hex.id] = True
        return result
    def getNFeatures(self):
        return 15

class AI_RAY(AI):
    def __init__(self, role, kwargs):
        AI.__init__(self, role, kwargs)
    def observation(self):
        
        size = 5

        #get all the red units
        red_units = []
        blue_units = []
        for unit in self.unitData.unitIndex:
            if "red" in unit:
                #get units hex
                #check to see it the unit is effective
                if self.unitData.unitIndex[unit].ineffective == False:
                    hex = self.unitData.unitIndex[unit].hex
                    #get hex offsets
                    x = np.divide(hex.x_offset,size)
                    y = np.divide(hex.y_offset,size)
                    red_units.append([x,y])
        #get all the blue units
            if "blue" in unit:
                if self.unitData.unitIndex[unit].ineffective == False:
                #get units hex
                    hex = self.unitData.unitIndex[unit].hex
                    #get hex offsets
                    x = np.divide(hex.x_offset,size)
                    y = np.divide(hex.y_offset,size)
                    blue_units.append([x,y])

        #getCityHexes
        city_hexes = []
        for city in self.mapData.getCityHexes():
            x = np.divide(city.x_offset,size)
            y = np.divide(city.y_offset,size)
            city_hexes.append([x,y])


        #make a 5 by 5 np array of zeros
        mat = np.zeros((10,5))

        #write the red units to the matrix
        index = 0
        for unit in red_units:
            mat[0][index] = 1
            mat[1][index] = unit[0]
            mat[2][index] = unit[1]
            index = index + 1
        
        #write blue units
        index = 0
        for unit in blue_units:
            mat[3][index] = 1
            mat[4][index] = unit[0]
            mat[5][index] = unit[1]
            index = index + 1

        #write city hexes
        index = 0
        for city in city_hexes:
            mat[6][index] = 1
            mat[7][index] = city[0]
            mat[8][index] = city[1]
            index = index + 1
        
        
        phase_indicator = 0.9**self.phaseCount

        #write phase indicator to the matrix
        for x in range(0,5):
            mat[9][x] = phase_indicator

        
        #print(mat_to_pass)
        next_mover = self.nextMover()
        legal_move_hexes = self.legalMoveHexes(next_mover)
        phase_indicator = 0.9**self.phaseCount
        #print(mat)
        return mat


    def legalMoveHexes(self, mover):
        result = {}
        if mover:
            fireTargets = mover.findFireTargets(self.unitData)
            for unt in fireTargets:
                result[unt.hex.id] = True
            moveTargets = mover.findMoveTargets(self.mapData, self.unitData)
            for hex in moveTargets:
                result[hex.id] = True
        return result
    def getNFeatures(self):
        return 1

#this AI feature maker is designed to support naal operations, reduced number of features to what acctually occres in the Naval matric
#there is an included "blue goal matrix" to support passing in information on the blue goal for transport ships
#this may need to be proken out at a later date
class NAVY_SIMPLE(AI):
    def __init__(self, role, kwargs):
        AI.__init__(self, role, kwargs)
    def observation(self):
        next_mover = self.nextMover()
        legal_move_hexes = self.legalMoveHexes(next_mover)
        phase_indicator = 0.9**self.phaseCount
        return np.stack( [
                            self.feature(moverFeatureFactory(next_mover),"unit"),
                            self.feature(canMoveFeature,"unit"),
                            self.feature(legalMoveFeatureFactory(legal_move_hexes),"hex"),
                            self.feature(blueUnitFeature,"unit"), 
                            self.feature(redUnitFeature,"unit"),
                            self.feature(unitTypeFeatureFactory("destroyer"),"unit"),
                            self.feature(unitTypeFeatureFactory("submarine"),"unit"),
                            self.feature(unitTypeFeatureFactory("transport"),"unit"),
                            self.feature(terrainFeatureFactory("ocean"),"hex"),
                            self.feature(terrainFeatureFactory("land"),"hex"),
                            self.feature(constantFeatureFactory(phase_indicator),"hex"),
                            
                            #blue goal for transport ships
                            self.feature_navy_goal(goalFeatureFactory(),"hex")
                         ] )
    def legalMoveHexes(self, mover):
        result = {}
        if mover:
            fireTargets = mover.findFireTargets(self.unitData)
            for unt in fireTargets:
                result[unt.hex.id] = True
            moveTargets = mover.findMoveTargets(self.mapData, self.unitData)
            for hex in moveTargets:
                result[hex.id] = True
        return result
    def getNFeatures(self):
        return 12


async def client(ai, uri):
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            print(f"Message received by AI over websocket: {message[:100]}")
            result = ai.process(message)
            if result:
                await websocket.send( result )

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
    
def moverFeatureFactory(moving_unit):
    def inner(unt):
        if unt==moving_unit:
            return 1.0
        return 0.0
    return inner

def unitTypeFeatureFactory(type):
    def inner(unt):
        if unt.type == type:
            return 1.0
        return 0.0
    return inner

#goal feature factory added to support layer for blocade running goals
def goalFeatureFactory():
    def inner(hex):
        if hex:
            return 1.0
        return 0.0
    return inner

def terrainFeatureFactory(terrain):
    def inner(hex):
        if hex.terrain == terrain:
            return 1.0
        return 0.0
    return inner 

def legalMoveFeatureFactory(legal_move_hexes):
    def inner(hex):
        if hex.id in legal_move_hexes:
            return 1.0
        return 0.0
    return inner

def constantFeatureFactory(value):
    def inner(hex):
        return value
    return inner
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("faction")
    parser.add_argument("--uri")
    args = parser.parse_args()
    
    ai = AI(args.faction)
    uri = args.uri
    if not uri:
        uri = "ws://localhost:9999"
    asyncio.get_event_loop().run_until_complete(client(ai, uri))
