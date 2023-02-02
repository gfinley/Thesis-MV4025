import asyncio
import websockets
import json
import argparse
import numpy as np

# This AI has a representation of the map and units, and updates the unit representation as it changes
import map
import unit

action_count = 0

#azure imports
import urllib.request
import json
import os
import ssl
import time
#end azure imports

#lab4 utils import
import Lab4_util
import naval_utils

global util_red_units 
global util_blue_units

global NEXT_SPREAD_RED_PHASE
global NEXT_SPREAD_BLUE_PHASE

class NoNegativesRewArt:
    def __init__(self, own_faction=None):
        self.negative_rewards = 0
    def engineeredReward(self, reward, unitData=None, is_terminal = False):
        # Make this function just "return(reward)" to use raw rewards
        if reward < 0:
            self.negative_rewards += 1
            # Negative rewards turn into zero
            return 0
        # Positive rewards get discounted based on the number of negative rewards
        base = 10
        reward_discount = base / (base + self.negative_rewards)
        return reward * reward_discount

class BoronRewArt:
    # Boron did not use a terminal bonus (equiv. to terminal_bonus=0)
    def __init__(self, own_faction, terminal_bonus=25):
        self.own_faction = own_faction
        self.original_strength = None
        self.terminal_bonus = terminal_bonus
    def _totalFriendlyStrength(self, ownFaction, unitData):
        sum = 0
        for unt in unitData.units():
            if not unt.ineffective and unt.faction==ownFaction:
                sum += unt.currentStrength
        return sum
    def engineeredReward(self, raw_reward, unitData, is_terminal = False):
        if self.original_strength is None:
            self.original_strength = self._totalFriendlyStrength(self.own_faction, unitData)
        current_strength = self._totalFriendlyStrength(self.own_faction, unitData)
        if raw_reward < 0:
            raw_reward = 0
        if is_terminal:
            # Experimental
            raw_reward += self.terminal_bonus
        return raw_reward * current_strength / self.original_strength

class AI:
    evenXOffsets18 = ((0,-1), (1,-1), (1,0), (0,1), (-1,0), (-1,-1),
                        (0,-2), (1,-2), (2,-1), (2,0), (2,1), (1,1), 
                        (0,2), (-1,1), (-2,1), (-2,0), (-2,-1), (-1,-2))
    oddXOffsets18 = ((0,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0),
                        (0,-2), (1,-1), (2,-1), (2,0), (2,1), (1,2), 
                        (0,2), (-1,2), (-2,1), (-2,0), (-2,-1), (-1,-1))
    def __init__(self, role, kwargs={}):
        self.role = role
        self.rewart_class = BoronRewArt
        self.reset()
    def reset(self):
        self.accumulated_reward = 0
        self.last_terminal = None
        self.attempted_moveD = {}
        self.response_fn = None
        self.last_score = 0
        self.mapData = None
        self.unitData = None
        self.phaseCount = 0
        self.rewArt = self.rewart_class(self.role)
    def getNFeatures(self):
        return 3
    def nextMover(self):
        for un in self.unitData.units():
            if un.faction==self.role and not self.attempted_moveD[un.uniqueId] and not un.ineffective:
                return un
        return None
    def sendToServer(self, messageO):
        if self.response_fn:
            self.response_fn(messageO)
    def updateLocalState(self, obs):
        for unitObs in obs['units']:
            uniqueId = unitObs['faction'] + " " + unitObs['longName']
            un = self.unitData.unitIndex[ uniqueId ]
            un.partialObsUpdate( unitObs, self.unitData, self.mapData )
        reward = obs['status']['score'] - self.last_score
        # Reward must have its sign flipped if red is to be trained
        is_terminal = obs['status']['isTerminal']
        self.accumulated_reward += self.rewArt.engineeredReward(reward, self.unitData, is_terminal)
        self.last_score = obs['status']['score']
    def process(self, message, response_fn):
        self.response_fn = response_fn # Store for later use
        msgD = json.loads(message)
        if msgD['type'] == "parameters":
            paramD = msgD['parameters']
            # reset state variables
            self.reset()
            self.mapData = map.MapData()
            self.unitData = unit.UnitData()
            self.rewArt = self.rewart_class(self.role)
            self.phase = None
            for unt in self.unitData.units():
                if unt.faction == self.role:
                    self.attempted_moveD[unt.uniqueId] = False           
            map.fromPortable(paramD['map'], self.mapData)
            unit.fromPortable(paramD['units'], self.unitData, self.mapData)
            responseD = { "type":"role-request", "role":self.role}
        elif msgD['type'] == 'observation':
            obs = msgD['observation']
            self.last_terminal = obs['status']['isTerminal']
            if self.last_terminal:
                # Needed to provide final observation to gym agents
                self.updateLocalState(obs)
                responseD = {"type":"gym-pause"}
            elif obs['status']['onMove'] == self.role:
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
                    self.updateLocalState(obs)
                    if self.nextMover():
                        responseD = {"type":"gym-pause"}
                    else: # Possibly no friendlies left alive
                        responseD = { "type":"action", "action":{"type":"pass"} }
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
    def feature(self, fn, type):
        count = 0
        self.arrayIndex = {}
        self.inverseIndex = []
        for hexId in self.mapData.hexIndex:
            self.arrayIndex[hexId] = count
            self.inverseIndex.append(hexId)
            count += 1
        dim = self.mapData.getDimensions()
        mat = np.zeros((dim['height'],dim['width']))
        if type=="hex":
            for hexId in self.mapData.hexIndex:
                hex = self.mapData.hexIndex[hexId]
                mat[hex.y_offset, hex.x_offset] = fn(hex)
        else: # type=="unit"
            for unitId in self.unitData.unitIndex:
                unt = self.unitData.unitIndex[unitId]
                hex = unt.hex
                if hex:
                    mat[hex.y_offset, hex.x_offset] = fn(unt)
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
        mat = np.zeros((dim['height'],dim['width']))
            
        #above unhanged from original Atlatl function, below modified to take input from map goal data and create feature currently all cases will flow into
        #"else" part of function, structure maintained for furure use
        if type=="hex":
            for hexId in self.mapData.hexIndex:
                hex = self.mapData.hexIndex[hexId]
                x_mat, y_mat = hex.x_offset, hex.y_offset
                mat[y_mat, x_mat] = fn(hex)
        else: # type=="unit"
            for hex in [naval_utils.get_blue_goals()]:
                hex_actual = self.mapData.hexIndex[hex]
                if hex_actual:
                    x_mat, y_mat = hex_actual.x_offset, hex_actual.y_offset
                    mat[y_mat, x_mat] = 1
        return mat

    #new feature for position uncertiniy
    #I have forgotten what this does, check to delete later
    def feature_uncertinty(self,fn, type):
        count = 0
        self.arrayIndex = {}
        self.inverseIndex = []
        for hexId in self.mapData.hexIndex:
            self.arrayIndex[hexId] = count
            self.inverseIndex.append(hexId)
            count += 1
        dim = self.mapData.getDimensions()
        mat = np.zeros((dim['height'],dim['width']))

        #start the loop over all the hexes
        for hexID in self.mapData.hexIndex:
            hex = self.mapData.hexIndex[hexID]
            if hex.red_occupation_probabilty != 0:
                #get "legal move conidits"
                #legal_move_hexes = self.legalMoveHexes(next_mover)
                moveTargets = mover.findMoveTargets(self.mapData, self.unitData)
                return 0
        return 0


        


    def action_result(self):
        done = self.last_terminal 
        reward = self.accumulated_reward
        self.accumulated_reward = 0
        info = {'score':self.last_score}
        return (self.observation(), reward, done, info)
    def observation(self):
        next_mover = self.nextMover()
        # if next_mover:
        #     mover_feature = self.feature(moverFeatureFactory(next_mover),"unit")
        # else:
        #     mover_feature = self.feature(lambda x:0,"hex")
        return np.stack( [
                            self.feature(moverFeatureFactory(next_mover),"unit"),
                            self.feature(blueUnitFeature,"unit"), 
                            self.feature(redUnitFeature,"unit")
                         ] )
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
        if action==0:
            return self.noneOrEndMove() # wait
        hex = mover.hex
        moveTargets = mover.findMoveTargets(self.mapData, self.unitData)
        moveTargetIds = [hex.id for hex in moveTargets]
        fireTargets = mover.findFireTargets(self.unitData)
        if not moveTargets and not fireTargets:
            # No legal moves
            return self.noneOrEndMove()
        if hex.x_offset%2:
            delta = AI.oddXOffsets18[action-1]
        else:
            delta = AI.evenXOffsets18[action-1]
        to_hex_id = f'hex-{hex.x_offset+delta[0]}-{hex.y_offset+delta[1]}'
        if not to_hex_id in self.mapData.hexIndex:
            # Off-map move.
            return self.noneOrEndMove()
        to_hex = self.mapData.hexIndex[to_hex_id]
        if to_hex in moveTargets:
            return {"type":"action", "action":{"type":"move", "mover":mover.uniqueId, "destination":to_hex_id}}
        for fireTarget in fireTargets:
            if to_hex == fireTarget.hex:
                return {"type":"action", "action":{"type":"fire", "source":mover.uniqueId, "target":fireTarget.uniqueId}}
        # Illegal move request
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
        
        size = 1500

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
        mat = np.zeros((10,50))

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

        return  np.stack( [mat])


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
                            self.feature_navy_goal(goalFeatureFactory(),"other")
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


#goal feature factory added to support layer for blocade running goals
def goalFeatureFactory():
    def inner(hex):
        if hex:
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
