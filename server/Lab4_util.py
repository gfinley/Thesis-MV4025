from pickletools import read_decimalnl_short
import numpy as np
import map


global util_red_units 
global util_blue_units
global util_game_Index
global units_hidding

#added for red/blue mirror functionality
global units_hidding_blue

# gloabals for spreading phase
#added to allow possible red and blue AI's

global NEXT_SPREAD_RED_PHASE
global NEXT_SPREAD_BLUE_PHASE

NEXT_SPREAD_RED_PHASE = 2
NEXT_SPREAD_BLUE_PHASE = 1


global possible_hexes
util_red_units = []
util_blue_units = []
util_game_Index = []

possible_hexes = []


game_size = 1500

#default for origional blue
global mat 

#added to support red/blue mirror functionality
global red_mat


#this function should not be depriciated
def assign_team_units(gamestate):
    units = gamestate.game_state
    for unit in units['units']:
        if unit['faction'] == "red":
            util_red_units.append(unit)

def assign_team_units2(units):
    global NEXT_SPREAD_RED_PHASE
    global NEXT_SPREAD_BLUE_PHASE
    NEXT_SPREAD_RED_PHASE = 2
    NEXT_SPREAD_BLUE_PHASE = 1
    global util_red_units
    global util_blue_units 
    util_red_units = []
    util_blue_units = []
    for unit in units:
        temp_unit = units[unit]
        if temp_unit.faction == "red":
            util_red_units.append(temp_unit)
        if temp_unit.faction == "blue":
            util_blue_units.append(temp_unit)
    return


#this function is replicated below for red/blue mirror functionality
def assign_initial_red_uncertinty(map_data):
    global util_red_units 
    global util_blue_units

    global possible_hexes
    global mat
    mat = np.zeros((game_size,game_size))
    possible_hexes = []
    for unit in util_red_units:
        x_offset = unit.hex.x_offset
        y_offset = unit.hex.y_offset
        mat[y_offset, x_offset] = 1
        possible_hexes.append([unit.uniqueId, unit.hex.id] )
    #temp for spreading of uncertinty
    #spread_uncertinty(game_state)
    #print(mat)
    return

#function added to allow for AI to play as red against blue, is a duplicate of above function
#for improvment combine with previous function
def assign_initial_blue_uncertinty(map_data):
    global util_red_units 
    global util_blue_units

    global possible_hexes_blue
    global red_mat
    red_mat = np.zeros((game_size,game_size))
    possible_hexes_blue = []
    for unit in util_blue_units:
        x_offset = unit.hex.x_offset
        y_offset = unit.hex.y_offset
        red_mat[y_offset, x_offset] = 1
        possible_hexes_blue.append([unit.uniqueId, unit.hex.id] )
    return



#need some sort of data sctructire to contain where a unit may be.
# ['red 0', ['hex-1-0']] then this would allow 

def spread_uncertinty(self, map_data, unit_data):  #added to support red/blue mirror functionality
    global possible_hexes
    global possible_hexes_blue

    #added to enable use of red/blue mirror functionality
    temp_possible_hexes = []
    if self.role == "blue":
        temp_possible_hexes = possible_hexes
    if self.role == "red":
        temp_possible_hexes = possible_hexes_blue


    global units_hidding
    global units_hidding_blue
    temp_units_hidding = {}
    temp_holder = []
    for unit in unit_data.unitIndex:
        if unit_data.unitIndex[unit].detected == False:
            temp_units_hidding[unit] = 1
            for poss_hex in temp_possible_hexes:
                #unit_faction = unit_data.unitIndex[unit]
                if (unit_data.unitIndex[unit].faction != self.role) and (poss_hex[0] == unit):
                    #this is a possible hex get all possible spread spots including where it is
                    if [unit,poss_hex[1]] not in temp_holder:
                        temp_holder.append([unit , poss_hex[1]])

                    hex = map_data.hexIndex[poss_hex[1]]
                    neighbors = map.getNeighborHexes(hex, map_data)
                    for neigh in neighbors:
                        if [unit,neigh.id] not in temp_holder:
                            temp_units_hidding[unit] += 1
                            temp_holder.append([unit,neigh.id])

    if self.role == "blue":
        units_hidding = temp_units_hidding
    if self.role == "red":
        units_hidding_blue = temp_units_hidding
        #all of the possible squares should have been located
    #asisgn everythig to mat
    if self.role == "blue":
        possible_hexes = temp_holder
    if self.role == "red":
        possible_hexes_blue = temp_holder
    return


def get_uncertinty(self,map_data,unit_data):
    global possible_hexes
    global possible_hexes_blue
    global units_hidding
    global units_hidding_blue


    temp_possible_hexes = []
    temp_units_hidding = {}
    #check to see if any other team units can see any of the allied vision

    if self.role == "blue":
        hex_to_check = possible_hexes
    if self.role == "red":
        hex_to_check = possible_hexes_blue

    for possible_hex in hex_to_check:
        flag = False
        hex2 = map_data.hexIndex[possible_hex[1]]
        for unit in unit_data.unitIndex:
            if unit_data.unitIndex[unit].faction == self.role and  unit_data.unitIndex[unit].ineffective == False :
                hex1 = unit_data.unitIndex[unit].hex
                if (check_if_in_range(hex1,hex2,2)):
                    flag = True
                    break
        if not flag:    
            temp_possible_hexes.append(possible_hex)
            #check failed, need to remove number from possible hex number tracker

    if self.role == "blue":
        possible_hexes = temp_possible_hexes
    if self.role == "red":
        possible_hexes_blue = temp_possible_hexes
    #need toi remake possibilites numbers
    temp_units_hidding = {}
    #make initial list
    for unit in unit_data.unitIndex:
        if unit_data.unitIndex[unit].faction != self.role and unit_data.unitIndex[unit].detected == False:
            temp_units_hidding[unit] = 0
    for item in temp_possible_hexes:
        if unit_data.unitIndex[item[0]].detected == False:
            temp_units_hidding[item[0]] += 1

    units_hidding = temp_units_hidding


    mat = mat = np.zeros((game_size,game_size))

    for unit in unit_data.unitIndex:
        if (unit_data.unitIndex[unit].detected == False):
            for item in temp_possible_hexes:
                if item[0] == unit:
                    hex = map_data.hexIndex[item[1]]
                    mat[hex.y_offset, hex.x_offset] += ( 1/ (units_hidding[unit] ) )
    #print(mat)
    return mat

def check_if_in_range(hex1, hex2, range):
    vision_range = range
    range = map.gridDistance(hex1.x_grid, hex1.y_grid, hex2.x_grid, hex2.y_grid)  
    if range >= vision_range:
        return False
    else:
        return True
