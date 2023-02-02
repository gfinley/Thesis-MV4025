from ctypes import util
import random
import math
import json
import map
import unit
import copy

##Lab4 add for utils
import Lab4_util

global util_red_units
global util_blue_units
global util_game_Index


import reporting
from reporting import historian



def from_file_factory(filename):
    scenario_dir = "scenarios/"
    scenario_S = open(scenario_dir+filename).read()
    scenarioPo = json.loads(scenario_S)
    def inner():
        return scenarioPo
    return inner  

def flip_colors(scenario):
    flipped_units = []
    for unit in scenario['units']:
        flipped_unit = copy.copy(unit)
        if unit['faction']=="red":
            flipped_unit['faction']="blue"
        else:
            flipped_unit['faction']="red"
        flipped_units.append(flipped_unit)
    flipped_scenario = copy.copy(scenario)
    flipped_scenario['units'] = flipped_units
    return flipped_scenario

def clear_square_factory(size=6, min_units=2, max_units=4, num_cities=0, scenarioSeed=None, scenarioCycle=0, balance=False, max_phases=10, fog_of_war=False):
    balance_next = False
    last_scenario = None
    priorstate = random.getstate()
    random.seed(scenarioSeed)
    randstate = random.getstate()
    random.setstate(priorstate)
    count = 0
    def inner():
        nonlocal randstate, scenarioSeed, scenarioCycle, count, balance_next, last_scenario
        if balance:
            if balance_next:
                balance_next = False
                return flip_colors(last_scenario)
            else:
                balance_next = True
        priorstate = random.getstate()
        random.setstate(randstate)
        if size<4:
            raise Exception(f'Requested size ({size}) too small (minimum is 4)')
        mapData = map.MapData()
        mapData.createHexGrid(size,size)
        unitData = unit.UnitData()
        def _add_units(faction,start_hexes):
            n = random.randint(min_units,max_units)
            for i in range(n):
                hex = random.choice(start_hexes)
                start_hexes.remove(hex)
                u_param = {"hex":hex,"type":"infantry","longName":str(i),"faction":faction,"currentStrength":100}
                unt = unit.Unit(u_param,unitData,mapData)
                #if(faction == "red"):
                #    print("start red hex: {} {}".format(hex.x_offset, hex.y_offset))

            return n
        blue_side, red_side = random.choice((("north","south"),("south","north"),("east","west"),("west","east")))
        blue_hexes = get_setup_hex_ids(size,blue_side)
        red_hexes = get_setup_hex_ids(size,red_side)
        n_blue = _add_units("blue",blue_hexes)
        n_red = _add_units("red",red_hexes)

        #print("N :{}, Red:{}, Blue:{},".format(count+1,n_red,n_blue),end='')
                #integration for Lab4
        #Lab4_util.assign_team_units2(unitData.unitIndex)

        #initial uncertinty set to 1
        #Lab4_util.assign_initial_red_uncertinty(mapData)
        #Lab4_util.assign_initial_blue_uncertinty(mapData)

        #integration for historian
        
        
        for i in range(num_cities):
            def place_city(city_loc):
                city_hexes = get_setup_hex_ids(size,city_loc)
                hex_id = random.choice(city_hexes)
                hex = mapData.hexIndex[hex_id]
                hex.terrain = "urban"
            if n_blue<n_red:
                place_city(blue_side)
            elif n_red<n_blue:
                place_city(red_side)
            else: # n_blue==n_red
                if blue_side in ["north", "south"]:
                    city_loc = "ns-middle"
                else:
                    city_loc = "ew-middle"
                place_city(city_loc)  

        count = count + 1
        if scenarioCycle:
            count %= scenarioCycle
        if scenarioCycle!=0 and count==0:
            random.seed(scenarioSeed)
        randstate = random.getstate()    
        score = {"maxPhases":max_phases,"lossPenalty":-1,"cityScore":24}
        random.setstate(priorstate)
        scenario = {"map":mapData.toPortable(), "units":unitData.toPortable(), "score":score}
        scenario["map"]["fogOfWar"] = fog_of_war
        last_scenario = scenario
        util_game_Index = unitData.unitIndex
        unitList =  list(unitData.unitIndex.items())
        #parse the list
        #unitList_2 = []
        #for single_unit in unitList:
        #    unitList_2.append([single_unit[0],single_unit[1].hex.x_offset,single_unit[1].hex.y_offset])
        #print(*unitList_2,sep=",")

        #added for historian
        historian.reset()
        historian.add_mapData(mapData)
        historian.add_unitData(unitData)
        historian.render_start()
        return scenario
    return inner

def get_setup_hex_ids(size, side):
    low_setup_margin = math.floor(size/2)-1
    high_setup_margin = math.floor(size/2)+1
    hexes = []
    if side == "north":
        i_min, i_max, j_min, j_max = 0, size-1, 0, low_setup_margin
    elif side == "east":
        i_min, i_max, j_min, j_max = high_setup_margin, size-1, 0, size-1
    elif side == "south":
        i_min, i_max, j_min, j_max = 0, size-1, high_setup_margin, size-1
    elif side == "west":
        i_min, i_max, j_min, j_max = 0, low_setup_margin, 0, size-1
    elif side == "ns-middle":
        i_min, i_max, j_min, j_max = 0, size-1, low_setup_margin+1, high_setup_margin-1
    else: # "ew-middle"
        i_min, i_max, j_min, j_max = low_setup_margin+1, high_setup_margin-1, 0, size-1
    for i in range(i_min,i_max+1):
        for j in range(j_min,j_max+1):
            hexes.append("hex-"+str(i)+"-"+str(j))
    return hexes


if __name__=="__main__":
    print( clear_square_factory(num_cities=1)() )