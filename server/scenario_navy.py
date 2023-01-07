from ctypes import util
import random
import math
import json
import map
import unit
import copy


import naval_utils



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


def clear_ocean_factory(size=6, min_units=1, max_units=3, num_cities=0, scenarioSeed=None, scenarioCycle=0, balance=False, max_phases=10, fog_of_war=False):
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
        mapData.createOceanHexGrid(size,size)
        unitData = unit.UnitData()
        def _add_units(faction,start_hexes):
            n = random.randint(min_units,max_units)
            if faction=="red":
                n = 1
            for i in range(n):
                hex = random.choice(start_hexes)
                start_hexes.remove(hex)
                u_param = {"hex":hex,"type":"destroyer","longName":str(i),"faction":faction,"currentStrength":100}
                unt = unit.Unit(u_param,unitData,mapData)
                #if(faction == "red"):
                #    print("start red hex: {} {}".format(hex.x_offset, hex.y_offset))

            return n
        blue_side, red_side = random.choice((("north","south"),("south","north"),("east","west"),("west","east")))
        blue_hexes = get_setup_hex_ids(size,blue_side)
        red_hexes = get_setup_hex_ids(size,red_side)
        n_blue = _add_units("blue",blue_hexes)
        n_red = _add_units("red",red_hexes)

        count = count + 1
        if scenarioCycle:
            count %= scenarioCycle
        if scenarioCycle!=0 and count==0:
            random.seed(scenarioSeed)
        randstate = random.getstate()    
        score = {"maxPhases":max_phases,"lossPenalty":-1,"cityScore":0}
        random.setstate(priorstate)
        scenario = {"map":mapData.toPortable(), "units":unitData.toPortable(), "score":score}
        scenario["map"]["fogOfWar"] = fog_of_war
        last_scenario = scenario
        return scenario
    return inner


def island_ocean_factory(size=6, min_units=1, max_units=3, number_of_islands=1, scenarioSeed=None, scenarioCycle=0, balance=False, max_phases=10, fog_of_war=False, island_size=2):
    balance_next = False
    last_scenario = None
    priorstate = random.getstate()
    random.seed(scenarioSeed)
    randstate = random.getstate()
    random.setstate(priorstate)
    count = 0
    def inner():
        nonlocal randstate, scenarioSeed, scenarioCycle, count, balance_next, last_scenario
        global unit #this was rereiered to fix unbounded local variable error, but I don't know why, works in below function
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
        
        unitData = unit.UnitData()
        mapData = map.MapData()
        mapData.createOceanHexGrid(size,size)
        
        def _add_units(faction,start_hexes):
            n = random.randint(min_units,max_units)
            for i in range(n):
                hex = random.choice(start_hexes)
                start_hexes.remove(hex)
                u_param = {"hex":hex,"type":"destroyer","longName":str(i),"faction":faction,"currentStrength":100}
                unt = unit.Unit(u_param,unitData,mapData)
                #if(faction == "red"):
                #    print("start red hex: {} {}".format(hex.x_offset, hex.y_offset))
            return n

        blue_side, red_side = random.choice((("north","south"),("south","north"),("east","west"),("west","east")))
        blue_hexes = get_setup_hex_ids(size,blue_side)
        red_hexes = get_setup_hex_ids(size,red_side)
        n_blue = _add_units("blue",blue_hexes)
        n_red = _add_units("red",red_hexes)

        all_hexes = mapData.hexes()
        
        #build all hexes
        island_hexes = []
        for hex in all_hexes:
            if hex.terrain == "ocean":
                island_hexes.append(hex)
        for single_unit in unitData.units():
            island_hexes.remove(single_unit.hex)

        #pick some number of hexes to make as islands
        island_hexes = random.sample(island_hexes, number_of_islands)

        #change every island in mapData to the terrain type "island"
        for hex in mapData.hexes():
            if hex in island_hexes:
                hex.terrain = "land"

        count = count + 1
        if scenarioCycle:
            count %= scenarioCycle
        if scenarioCycle!=0 and count==0:
            random.seed(scenarioSeed)
        randstate = random.getstate()    
        score = {"maxPhases":max_phases,"lossPenalty":-1,"cityScore":0}
        random.setstate(priorstate)
        scenario = {"map":mapData.toPortable(), "units":unitData.toPortable(), "score":score}
        scenario["map"]["fogOfWar"] = fog_of_war
        last_scenario = scenario
        return scenario
    return inner

def island_large_ocean_factory(size=6, min_units=1, max_units=3, number_of_islands=1, scenarioSeed=None, scenarioCycle=0, balance=False, max_phases=10, fog_of_war=False, island_size=2):
    balance_next = False
    last_scenario = None
    priorstate = random.getstate()
    random.seed(scenarioSeed)
    randstate = random.getstate()
    random.setstate(priorstate)
    count = 0
    def inner():
        nonlocal randstate, scenarioSeed, scenarioCycle, count, balance_next, last_scenario
        global unit #this was rereiered to fix unbounded local variable error, but I don't know why, works in below function
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
        
        unitData = unit.UnitData()
        mapData = map.MapData()
        mapData.createOceanHexGrid(size,size)
        
        def _add_units(faction,start_hexes):
            n = random.randint(min_units,max_units)
            for i in range(n):
                hex = random.choice(start_hexes)
                start_hexes.remove(hex)
                u_param = {"hex":hex,"type":"destroyer","longName":str(i),"faction":faction,"currentStrength":100}
                unt = unit.Unit(u_param,unitData,mapData)
                #if(faction == "red"):
                #    print("start red hex: {} {}".format(hex.x_offset, hex.y_offset))
            return n

        blue_side, red_side = random.choice((("north","south"),("south","north"),("east","west"),("west","east")))
        blue_hexes = get_setup_hex_ids(size,blue_side)
        red_hexes = get_setup_hex_ids(size,red_side)
        n_blue = _add_units("blue",blue_hexes)
        n_red = _add_units("red",red_hexes)

        all_hexes = mapData.hexes()
        
        #build all hexes
        island_hexes = []
        for hex in all_hexes:
            if hex.terrain == "ocean":
                island_hexes.append(hex)
        for single_unit in unitData.units():
            island_hexes.remove(single_unit.hex)

        #pick one random hex to make the islamd seed
        island_seed = random.sample(island_hexes, 1)
        
        island_land_left_to_grow = number_of_islands - 1
        
        unit_hexes = []
        for unit_single in unitData.units():
            unit_hexes.append(unit_single.hex)

        #givin a starting hex find its neighbors, pick one then grow the island
        possible_island_hexes = []
        possible_island_hexes = map.getNeighborHexes(island_seed[0], mapData) 
        for single_hex in list(possible_island_hexes):
            if single_hex not in island_hexes:
                possible_island_hexes.remove(single_hex)
        
        island_seed[0].terrain = "land"
        
        
        dead_island_seed = False
        #get hexes of all unit squares
        if len(possible_island_hexes) <2:
            #print("not enough hexes to grow island")
            dead_island_seed = True
            pass

        if dead_island_seed == False:
            for x in range(0,island_land_left_to_grow):
                if(len(possible_island_hexes) == 0):
                    break
                #pick a random neighbor
                island_neighbors_target = random.sample(possible_island_hexes, 1)
                #add the neighbor to the island
                island_neighbors_target[0].terrain = "land"
                #get new neighbors
                temp_island_neighbors = map.getNeighborHexes(island_neighbors_target[0], mapData)
                for island in temp_island_neighbors:
                    if island in island_hexes:
                        possible_island_hexes.append(island)
                        

        #change every island in mapData to the terrain type "island"
#        for hex in mapData.hexes():
#            if hex in island_hexes:
#                hex.terrain = "land"

        count = count + 1
        if scenarioCycle:
            count %= scenarioCycle
        if scenarioCycle!=0 and count==0:
            random.seed(scenarioSeed)
        randstate = random.getstate()    
        score = {"maxPhases":max_phases,"lossPenalty":-1,"cityScore":0}
        random.setstate(priorstate)
        scenario = {"map":mapData.toPortable(), "units":unitData.toPortable(), "score":score}
        scenario["map"]["fogOfWar"] = fog_of_war
        last_scenario = scenario
        return scenario
    return inner


def blockad_run_simple_factory(size=6, min_units=1, max_units=3, num_blue_transports = 1, scenarioSeed=None, scenarioCycle=0, balance=False, max_phases=15, fog_of_war=False):
    #work around for gamedata not transfering through the atlatl
    
    #print(naval_utils.blue_goals)
    
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
        mapData.createOceanHexGrid(size,size)
        unitData = unit.UnitData()
        def _add_units(faction,start_hexes):
            n = random.randint(min_units,max_units)
            for i in range(n):
                hex = random.choice(start_hexes)
                start_hexes.remove(hex)
                u_param = {"hex":hex,"type":"destroyer","longName":str(i),"faction":faction,"currentStrength":100}
                unt = unit.Unit(u_param,unitData,mapData)
                #if(faction == "red"):
                #    print("start red hex: {} {}".format(hex.x_offset, hex.y_offset))
            return n
                
        #function added to support adding different unit type, maintained old one for backwards compatability
        def _add_units_dynamic(faction,start_hexes,unit_type,number):
            for i in range(number):
                hex = random.choice(start_hexes)
                start_hexes.remove(hex)
                #"unit type will be added, must match a unit combat_navy "
                u_param = {"hex":hex,"type":unit_type ,"longName":str(i),"faction":faction,"currentStrength":100}
                unt = unit.Unit(u_param,unitData,mapData)
                #if(faction == "red"):
                #    print("start red hex: {} {}".format(hex.x_offset, hex.y_offset))
            return number
        
        blue_side, red_side = random.choice((("north","south"),("south","north"),("east","west"),("west","east")))
        blue_hexes = get_setup_hex_ids(size,blue_side)
        red_hexes = get_setup_hex_ids(size,red_side)
        n_blue = _add_units("blue",blue_hexes)
        n_red = _add_units("red",red_hexes)
        n_blue_transport = _add_units_dynamic("blue",blue_hexes,"transport",num_blue_transports)
        
        
        #for the blocade running scenario, we want to make sure that the blue side has a transport that has a target position within the red side
        #get random red hex
        #shifted to using naval_utils as array holder, could not get to carry though in scenario variable
        #this does not change the map at all, for a visualization the hex type will need to be changed to somesort of "goal" hex
        naval_utils.clear_blue_goals()
        naval_utils.set_blue_goals(random.choice(red_hexes))
        
        #make a function that takes the goal hex to an array for portablility
        def _goal_toPortable(hexes):
            goal_array = []
            for hex in hexes:
                goal_array.append(mapData.hexIndex[hex].portableCopy())
            return goal_array
            

        count = count + 1
        if scenarioCycle:
            count %= scenarioCycle
        if scenarioCycle!=0 and count==0:
            random.seed(scenarioSeed)
        randstate = random.getstate()    
        score = {"maxPhases":max_phases,"lossPenalty":-1,"cityScore":0}
        random.setstate(priorstate)
        scenario = {"map":mapData.toPortable(), "units":unitData.toPortable(), "score":score}
        scenario["map"]["fogOfWar"] = fog_of_war
        last_scenario = scenario
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
    print( clear_ocean_factory(num_cities=0)() )