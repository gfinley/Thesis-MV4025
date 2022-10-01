import scenario
import scenario_navy

scenario_generator_registry = {
    "clear-inf-6" : (scenario.clear_square_factory, {'size':6, 'min_units':2, 'max_units':4}),
    "clear-inf-5" : (scenario.clear_square_factory, {'size':5, 'min_units':2, 'max_units':4}),
    "city-inf-5" : (scenario.clear_square_factory, {'size':5, 'min_units':2, 'max_units':4, 'num_cities':1}),
    "city-inf-5-bal" : (scenario.clear_square_factory, {'size':5, 'min_units':2, 'max_units':4, 'num_cities':1, 'balance':True}),
    "fog-inf-7" : (scenario.clear_square_factory, {'size':7, 'min_units':1, 'max_units':4, 'num_cities':0, 'max_phases':15, 'fog_of_war':True}),
    "test-inf-1500" : (scenario.clear_square_factory, {'size':1500, 'min_units':20, 'max_units':30, 'num_cities':5, 'max_phases':150, 'fog_of_war':False}),
    "fog-inf-200" : (scenario.clear_square_factory, {'size':200, 'min_units':20, 'max_units':30, 'num_cities':0, 'max_phases':250, 'fog_of_war':True}),
}