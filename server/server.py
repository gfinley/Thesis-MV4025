from gameserver import GameServer
from game import Game
import json
import random
import airegistry
import argparse
import scenario
import game_dispenser
from scenario_gen_reg import scenario_generator_registry

import current_game_access

import asyncio

# def generic_client(role, inS):
#     inO = json.loads(inS)
#     if 'type' in inO and inO['type']=='parameters':
#         return json.dumps( {'type':'role-request', 'role':role} )
#     # inO is a game state
#     obs = inO['observation']
#     if game.is_terminal(obs):
#         return None
#     elif obs.observation.status.onMove==role:
#         return json.dumps( {'type':'action', 'action':random.choice(game.legal_actions(obs))} )
# def blue_client(inS):
#     return generic_client("blue", inS)
# def red_client(inS):
#     return generic_client("red", inS)

#global game, server, gym_ai

def init(args):
    #global game, server, gym_ai
    global server, gym_ai
    client_functions = []
    if args.blueAI:
        if not args.blueAI in airegistry.ai_registry:
            raise Exception(f'blueAI with name {args.blueAI} not found in AI registry')
        constructor, kwargs = airegistry.ai_registry[args.blueAI]
        if args.blueNeuralNet:
            kwargs["neuralNet"] = args.blueNeuralNet
        ai = constructor("blue", kwargs)
        if args.blueAI in {"gym", "gymx2", "gym12", "gym13", "gym14","gym15","ray"}:
            gym_ai = ai
        client_functions.append(ai.process)
        
    if args.redAI:
        if not args.redAI in airegistry.ai_registry:
            raise Exception(f'redAI with name {args.redAI} not found in AI registry')
        constructor, kwargs = airegistry.ai_registry[args.redAI]
        if args.redNeuralNet:
            kwargs["neuralNet"] = args.redNeuralNet
        ai = constructor("red", kwargs)
        if args.redAI == "gym" or args.redAI == "gymx2":
            gym_ai = ai
        client_functions.append(ai.process)
        
    if args.scenario[-4:]==".scn":
        scenario_generator = scenario.from_file_factory(args.scenario)
    else:
        constructor, kwargs = scenario_generator_registry[args.scenario]
        if args.scenarioSeed:
            kwargs['scenarioSeed'] = args.scenarioSeed
        if args.scenarioCycle:
            kwargs['scenarioCycle'] = args.scenarioCycle
        scenario_generator = constructor(**kwargs)
    game_disp = game_dispenser.ScenarioGeneratorGameDispenser(scenario_generator)

    server = GameServer(game_disp, client_functions, open_socket=args.openSocket, verbose=args.v, red_log=args.redReplay, blue_log=args.blueReplay, n_reps=args.nReps)
    current_game_access.set_gameserver(server)
    server.run()

        
########### Gym emulation support ############
    
def mapDimensionBackdoor():
    return server.game.mapData.getDimensions()

def getGymAI():
    return gym_ai

def reset():
    #addMessageRunLoop({"type":"reset-request"})
    addMessageRunLoop({"type":"next-game-request"})
    return getGymAI().observation()

def addMessageRunLoop(messageO):
    getGymAI().sendToServer(messageO)
    asyncio.get_event_loop().run_forever()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument('-v', action='store_true')
    parser.add_argument("--redAI")
    parser.add_argument("--blueAI")
    parser.add_argument("--blueReplay")
    parser.add_argument("--redReplay")
    parser.add_argument("--openSocket", action='store_true')
    parser.add_argument("--exitWhenTerminal", action='store_true')
    parser.add_argument("--scenarioSeed", type=int)
    parser.add_argument("--scenarioCycle", type=int)
    parser.add_argument("--nReps", type=int, default=0)
    parser.add_argument("--redNeuralNet")
    parser.add_argument("--blueNeuralNet")
    args = parser.parse_args()
    init(args)
