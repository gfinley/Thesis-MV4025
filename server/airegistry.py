IMPORT_NEURAL = True

import ai.passive
import ai.random_actor
import ai.shootback
import ai.potential_field
import ai.dijkstra_demo
import ai.mcts
import ai.gym_ai_surrogate
import ai.pass_agg
import ai.pass_agg_fog
import ai.burt_reynolds_lab2




if IMPORT_NEURAL:
    import ai.neural
    import ai.neural_ray

ai_registry = {
              "passive" : (ai.passive.AI, {}),
              "random" : (ai.random_actor.AI, {}),
              "shootback" : (ai.shootback.AI, {}),
              "field" : (ai.potential_field.AI, {}),
              "pass-agg" : (ai.pass_agg.AI, {}),
              "pass-agg-fog" : (ai.pass_agg_fog.AI, {}),
              "dijkstra" : (ai.dijkstra_demo.AI, {}),

              "mcts1k" : (ai.mcts.AI, {"max_rollouts":1000, "debug":False}),
              "mcts10k" : (ai.mcts.AI, {"max_rollouts":10000, "debug":False}),
              "mctsd" : (ai.mcts.AI, {"max_rollouts":10000, "debug":True}),

              "gym" : (ai.gym_ai_surrogate.AI, {}),
              "gymx2" : (ai.gym_ai_surrogate.AIx2, {}),
              "gym12" : (ai.gym_ai_surrogate.AITwelve, {}),
              "gym13" : (ai.gym_ai_surrogate.AI13, {}),
              "gym14" : (ai.gym_ai_surrogate.AI14, {}),
              "gym15" : (ai.gym_ai_surrogate.AI15, {}),
              
              "burt-reynolds-lab2" : (ai.burt_reynolds_lab2.AI, {}),

              "NAVY_SIMPLE" : (ai.gym_ai_surrogate.NAVY_SIMPLE, {}),
              "ray" : (ai.gym_ai_surrogate.AI_RAY, {}),
             }
             
if IMPORT_NEURAL:
    ai_registry["neural"] = (ai.neural.AI, {"doubledCoordinates":False})
    ai_registry["cnn"] = (ai.neural.AI, {"doubledCoordinates":True})
    ai_registry["hex12"] = (ai.neural.AITwelve, {"doubledCoordinates":False})
    ai_registry["hex13"] = (ai.neural.AI13, {"doubledCoordinates":False})
    ai_registry["hex14"] = (ai.neural.AI14, {"doubledCoordinates":False})
    ai_registry["hex14dqn"] = (ai.neural.AI14, {"dqn":True, "doubledCoordinates":False})
    ai_registry["mando-fun-lab3"] = (ai.neural.AI14, {"neuralNet":"ai/mandofun_c0.zip", "dqn":True, "doubledCoordinates":False}),
    ai_registry["mod_1"] = (ai.neural.AI15, {"neuralNet":"ai/model_dqn4_500000.zip", "dqn":True, "doubledCoordinates":False}),
    ai_registry["mod_2"] = (ai.neural.AI15, {"neuralNet":"ai/model_mod_2_500000.zip", "dqn":True, "doubledCoordinates":False})
    ai_registry["mod_4"] = (ai.neural.AI15, {"neuralNet":"ai/model_mod_4_2000000.zip", "dqn":True, "doubledCoordinates":False})
    ai_registry["mod_5"] = (ai.neural.AI15, {"neuralNet":"ai/model_mod_5_2000000.zip", "dqn":True, "doubledCoordinates":False})
    ai_registry["mod_6"] = (ai.neural.AI15, {"neuralNet":"ai/model_mod_6_3000000.zip", "dqn":True, "doubledCoordinates":False})
    ai_registry["mod_7"] = (ai.neural.AI15, {"neuralNet":"ai/model_mod_7_2000000.zip", "dqn":True, "doubledCoordinates":False})
    ai_registry["mod_8"] = (ai.neural.AI15, {"neuralNet":"ai/model_mod_8_2000000.zip", "dqn":True, "doubledCoordinates":False})
    ai_registry["mod_9"] = (ai.neural.AI15, {"neuralNet":"ai/model_mod_9_3000000.zip", "dqn":True, "doubledCoordinates":False})

    ai_registry["Ray_ai_test"] = (ai.neural_ray.NAVY_SIMPLE, {"neuralNet":"ai/model.pt", "dqn":True, "doubledCoordinates":False})
    ai_registry["Navy"] = (ai.neural_ray.NAVY_SIMPLE, {"neuralNet":"ai/model.pt", "dqn":False, "doubledCoordinates":False})
     


    

