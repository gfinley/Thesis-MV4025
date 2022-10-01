import math

Infinity = math.inf

cost = {}

#navy units mobility cost
cost["destroyer"] = {
    "ocean": 100,
    "land": Infinity,
}
cost["submarine"] = {
    "ocean": 100,
    "land": Infinity,
}
cost["transport"] = {
    "ocean": 100,
    "land": Infinity,
}


#dont know what this is for
##TODO Figure this out
stackingLimit = 1