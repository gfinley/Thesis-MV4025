#combat range of the unit
# units will be destryoer, submarine, and transport ship
range = {
    "destroyer" : 2,
    "submarine" : 2,
    "transport" : 1
}

sight = {   
    "destroyer" : 1,
    "submarine" : 3,
    "transport" : 1
}

#variable for the probability of detecting an enemy unit

ineffectiveThreshold = 0.25 # 50%

firepower_scaling = 0.50


#firepower matrixes for the units against other ubnits
firepower = {}
firepower["destroyer"] = {
    "destroyer" : 1.0,
    "submarine" : 1.0,
    "transport" : 0.5
}
firepower["submarine"] = {
    "destroyer" : 1.0,
    "submarine" : 1.0,
    "transport" : 0.5
}
firepower["transport"] = {
    "destroyer" : 0.5,
    "submarine" : 0.5,
    "transport" : 1.0
}

#defensive firepower matrixes for the units against other ubnits
##RESERVED FOR FUTURE USE
defensivefp = {}
defensivefp["destroyer"] = {
    "destroyer" : 0,
    "submarine" : 0,
    "transport" : 0
}
defensivefp["submarine"] = {
    "destroyer" : 0,
    "submarine" : 0,
    "transport" : 0
}
defensivefp["transport"] = {
    "destroyer" : 0,
    "submarine" : 0,
    "transport" : 0
}

#terrain effects
terrain_multiplier = {}
terrain_multiplier["destroyer"] = {
    "ocean" : 1.0,
    "land" : 0.0
}
terrain_multiplier["submarine"] = {
    "ocean" : 1.0,
    "land" : 0.0
}
terrain_multiplier["transport"] = {
    "ocean" : 1.0,
    "land" : 0.0
}
