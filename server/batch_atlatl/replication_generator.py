import json
import copy
import random
import sys

# Example list specification: 
#      "name" : "param_A", "levels" : [1, 7, 11]
# Example even spacing specification: 
#     "name" : "param_B", "begin" : 2, "end" : 10, "count" : 5

# Users should only change design_str
# seed, deep, algorithm, normed, residuals
design_str = """
{
	"replications" : 1,
	"factors" : 
		[
			{
				"name" : "deep",
				"levels" : [ false, true ]
			},
      {
				"name" : "algorithm",
				"levels" : [ "PPO", "A2C" ]
			},
      {
				"name" : "normed",
				"levels" : [ false, true ]
			},
      {
				"name" : "residuals",
				"levels" : [ false, true ]
			}
		]
}
"""

def designPoints( designO ):
  points = []
  factors = designO["factors"]
  for factor in factors:
    if "levels" in factor.keys():
      continue
    if factor["count"] == 1:
      step = 0
    else:
      step = ( factor["end"] - factor["begin"] ) / ( factor["count"] - 1 )
    values = [ ]
    value = factor["begin"]
    for i in range(factor["count"]):
      values.append( value )
      value += step
    factor["levels"] = values
  level_indices = []
  for factor in factors:
    level_indices.append( 0 )
  done = False
  while  not done:
    point = {}
    for i in range(len(factors)):
      name = factors[i]["name"]
      point[name] = factors[i]["levels"][ level_indices[i] ]
    points.append( point )
    
    cf = 0
    level_indices[cf] += 1
    while ( level_indices[cf] == len(factors[cf]["levels"]) ):
      level_indices[cf] = 0
      cf += 1
      if cf == len(factors):
        done = True
        break
      level_indices[cf] += 1
  return points


design = json.loads( design_str )

replications = design["replications"]

design_points = designPoints( design )

# Create replications and add run IDs and seeds
maxsize =  2147483647
replicationA = []
id = 0
for point in design_points:
  for i in range(replications):
    rep = copy.copy(point)
    rep["seed"] = random.randint(0, maxsize)
    rep["id"] = id
    id += 1
    replicationA.append( rep )

# Print one replication per line
print("[")
for i in range(len(replicationA)):
	replication_json = json.dumps( replicationA[i] )
	if i < len(replicationA)-1:
		print( replication_json + "," )
	else:
		print( replication_json )
print("]")


