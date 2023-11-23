import json

# Opening JSON file
f = open('/home/philip.relton/projects/MLOverlaps/data/runfiles/MULTIDETECTOR/MULTIDETECTOR_test_signal_catalog.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)
  
# Iterating through the json
# list

print(len(data))
print(data.keys())