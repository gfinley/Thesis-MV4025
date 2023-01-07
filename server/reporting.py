
from PIL import Image, ImageDraw
import math
from datetime import datetime



#function that draws a hexagon with a given center and radius
def draw_hexagon(center, radius, color, draw, fill=None):
    x, y = center
    points = []
    for i in range(6):
        angle_deg = 60 * i
        angle_rad = math.pi / 180 * angle_deg
        points.append((x + radius * math.cos(angle_rad), y + radius * math.sin(angle_rad)))
    draw.polygon(points, fill=fill, outline=color)

#draws a line through given coordinantes
def draw_line(path, color, draw):
    for move in range(len(path)-1):
        start = path[move]
        start = get_center(start[0], start[1], 50)
        end = path[move+1]
        end = get_center(end[0], end[1], 50)
        draw.line((start, end), fill=color, width=5)

#function that makes a blank image with a given size
def make_blank_image(size):
    return Image.new('RGB', size, color='white')

def save_image(image, path):
    image.save(path)


#funetion to get x,y center of hexagon from x,y of top left corner
def get_center(x, y, size):
    x_distance = 3/2*size
    y_distance = math.sqrt(3)*size

    x_start = size
    y_start = .5*math.sqrt(3)*size
    if x % 2 == 0: #even row
        hor_distance = x_start + (x)*x_distance
        ver_distance = y_start + (y)*y_distance
    if x % 2 == 1:
        hor_distance = size+(x_distance)  + (x-1)*x_distance
        ver_distance = y_distance + (y)*y_distance

    return hor_distance, ver_distance

#function that draws a box, colors it, adds text inside it at a given center
def draw_box(center, size, text, color, draw):
    x, y = get_center(center[0], center[1], size)
    x1 = x - size
    x2 = x + size
    y1 = y - size
    y2 = y + size
    draw.rectangle((x1, y1, x2, y2), fill=color)
    middle_x = (x1+x2)/2
    middle_y = (y1+y2)/2

    draw.text((x1,middle_y), text, fill='black')

#draw a 5 pointed star betweeen two points
def draw_star(center1, center2, radius, size, color, draw):
    x1, y1 = get_center(center1[0], center1[1], size)
    x2, y2 = get_center(center2[0], center2[1], size)
    mid_x = (x1+x2)/2
    mid_y = (y1+y2)/2
    x = mid_x
    y = mid_y
    points = []
    for i in range(10):
        angle_deg = 36 * i
        angle_rad = math.pi / 180 * angle_deg
        points.append((x + radius * math.cos(angle_rad), y + radius * math.sin(angle_rad)))
    draw.polygon(points, fill="red", outline=color)



class Reporting:
    def __init__(self):
        self.mapData = {}
        self.map_X = -1
        self.map_Y = -1
        self.game_history = []
        self.red_units = []
        self.blue_units = []
        self.unit_path = {}
        self.image = make_blank_image((1000, 1000))

    def reset(self):
        self.mapData = {}
        self.map_X = -1
        self.map_Y = -1
        self.game_history = []
        self.red_units = []
        self.blue_units = []
        self.unit_path = {}
        self.image = make_blank_image((1000, 1000))

    #function not used
    #function takes the mapData and compares each unit x_offset and y_offset in mapData.unitIndex 
    #to the unit_path dictionary to see if the unit has moved, if so adds to the unit_path
    def update_unit_path(self, mapData):
        for unit in mapData.unitIndex:
            working_data = mapData.unitIndex[unit]
            name = working_data.longName
            faction = working_data.faction
            type = working_data.type
            uniqueID = working_data.uniqueId

            x_start = working_data.hex.x_offset
            y_start = working_data.hex.y_offset
            if self.unit_path[uniqueID][-1] != [x_start, y_start]:
                self.unit_path[uniqueID].append([x_start, y_start])

    #take input "{'type': 'move', 'mover': 'blue 2', 'destination': 'hex-1-1'}" and update the move
    def update_move(self, move):
        #get the unit id from the move
        unit_id = move['mover']
        #get the destination from the move
        destination = move['destination']
        #get the x and y of the destination
        x, y = destination.split('-')[1:]
        #add the x and y to the unit_path
        self.unit_path[unit_id].append([int(x), int(y)])
    
    #function takes "{'type': 'fire', 'source': 'blue 2', 'target': 'red 3'}" and handels it
    def update_combat(self, combat):
        #get the unit id from the move
        unit_id = combat['source']
        #get the destination from the move
        destination = combat['target']
        #get the x and y of the destination
        x, y = destination.split('-')[1:]
        #add the x and y to the unit_path
        self.unit_path[unit_id].append([int(x), int(y)])

    def handel_move(self, message):
        #check to see if game is running
        #TODO add check to see if game is running
        #check to see if the message is type 'observation'
        #print(message)
        pass
    
    #function that processed all the moves and draws lines onto map
    def process_game(self):
        for unit in self.unit_path:
            path = self.unit_path[unit]
            unit_color = unit.split(' ')[0]
            draw_line(path, unit_color,  ImageDraw.Draw(self.image))
        self.save_image()
        self.reset()
    def update(self):
        self.game_history.append(self.game)
        self.red_units.append(self.game.red_units)
        self.blue_units.append(self.game.blue_units)
        self.unit_path[self.game.red_units] = self.game.red_units.path
        self.unit_path[self.game.blue_units] = self.game.blue_units.path

    #function for adding a unit to the unit list
    def add_unitData(self, unitData):
        for unit in unitData.unitIndex:
            working_data = unitData.unitIndex[unit]
            #get all the data
            name = working_data.longName
            faction = working_data.faction
            type = working_data.type
            uniqueID = working_data.uniqueId

            x_start = working_data.hex.x_offset
            y_start = working_data.hex.y_offset

            #add the starting position for the unit to unit path
            if uniqueID not in self.unit_path:
                self.unit_path[uniqueID] = [[x_start, y_start]]

            if faction == 'red':
                self.red_units.append(uniqueID)
            if faction == 'blue':
                self.blue_units.append(uniqueID)
        return
    
    #function for adding mapData
    def add_mapData(self, mapData):
        self.mapData = mapData
        #look at each element in the map and find the max x and y after spliting name on "-"
        for loc in mapData.hexIndex:
            working_data = mapData.hexIndex[loc]
            x, y = working_data.x_offset, working_data.y_offset
            if x > self.map_X:
                self.map_X = x
            if y > self.map_Y:
                self.map_Y = y
        return
    
    #function for rendering the start of a game
    def render_start(self):
        for loc in self.mapData.hexIndex:
            working_data = self.mapData.hexIndex[loc]
            x, y = working_data.x_offset, working_data.y_offset
            draw_hexagon(get_center(x, y, 50), 50, 'black', ImageDraw.Draw(self.image))
            if working_data.terrain != 'clear':
                draw_hexagon(get_center(x, y, 50), 50, 'black', ImageDraw.Draw(self.image),fill='grey')
        for unit in self.red_units:
            x, y = self.unit_path[unit][0]
            x,y = get_center(x, y, 50)
            draw_hexagon((x, y), 20, 'red', ImageDraw.Draw(self.image))
        for unit in self.blue_units:
            x, y = self.unit_path[unit][0]
            x,y = get_center(x, y, 50)
            draw_hexagon((x, y), 20, 'blue', ImageDraw.Draw(self.image))
        return

    def save_image(self, path='../AARs/'):
        dt = datetime.datetime.now()
        path = path + str(dt)
#        save_image(self.image, path)



historian = Reporting()




#make the image
#image = make_blank_image((1000, 1000))

#draw a hexagon on the image
#for x in range(5):
#    for y in range(5):
#        x_center, y_center = get_center(x, y, 50)
#        #print("\t", x_center, y_center)
#        draw_hexagon((x_center, y_center), 50, 'black', ImageDraw.Draw(image))

#save the image


#unit1_line = [(0,0), (1,1), (2,2), (3,3), (4,4)]
#unit2_line = [(0,4), (1,3), (2,2), (3,1), (4,0)]
#draw_line(unit1_line, 'red', ImageDraw.Draw(image))
#draw_line(unit2_line, 'blue', ImageDraw.Draw(image))
#draw_box((6,0), 50, 'Unit ones Path', 'cornflowerblue', ImageDraw.Draw(image))

#draw_star((0,0), (0,1), 5, 50, 'black', ImageDraw.Draw(image))

#save_image(image, '../AARs/test.png')