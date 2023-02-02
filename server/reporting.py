
from PIL import Image, ImageDraw
import math
from datetime import datetime
import json
import time

#draws a box of size x_size, y_size, that is shaded in input_color, writes unit_name, and unit_type, and final_health in the box
def draw_unit_box(x, y, x_size, y_size, unit_name, unit_type, final_health, input_color, draw):
    x1 = x 
    x2 = x + x_size
    y1 = y 
    if y1 < 0:
        y1 = 0
    y2 = y + y_size
    draw.rectangle((x1, y1, x2, y2), fill=input_color, outline='black')
    middle_x = (x1+x2)/2
    middle_y = (y1+y2)/2
    draw.text((middle_x-25,middle_y-10), unit_name, fill='black', )
    draw.text((middle_x-25,middle_y), unit_type, fill='black')
    draw.text((middle_x-25,middle_y+10), final_health, fill='black')

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
        self.combat_history = []
        self.action_count = 0

        #version 2 of the historian tapped into the unit.toportable function
        self.all_game_history = []
        self.unit_current_status = {}
        self.move_number = 0
        self.frame = []
        self.frames = []

        #toggle for on on off
        self.on = False


    def reset(self):
        self.mapData = {}
        self.map_X = -1
        self.map_Y = -1
        self.game_history = []
        self.red_units = []
        self.blue_units = []
        self.unit_path = {}
        self.image = make_blank_image((1000, 1000))
        self.combat_history = []
        self.action_count = 0

        #version 2 of the historian tapped into the unit.toportable function
        self.all_game_history = []
        self.unit_current_status = {}
        self.move_number = 0
        self.frame = []
        self.frames = []

    #function not used
    #function takes the mapData and compares each unit x_offset and y_offset in mapData.unitIndex 
    #to the unit_path dictionary to see if the unit has moved, if so adds to the unit_path
    def update_unit_path(self, mapData):
        if self.on == False:
            return

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
        if self.on == False:
            return        
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
        if self.on == False:
            return        #get the unit id from the move
        unit_id = combat['source']

        x_src, y_src = self.unit_path[unit_id][-1]
        #get the destination from the move
        destination = combat['target']
        #get the x and y of the destination
        x_tgt, y_tgt = self.unit_path[destination][-1]
        self.combat_history.append([x_src, y_src, x_tgt, y_tgt])
        #add the x and y to the unit_path


    def handel_move(self, message):
        if self.on == False:
            return
      #check to see if game is running
        #TODO add check to see if game is running
        #check to see if the message is type 'observation'
        #print(message)

        #check to see if message is a dict
        if type(message) != dict:
            #load message as a dict
            message_json = json.loads(message)
        else:
            message_json = message


        #catch to misc messages first
            #pass, its '{"type": "action", "action": {"type": "pass"}}'
            #gym-pause '{"type": "gym-pause"}'
            #role-request '{"type": "role-request", "role": "red"}'
        if message_json['type'] == 'action':
            if message_json['action']['type'] == 'pass':
                pass
        if message_json['type'] == 'gym-pause':
                pass
        if message_json['type'] == 'role-request':
                pass
        #end of non-historian messages

        #start historian game messages
        # {'type': 'move', 'mover': 'blue 0', 'destination': 'hex-1-3'}
        if message_json['type'] == 'move':
            self.update_move(message_json)
            pass
        if message_json['type'] == 'fire':
            self.update_combat(message_json)
            pass
        if message_json['type'] == 'action':
            if message_json['action']['type'] == 'move':
                self.update_move(message_json['action'])
                pass
            if message_json['action']['type'] == 'fire':
                self.update_combat(message_json['action'])
                pass
        
        #check to see it the game is over
        if ('reset'  in message_json) or ('end' in message_json):
            self.process_game()

        
    
        pass
    
    #function that processed all the moves and draws lines onto map
    def process_game(self):
        if self.on == False:
            return
        size = 50 #this is my standard for development
        max_canvas_size = 1000 #canvas size set to 1000 for development
        x_box_size = 50
        y_box_size = 50
        for unit in self.unit_path:
            path = self.unit_path[unit]
            unit_color = unit.split(' ')[0]
            draw_line(path, unit_color,  ImageDraw.Draw(self.image))
        for unit in self.red_units:
            x, y = self.unit_path[unit][-1]
            x,y = get_center(x, y, 50)
            draw_hexagon((x, y), 20, 'red', ImageDraw.Draw(self.image))
        for unit in self.blue_units:
            x, y = self.unit_path[unit][-1]
            x,y = get_center(x, y, 50)
            draw_hexagon((x, y), 20, 'blue', ImageDraw.Draw(self.image))
        #draw hexes for end of game units

        for combat in self.combat_history:
            x_src, y_src, x_tgt, y_tgt = combat
            draw_line([(x_src, y_src), (x_tgt, y_tgt)], 'black', ImageDraw.Draw(self.image))
        self.save_image()
        self.reset()
    def update(self):
        if self.on == False:
            return
        self.game_history.append(self.game)
        self.red_units.append(self.game.red_units)
        self.blue_units.append(self.game.blue_units)
        self.unit_path[self.game.red_units] = self.game.red_units.path
        self.unit_path[self.game.blue_units] = self.game.blue_units.path

    #function to make game unit status boxes
    def frame_unit_report(self):
        frame = make_blank_image((500, 500))
        #get the location of unit spot
        #get number of boxes that will fit
        max_canvas_size = 500
        x_start = 0
        x_box_size = 100
        y_box_size = 100
        num_boxes = int((max_canvas_size - x_start) / x_box_size)
        ii = 0
        yy = 0
        all_units = self.red_units + self.blue_units
        for unit in all_units:
            #check to see if unit survied
            #unit status
            status = self.unit_current_status[unit]
            final_life = status[1]
            if ii == num_boxes:
                ii = 0
                yy += 1
            x = x_start + ii * x_box_size
            y = yy * y_box_size
            unit_color = unit.split(' ')[0]
            #change blue to cornflowerblue and red to salmon
            if unit_color == 'blue':
                unit_color = 'cornflowerblue'
            if unit_color == 'red':
                unit_color = 'salmon'
            draw_unit_box(x, y,
                x_size=x_box_size,
                y_size=y_box_size, 
                unit_name=unit, 
                unit_type="Inf",
                input_color=str(unit_color),
                final_health=str(final_life),
                draw = ImageDraw.Draw(frame)
            )
            ii += 1
        return frame
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
        if self.on == False:
            return
        for loc in self.mapData.hexIndex:
            working_data = self.mapData.hexIndex[loc]
            x, y = working_data.x_offset, working_data.y_offset
            draw_hexagon(get_center(x, y, 50), 50, 'black', ImageDraw.Draw(self.image))
            if working_data.terrain != 'clear':
                draw_hexagon(get_center(x, y, 50), 50, 'black', ImageDraw.Draw(self.image),fill='grey')
        return

    #function for rendering a frame of the game
    def render_frame(self, unitData):
        if self.on == False:
            return
        if self.red_units == []:
            return make_blank_image((500, 500))
        image = make_blank_image((500, 500))
        for loc in self.mapData.hexIndex:
            working_data = self.mapData.hexIndex[loc]
            x, y = working_data.x_offset, working_data.y_offset
            draw_hexagon(get_center(x, y, 50), 50, 'black', ImageDraw.Draw(image))
            if working_data.terrain != 'clear':
                draw_hexagon(get_center(x, y, 50), 50, 'black', ImageDraw.Draw(image),fill='grey')
        for unit in self.unit_current_status:
            #check to see if unit is alive
            if self.unit_current_status[unit][2] != True :
                x, y = int(self.unit_current_status[unit][0][0]), int(self.unit_current_status[unit][0][1])
                #get the name of the unit
                name = unit.split(' ')[0]
                if name == 'red':
                    draw_hexagon(get_center(x, y, 50), 20, 'red', ImageDraw.Draw(image))
                if name == 'blue':
                    draw_hexagon(get_center(x, y, 50), 20, 'blue', ImageDraw.Draw(image))
        return image



    def handle_unit_to_portable(self, unit_info):
        if self.on == False:
            return
        #input is an array of "{'type': 'infantry', 'faction': 'blue', 'longName': '0', 'currentStrength': 100, 'hex': 'hex-3-0', 'canMove': True, 'ineffective': False, 'detected': False}"
        
        #a "move move" is { 'type': 'move', 'name' : 'name', 'start': "[x,y]", 'end': "[x,y]"}
        #a "damage move" is { 'type': 'damage', 'name' : 'name', 'life': 'life', "ineffective": "ineffective"}
        #get list of all the units
        all_units = self.red_units + self.blue_units
        unit_data = {}
        #need dict of unit life and position
        for unit in unit_info:
            #get all the unit info
            name = unit['faction'] + " " +  unit['longName']
            strength = unit['currentStrength']
            try:
                x, y = unit['hex'].split('-')[1:]
            except:
                print("error: unit hex was not in correct format")
                print(unit['hex'])
                x, y = 0, 0
            ineffective = unit['ineffective']

            #create a status
            status = [ [x,y], strength, ineffective]

            if name not in self.unit_current_status.keys():
                self.unit_current_status[name] = status

            if name not in all_units:
                print("something went wrong unit was not in historians known units")
                pass
            else:
                #check if unit status is different
                if self.unit_current_status[name] != status:
                    #something changed with this unit
                    old_status = self.unit_current_status[name]
                    if old_status[0] != status[0]:
                        print("debug: unit moved")
                        #unit moved
                        move = [ {'type': 'move', 'name' : name, 'start': old_status[0], 'end': status[0]}, self.move_number]
                        self.move_number += 1
                        self.unit_current_status[name] = status
                        self.all_game_history.append(move)
                    if old_status[1] != status[1]:
                        #unit took damage
                        print("debug: unit took damage")
                        damage = [ {'type': 'damage', 'name' : name, 'life': status[1], "ineffective": status[2]}, self.move_number]
                        self.move_number += 1
                        self.unit_current_status[name] = status
                        self.all_game_history.append(damage)   
        #make a frame image of the position of all of the units on the hex map
        frame = self.render_frame(unit_info)
        self.frames.append(frame)
        return

    def save_image(self, path='../AARs/'):
        if self.on == False:
            return
        #get current timestamp
        timestr = time.strftime("%Y%m%d-%H%M%S")
        path = path + str(timestr) + '.png'
        print("game saved at {}".format(path))
        save_image(self.image, path)
        self.save_gif(path)


    def save_gif(self,path):
        if self.on == False:
            return
        if self.frames == []:
            print("no frames to save")
            return
        #turn all the frames from self.frame into a gif
        #for every frame paste it into the game replay report
        new_frames = []
        for frame in self.frames:
            new_frame = self.image.copy()
            new_frame.paste(frame, (0,500))
            #get the frame for the unit status
            status_frame = self.frame_unit_report()
            new_frame.paste(status_frame, (499,0))

            new_frames.append(new_frame)

        path = path[:-4] + '.gif'
        frame_one = new_frames[0]
        frame_one.save(path, format='GIF', append_images=new_frames[1:], save_all=True, duration=100, loop=0)
        print("gif saved")


    #function looks at all unit data and sees if there was a changin position or life, the creates an array of all the changes
    


historian = Reporting()


