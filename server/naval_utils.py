global blue_goals





blue_goals = []

def get_blue_goals():
    global blue_goals
    return blue_goals

def set_blue_goals(goals):
    global blue_goals
    blue_goals = goals
    
def clear_blue_goals():
    global blue_goals
    blue_goals = []