# Car driving simulator
# 
# The main parameter controlling this simulator is self.autopilot.
#
# If self.autopilot = "no", then the user can drive the blue car with the arrow keys:
# 	Up = Speed up
# 	Down = Slow down
# 	Left = Move left
# 	Right = Move right
#
# If self.autopilot = "yes", then the blue car is driven by the policy specified in self.policy_fname.
#
# Each line of self.policy_fname specifies the action distribution for one state.
# Each line is a string that has the following form:
# 	st;p1,p2,p3
# where st is the state, and pi is the probability of taking action i in state st.
#
# The state of the simulator is a string describing the positions of all the cars, and the speed of the blue car.
# Each state has the following form:
# 	[speed, blue_car_x, [red_car_x, red_car_y]]
# Where:
# 	speed = Speed of the blue car.
# 	blue_car_x = X-axis displacement of the blue car.
# 	red_car_x = X-axis displacement of the red car.
# 	red_car_y = Y-axis displacement of the red car.
# The initial state is "[1, 160]". It is the only state in which the red car does not appear.
#
# In autopilot mode, only three actions are allowed at any given time:
# 	In the initial state, the three actions correspond to selecting one of the three possible speeds.
# 	In all non-initial states, the three actions correspond to staying put, moving left, and moving right.
# So a policy can use just a single speed during a simulation. This simplification is convenient, and does not exclude any optimal policies.
#

from Tkinter import *
import random

class App:
    def __init__(self, master):

        frame = Frame(master)
        frame.pack()

        # Parameters
        self.h = 160 + 20
        self.w = 300
        self.interval = 100
	self.step = 5
        self.speeds_list = [20, 40, 80]
	self.speeds_names = ["Slow", "Med.", "Fast"]
        self.autopilot = "yes"
	self.policy_fname = "policy.txt"
        
        # Control
        self.speed = 1
        self.action = "none"
        self.policy = dict([])

        # Variables
        self.collisions = 0
        self.offroads = 0

        # Draw field of play
        h, w = self.h, self.w
        master.title("Car Game")
        master.geometry(str(w) + "x" + str(h) + "+100+100")        
        self.c = Canvas(frame, width=w, height=h, bg="#646464")
        self.left_grass = self.c.create_rectangle(0, 0, 2*w/5, h, fill="green")
        self.right_grass = self.c.create_rectangle(3*w/5, 0, w, h, fill="green")

        # Draw info area
        self.info_str = StringVar()
        self.update_info()
        info_label = Label(frame, textvariable=self.info_str, font=("Helvetica", "12"))
        self.c.create_window(w-5, 5, anchor=NE, window=info_label)

        # Draw my car
        self.my_car = self.c.create_rectangle((w/2)-10, h - 50, (w/2)+10, h - 10, fill="blue")

        # Bind event listeners
        self.c.bind("<KeyPress-Up>", self.on_up)
        self.c.bind("<KeyPress-Down>", self.on_down)
        self.c.bind("<KeyPress-Left>", self.on_left)
        self.c.bind("<KeyPress-Right>", self.on_right)

        if (self.autopilot == "yes"):
            self.read_policy(self.policy_fname)

        self.c.focus_force()
        self.c.pack()

	# Call the main function
	self.c.after(0, self.update_cars)

    # Event listeners
    def on_up(self, event):
        self.action = "up"
        
    def on_down(self, event):
        self.action = "down"
        
    def on_left(self, event):
        self.action = "left"
        
    def on_right(self, event):
        self.action = "right"

    # This function does most of the work  
    def update_cars(self):
        h, w = self.h, self.w
        speed = self.speeds_list[self.speed]
        x0 = self.c.coords(self.my_car)[0]        

        # Determine the current state
        state = []
        state.append(self.speed)
        state.append(int(round(self.c.coords(self.my_car)[2])))
        state = state + [[int(round(x)) for x in self.c.coords(car)[2:4]] for car in self.c.find_withtag("other_cars")]
        print str(state)

        # Have auto-pilot choose action
        if (self.autopilot == "yes"):
            if (str(state) in self.policy):
                self.action = self.select_action(state)
            else:
                self.action = "none"            
            print str(state), self.action

        # Move the other cars
        self.c.move("other_cars", 0, speed)

        # Handle the current action
        if (self.action == "left"):
            if (x0 > (2*self.w/5 - 20)):
                self.c.move(self.my_car, -self.step, 0)
        elif (self.action == "right"):
            if (x0 < 3*self.w/5):        
                self.c.move(self.my_car, +self.step, 0)
        elif (self.action == "up"):
            if (self.speed < 2):
                self.speed = self.speed + 1
        elif (self.action == "down"):
            if (self.speed > 0):
                self.speed = self.speed - 1

        self.action = "none"
        self.update_info()
        
        # Delete any cars that have moved below my car 
        [self.c.delete(car) for car in self.c.find_withtag("other_cars") if self.c.coords(car)[1] >= self.h - 10]

        # Generate a new car if not enough other cars 
        if (len(self.c.find_withtag("other_cars")) == 0):
            r = random.randrange(2*w/5, 3*w/5, 20)
            self.c.create_rectangle(r, -30, r+20, 10, fill="red", tags=("other_cars"))

        # Detect collisions and off-roads
        (w, x, y, z) = self.c.coords(self.my_car)
        w, x, y, z = w+1, x+1, y-1, z-1
        colliders = set(self.c.find_overlapping(w, x, y, z))
        if ((self.left_grass in colliders) | (self.right_grass in colliders)):
            self.offroads = self.offroads + 1
        colliders = colliders - set([self.my_car, self.left_grass, self.right_grass])
        if (len(colliders) > 0):
            self.collisions = self.collisions + 1
        self.update_info()

        # Reset the timer
        self.c.after(self.interval, self.update_cars)

    # Update the info panel
    def update_info(self):
        speed = self.speeds_list[self.speed]
        self.info_str.set("Collisions = " + str(self.collisions) + "\nOff-roads = " + str(self.offroads) + "\n\nSpeed = " + str(self.speeds_names[self.speed]))

    # Read in the policy from a text file
    def read_policy(self, fname):
        f = open(fname, 'r')
        for line in f:
            line = line.strip()
            [state, probs_string] = line.split(';')
            self.policy[state] = probs_string

    # Given a state, draw an action from the distribution specified by the policy
    # Note that the initial state "[1, 160]" is treated differently than the others
    def select_action(self, state):
        probs = [float(s) for s in self.policy[str(state)].split(',')]
	if (str(state) == "[1, 160]"):
		action_list = ["none", "down", "up"]
	else:
        	action_list = ["none", "left", "right"]
        cum_probs = [reduce(lambda a, b: a+b, probs[0:i]) for i in range(1, 4)]
        r = random.random()
        indices = filter(lambda i: r < cum_probs[i], range(0, 3))
        return action_list[indices[0]]

root = Tk()
app = App(root)
root.mainloop()
