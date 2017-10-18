"""
Test a Q Learner in a navigation problem.  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
import time
from recommender.agent import Agent

aleas = True

# print out the map
def printmap(data):
    print("-"*data.shape[1]*3)
    for row in range(0, data.shape[0]):
        line_str = ""
        for col in range(0, data.shape[1]):
            if data[row,col] == 0:
                line_str += "   "
            if data[row,col] == 1:
                line_str += " o "
            if data[row,col] == 2:
                line_str += " * "
            if data[row,col] == 3:
                line_str += " X "
            if data[row,col] == 4:
                line_str += " . "
            if data[row,col] == 5:
                line_str += " S "
        print(line_str)
    print("-"*data.shape[1]*3)

# find where the robot is in the map
def getrobotpos(data):
    R = -999
    C = -999
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 2:
                C = col
                R = row
    if (R+C)<0:
        print("warning: start location not defined")
    return R, C

# find where the goal is in the map
def getgoalpos(data):
    R = -999
    C = -999
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 3:
                C = col
                R = row
    if (R+C)<0:
        print("warning: goal location not defined")
    return R, C

# move the robot according to the action and the map
def movebot(data,oldpos,a):
    testr, testc = oldpos

    if(aleas):
        # Add randomness
        # p= 0.8 execute "a"
        # p= 0.1 go left to "a" direction
        # p= 0.1 go right to "a" direction
        p = [0.8, 0.1, 0.1]
        if a == 0: #north
            choices = [0, 1, 3]
            rand_a = np.random.choice(choices,p=p)
        elif a == 1: #east
            choices = [1, 0, 2]
            rand_a = np.random.choice(choices,p=p)
        elif a == 2: #south
            choices = [2, 1, 3]
            rand_a = np.random.choice(choices,p=p)
        elif a == 3: #west
            choices = [3, 0, 2]
            rand_a = np.random.choice(choices,p=p)
    else:
        rand_a = a

    # update the test location
    if rand_a == 0: #north
        testr = testr - 1
    elif rand_a == 1: #east
        testc = testc + 1
    elif rand_a == 2: #south
        testr = testr + 1
    elif rand_a == 3: #west
        testc = testc - 1

    # see if it is legal. if not, revert
    if testr < 0: # off the map
        testr, testc = oldpos
    elif testr >= data.shape[0]: # off the map
        testr, testc = oldpos
    elif testc < 0: # off the map
        testr, testc = oldpos
    elif testc >= data.shape[1]: # off the map
        testr, testc = oldpos
    elif data[testr, testc] == 1: # it is an obstacle
        testr, testc = oldpos

    return (testr, testc) #return the new, legal location


# convert the location to a single integer
def discretize(pos):
    return pos[0]*10 + pos[1]

# run the code to test a learner
if __name__ == "__main__":

    verbose = False  # print lots of debug stuff if True

    # read in the map
    inf = open('testworlds/worldTest.csv')
    data = np.array([list(map(float, s.strip().split(','))) for s in inf.readlines()])
    originalmap = data.copy() #make a copy so we can revert to the original map later

    startpos = getrobotpos(data) #find where the robot starts
    goalpos = getgoalpos(data) #find where the goal is

    if verbose:
        printmap(data)

    rand.seed(5)

    #num_states=100
    learner = Agent(num_states=100,
                    num_actions=4,
                    random_actions_rate=0.98,
                    random_actions_decrease = 0.999,  # 0.9999,
                    verbose=verbose,
                    dyna_iterations=20) #initialize the learner

    #each iteration involves one trip to the goal
    #num_iter = 10000
    num_iter = 500
    for iteration in range(0,num_iter):
        steps = 0
        data = originalmap.copy()
        robopos = startpos
        state = discretize(robopos) #convert the location to a state
        action = learner.play_learned_response(state) #set the state and get first action
        while robopos != goalpos:

            # move to new location according to action and then get a new action
            newpos = movebot(data, robopos, action)
            if newpos == goalpos:
                r = 1  # reward for reaching the goal
            elif data[newpos[0], newpos[1]] == 5:
                r = -100
            else:
                r = -1  # negative reward for not being at the goal
            state = discretize(newpos)
            action = learner.play(r, state)

            data[robopos] = 4  # mark where we've been for map printing
            data[newpos] = 2  # move to new location
            robopos = newpos  # update the location
            if verbose:
                printmap(data)
            if verbose:
                time.sleep(1)
            steps += 1

        print(iteration, ",", steps)
        #print("rar: ",learner.rar)
    printmap(data)
