import numpy as np
import random


class Agent(object):
    """ Generic Reinforcement learner. Will use Q learning, dyna Q, and some custom additions.
    (Initially based on the template for the Machine Learning for Trading course, by Tucker Balch)"""

    def __init__(self,
                 num_states,
                 num_actions,
                 alpha=0.2,
                 gamma=0.9,
                 random_actions_rate=0.9,
                 random_actions_decrease=0.999,
                 dyna_iterations=0,
                 verbose=False,
                 name='NN'):

        self.verbose = verbose

        # Dimensions of the problem
        self.num_states = num_states
        self.num_actions = num_actions

        # Parameters
        self.alpha = alpha
        self.gamma = gamma
        self.random_actions_rate = random_actions_rate
        self.random_actions_decrease = random_actions_decrease
        self.dyna_iterations = dyna_iterations

        # Initialization
        self.s = 0
        self.a = 0
        self.Q = 1.0 - 2*np.random.rand(num_states, num_actions)
        # QExplore keeps track of how many times the (s,a) pair was visited (with Q update)
        self.QExplore = np.ones((num_states, num_actions))
        # T and R for the hallucination models
        # Probabilities of transition
        self.T = 0.00001*np.ones((num_states, num_actions, num_states))
        # Expected immediate reward
        self.R = 1.0 - 2*np.random.rand(num_states, num_actions)
        self.name = name

    def random_action(self, s, actions=None):
        """
        This function chooses a random action, but not uniformly.
        It addresses the problem that a totally random exploration is very slow.
        So it keeps track of the explored (state,action) pairs and looks for new things to do.

        :param s: the current state
        :param actions: A list of possible actions
        :return: action
        """

        if actions is None:
            actions = range(self.num_actions)

        probas = 1/self.QExplore[s, actions]
        # Normalize
        probas /= np.sum(probas)

        action = np.random.choice(actions, p=probas)
        # action = random.randint(0, self.num_actions-1)

        return action

    def choose_action(self, s):
        """
        Chooses an action. With "random_actions_rate" probability it returns a random action.
        If it doesn't, then it returns the best option from the Q table.
        It doesnt' update the Q table nor the random_actions_rate variable.

        :param s: is the current state
        :return: action
        """
        do_explore = (random.random() < self.random_actions_rate)
        if do_explore:
            action = self.random_action(s)
        else:
            actions = range(self.num_actions)
            max_q = np.max(self.Q[s])

            # Now, get all the actions that have Q == maxQ
            optimal_actions = []
            for action_temp in actions:
                if self.Q[s, action_temp] == max_q:
                    optimal_actions.append(action_temp)

            # Choose one of the optimal choices, at random
            # (I could use the QExplore to choose also...)
            action = random.choice(optimal_actions)

        return action

    def hallucinate(self, s):
        # Initialize the hallucinating states and actions (the real ones shouldn't change)
        # Should hallucinations be more random?? To test later...
        # h_radom_actions_rate = self.random_actions_rate
        h_s = s
        for i in range(self.dyna_iterations):
            # Get new action
            h_a = self.choose_action(h_s)

            # Simulate transitions and rewards
            transition_probas = self.T[h_s, h_a, :]/np.sum(self.T[h_s, h_a, :])
            h_s_prime = np.random.choice(range(self.num_states), p=transition_probas)
            h_r = self.R[h_s, h_a]

            # Update Q
            # Get the best Q for h_s'
            max_q_prime = np.max(self.Q[h_s_prime])

            # Now use the formula to update Q
            self.Q[h_s, h_a] = (1-self.alpha)*self.Q[h_s, h_a] + \
                self.alpha*(h_r + self.gamma * max_q_prime)

            # Update the state
            h_s = h_s_prime

    def play_learned_response(self, new_state):
        """
        This function does the same as "play", but without updating the Q table. Given a new state, it chooses an action
        according to the best learned policy, so far.
        It does update the state.
        :param new_state: The resulting state for the previous action, or the state that was externally set.
        :returns: The chosen action
        """

        # Choose an action
        action = self.choose_action(new_state)

        # Update the state and action
        self.s = new_state
        self.a = action

        if self.verbose:
            print("s =", new_state, "a =", action)
        return action

    def play(self, reward, new_state):
        """
        Given a new state, and a reward for the previous action,
        chooses an action, updating the Q table in the process.
        :param new_state: The resulting state for the previous action.
        :param reward: The reward for the previous action.
        :returns: The chosen action.
        """

        # Update Q ------------------------------------------
        # Get the best Q for s'
        maxQprime = np.max(self.Q[new_state])

        # Now use the formula to update Q
        self.Q[self.s, self.a] = (1-self.alpha)*self.Q[self.s, self.a] + \
            self.alpha*(reward + self.gamma * maxQprime)

        # Hallucinate some experience...
        # Update T
        self.T[self.s, self.a, new_state] += 1
        # Update R
        self.R[self.s, self.a] = (1-self.alpha)*self.R[self.s, self.a] + self.alpha * reward
        # Hallucinate!
        self.hallucinate(new_state)
        # End of Update Q -----------------------------------

        # Choose an action and then update random_action_rate (!)
        action = self.choose_action(new_state)
        self.random_actions_rate *= self.random_actions_decrease

        # Update the state and action
        self.s = new_state
        self.a = action

        # Update QExplore
        self.QExplore[new_state, action] += 1.0

        # Print some debugging messages
        if self.verbose:
            print("s = {}  a = {}  reward = {}".format(new_state, action, reward))

        return action

    def __str__(self):
        return self.name

    __repr__ = __str__
