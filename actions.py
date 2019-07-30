from keras.models import Sequential
import numpy as np
import random



class Action:
    action = None

    #all_actions = [np.append(np.array([], dtype=Vote), item) for item in product(np.array([Vote.BUY, Vote.SELL, Vote.HOLD]), repeat=2)]


    all_actions = np.array(["b_more_A", "b_more_B", "b_A_h_B", "h_A_b_B", "b_A_s_B", "s_A_b_B", "h_A_h_B", "h_A_s_B", "s_A_h_B", "s_A_s_B"])


    def __init__(self, model: Sequential, epsilon: float, current_state, batch_size):
        self.batch_size = batch_size
        self.model = model
        self.epsilon = epsilon
        self.current_state = current_state
        return None

    # gets random action or greedy action based on epsilon
    def get_action(self):

        if random.random() < self.epsilon:
            action = self.get_random_action()

        else:
            action = self.get_greedy_action(self.current_state)

        return action

    # randomly chooses a action from all actions
    def get_random_action(self):
        random_action_indx = np.random.randint(0, len(self.all_actions))
        action = self.all_actions[random_action_indx]
        return action

    # gets action with hiqhest Q-Value
    def get_greedy_action(self, current_state):
        current_state = np.expand_dims(current_state, axis=1).reshape((1, len(current_state)))
        max_action_indx = np.argmax(self.model.predict(current_state))
        action = np.vstack(self.all_actions)[max_action_indx]
        return action

