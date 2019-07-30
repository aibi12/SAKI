import random
from collections import deque
from typing import List
import numpy as np
import stock_exchange
from experts.obscure_expert import ObscureExpert
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from framework.order import Company
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger
from states import State
from actions import Action
import copy


class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = False,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader', ):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """
        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.train_while_trading = train_while_trading


        # Parameters for neural network
        self.state_size = 12
        self.action_size = 10
        self.hidden_size = 50

        # Parameters for deep Q-learning
        self.gamma = 1
        self.learning_rate = 0.001
        self.learning_rate_min = 0.0001
        self.learning_rate_decay = 0.9995
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.1
        self.decay_epsilon = False
        self.batch_size = 64
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=2000)

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None
        self.last_action_a = None
        self.last_action_b = None
        self.last_action = None
        self.last_portfolio_value = None

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.get_name())
            logger.info(f"DQL Trader: Loaded trained model")
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, kernel_initializer='normal', activation='relu'))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])


    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    '''
    def trade_expert(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:

        self.experts_vote_a = self.expert_a.vote(stock_market_data[Company.A])
        self.experts_vote_b = self.expert_b.vote(stock_market_data[Company.B])

        if self.experts_vote_a == Vote.BUY and self.experts_vote_b == Vote.BUY:
            action = "b_more_A"

        elif self.experts_vote_a == Vote.BUY and self.experts_vote_b == Vote.SELL:
            action = "b_A_s_B"

        elif self.experts_vote_a == Vote.BUY and self.experts_vote_b == Vote.HOLD:
            action = "b_A_h_B"

        elif self.experts_vote_a == Vote.SELL and self.experts_vote_b == Vote.BUY:
            action = "s_A_b_B"

        elif self.experts_vote_a == Vote.SELL and self.experts_vote_b == Vote.SELL:
            action = "s_A_s_B"

        elif self.experts_vote_a == Vote.SELL and self.experts_vote_b == Vote.HOLD:
            action = "s_A_h_B"

        elif self.experts_vote_a == Vote.HOLD and self.experts_vote_b == Vote.BUY:
            action = "h_A_b_B"

        elif self.experts_vote_a == Vote.HOLD and self.experts_vote_b == Vote.SELL:
            action = "h_A_s_B"

        elif self.experts_vote_a == Vote.HOLD and self.experts_vote_b == Vote.HOLD:
            action = "h_A_h_B"

        state = State(portfolio, stock_market_data)
        order_list = state.get_order_list_new(action, stock_market_data, portfolio)

        return order_list
    '''

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"

        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """

        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]



        # TODO Compute the current state
        state = State(portfolio, stock_market_data)
        current_state = state.get_actual_state_for_NN(self.expert_a, self.expert_b)
        reward = self.get_reward(current_state, self.last_state)
        self.s = len(current_state) - self.state_size

        # TODO Train Model
        if self.train_while_trading is True:
            # TODO Store state as experience (memory) and train the neural network only if trade() was called before at least once
            if len(self.memory) == 2000:
                self.memory.popleft()
            if len(self.memory) >= self.min_size_of_memory_before_training:
                # train model
                actual, target = self.get_actual_and_target(self.batch_size, self.memory, self.model)

                self.model.fit(actual, target, batch_size=self.batch_size)
                self.decay_epsilon = True


            # TODO Create actions for current state and decrease epsilon for fewer random actions
            action = Action(self.model, self.epsilon, np.array(current_state[self.s:]), self.batch_size).get_action()
            print(self.epsilon, self.learning_rate)

            if (self.epsilon > self.epsilon_min and self.decay_epsilon):
                self.epsilon *= self.epsilon_decay
            #elif self.epsilon < self.epsilon_min:
            #    self.epsilon = 0.2


            if (self.learning_rate > self.learning_rate_min and self.decay_epsilon):
                self.learning_rate *= self.learning_rate_decay
            elif self.learning_rate < self.learning_rate_min: # reset the learning rate to its default value
                self.learning_rate = 0.001

        else:
            action = Action(self.model, 0, np.array(current_state[self.s:]), self.batch_size).get_action()


        order_list = state.get_order_list_new(action, stock_market_data, portfolio)


        # TODO Save created state, actions and portfolio value for the next call of trade()
        if self.last_state is not None and self.last_action is not None:
            self.memory.append([self.last_state, self.last_action, reward, current_state])
        self.last_state = copy.copy(current_state)
        self.last_action = copy.copy(action)


        return order_list



    def get_actual_and_target(self, batch_size, memory, model):

        # get random batch of experiences from memory
        memory = np.array(memory)


        random_indx = random.randint(0, len(memory)-self.batch_size)


        experiences = memory[random_indx : self.batch_size + random_indx]


        # get actual state input for NN
        actual_states = np.vstack(np.array(experiences[:, 0]))[:, self.s:]

        # get actions
        actions = np.vstack(np.array(experiences[:, 1]))
        action_indx = self.get_action_indx_new(actions, Action.all_actions)

        # get rewards
        rewards = np.vstack(np.array(experiences[:, 2]))

        # get states after performing actions
        new_states = np.vstack(np.array(experiences[:, 3]))[:, self.s:]

        # forward pass for the current states
        target_Q = model.predict(actual_states, batch_size=batch_size)


        # forward pass for the new state
        target_new_state = np.expand_dims(self.gamma*np.max(model.predict(new_states, batch_size=batch_size), axis=1), axis=1)

        target_Q_action = np.add(rewards, target_new_state)
        target_Q_action = target_Q_action.reshape((batch_size, 1))
        target_Q_action = np.squeeze(target_Q_action)


        target_Q[action_indx] = target_Q_action


        return actual_states, target_Q


    def get_action_indx_new(self, actions, all_actions):
        action_indx = []
        for action in actions:
            action_indx.append(action == all_actions)

        return np.where(action_indx)


    def get_reward(self, current_state, last_state):
        reward = 0

        if last_state is not None:
            if current_state[0] > last_state[0]:
                reward = np.divide(current_state[0]-last_state[0], last_state[0])

            else:
                reward = -1

        else:
            print(0)

        return reward



# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 5


if __name__ == "__main__":
    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # Create the stock exchange and one traders to train the net
    stock_exchange = stock_exchange.StockExchange(10000.0)
    training_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True)

    # Save the final portfolio values per episode
    final_values_training, final_values_test = [], []

    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")

        # train the net
        stock_exchange.run(training_data, [training_trader])
        training_trader.save_trained_model()
        final_values_training.append(stock_exchange.get_final_portfolio_value(training_trader))

        # test the trained net
        testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
        stock_exchange.run(testing_data, [testing_trader])
        final_values_test.append(stock_exchange.get_final_portfolio_value(testing_trader))

        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])
    plt.show()

    plt.figure()
    #plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])
    plt.show()