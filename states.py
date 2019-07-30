from framework.portfolio import Portfolio
from framework.company import Company
from framework.stock_market_data import StockMarketData
from framework.vote import Vote
from actions import Action
import numpy as np
from framework.order import Order, OrderType

class State:

    def __init__(self, actual_portfolio: Portfolio, stock_market_data: StockMarketData):
        self.stock_market_data = stock_market_data
        self.actual_portfolio = actual_portfolio
        return None

    # creates one hot vector based on the experts votes for the NN input
    def experts_vote_NN(self, experts_vote_a, experts_vote_b, state_values):

        if experts_vote_a == Vote.BUY and experts_vote_b == Vote.BUY:
            state_values.append([1, 0, 0, 0, 0, 0, 0, 0, 0])

        elif experts_vote_a == Vote.BUY and experts_vote_b == Vote.SELL:
            state_values.append([0, 1, 0, 0, 0, 0, 0, 0, 0])

        elif experts_vote_a == Vote.BUY and experts_vote_b == Vote.HOLD:
            state_values.append([0, 0, 1, 0, 0, 0, 0, 0, 0])

        elif experts_vote_a == Vote.SELL and experts_vote_b == Vote.BUY:
            state_values.append([0, 0, 0, 1, 0, 0, 0, 0, 0])

        elif experts_vote_a == Vote.SELL and experts_vote_b == Vote.SELL:
            state_values.append([0, 0, 0, 0, 1, 0, 0, 0, 0])

        elif experts_vote_a == Vote.SELL and experts_vote_b == Vote.HOLD:
            state_values.append([0, 0, 0, 0, 0, 1, 0, 0, 0])

        elif experts_vote_a == Vote.HOLD and experts_vote_b == Vote.BUY:
            state_values.append([0, 0, 0, 0, 0, 0, 1, 0, 0])

        elif experts_vote_a == Vote.HOLD and experts_vote_b == Vote.SELL:
            state_values.append([0, 0, 0, 0, 0, 0, 0, 1, 0])

        elif experts_vote_a == Vote.HOLD and experts_vote_b == Vote.HOLD:
            state_values.append([0, 0, 0, 0, 0, 0, 0, 0, 1])

    def normalize(self, array):
        for i in range(len(array)):
            array[i] = np.divide(array[i], np.sum(array))

        return array

    # gets state as Array for NN input, includes portfolio value, stock prices and one hot vector of experts vote
    def get_actual_state_for_NN(self, expert_a, expert_b):

        state_values = []
        stock_data_a = self.stock_market_data[Company.A]
        experts_vote_a = expert_a.vote(stock_data_a)

        stock_data_b = self.stock_market_data[Company.B]
        experts_vote_b = expert_b.vote(stock_data_b)

        state_values.append(self.actual_portfolio.get_value(self.stock_market_data)) # portfolio
        state_values.append(stock_data_a.get_last()[-1]) # stock price A
        state_values.append(stock_data_b.get_last()[-1]) # stock price B
        self.experts_vote_NN(experts_vote_a, experts_vote_b, state_values) # one hot vector


        return self.normalize(np.hstack(state_values))


    def buy(self, stock_data, company, order_list, portfolio, percent):
        stock_price = stock_data.get_last()[-1]
        amount_to_buy = int(np.divide(portfolio.cash*percent, stock_price))

        if amount_to_buy > 0:
            order_list.append(Order(OrderType.BUY, company, amount_to_buy))

    def sell(self, company, order_list, portfolio):
        amount_to_sell = portfolio.get_stock(company)
        if amount_to_sell > 0:
            order_list.append(Order(OrderType.SELL, company, amount_to_sell))


    # creating the order list based on the choosed action
    def get_order_list_new(self, action: Action, stock_market_data, portfolio):
        order_list = []

        all_actions = np.array(
            ["b_more_A", "b_more_B", "b_A_h_B", "h_A_b_B", "b_A_s_B", "s_A_b_B", "h_A_s_B", "s_A_h_B",
             "s_A_s_B", "h_A_h_B"])

        assert np.isin(action, all_actions)

        stock_data_a = stock_market_data[Company.A]
        stock_data_b = stock_market_data[Company.B]

        if action == "b_more_A":
           self.buy(stock_data_a, Company.A, order_list, portfolio, 0.9)
           self.buy(stock_data_b, Company.B, order_list, portfolio, 0.1)

        elif action == "b_more_B":
            self.buy(stock_data_a, Company.A, order_list, portfolio, 0.1)
            self.buy(stock_data_b, Company.B, order_list, portfolio, 0.9)

        elif action == "b_A_h_B":
            self.buy(stock_data_a, Company.A, order_list, portfolio, 1)

        elif action == "h_A_b_B":
            self.buy(stock_data_b, Company.B, order_list, portfolio, 1)

        elif action == "b_A_s_B":
            self.buy(stock_data_a, Company.A, order_list, portfolio, 1)
            self.sell(Company.B, order_list, portfolio)

        elif action == "s_A_b_B":
            self.sell(Company.A, order_list, portfolio)
            self.buy(stock_data_b, Company.B, order_list, portfolio, 1)

        elif action == "h_A_s_B":
            self.sell(Company.B, order_list, portfolio)

        elif action == "s_A_h_B":
            self.sell(Company.A, order_list, portfolio)

        elif action == "s_A_s_B":
            self.sell(Company.A, order_list, portfolio)
            self.sell(Company.B, order_list, portfolio)

        elif action == "h_A_h_B":
            order_list = []

        else:

            print("action not found !")


        return order_list




