from profolio.commission import Commission_System

class Profolio():
    """
    Simulate a real world stock profolio
    """
    def __init__(self, tickers, cash):
        self.shares = {ticker:0 for ticker in tickers}
        self.tickers = tickers
        self.system = Commission_System()
        self.holding = set() #set of int
        self.cash = cash  #float

    def calculate_net_worth(self,stock_price):
        net_worth = self.cash
        for idx in self.holding:
            ticker = self.tickers[idx]
            net_worth += self.shares[ticker]*stock_price[idx]
        
        return round(net_worth,4)

    def buy(self, buy_list, stock_price):
        """
        Args:
            buy_list : numpy array shape (50,) | list
            stock_price: numpy array shape (N_stocks,)
        Return:
            total_trading_fee: total trading fee by buying stocks
        """
        num_stocks = len(buy_list)
        total_trading_fee = 0

        if num_stocks > 0:
            amount_per_stock = self.cash/num_stocks
        else:
            return total_trading_fee
        
        for idx in buy_list:
            ticker = self.tickers[idx]
            price = stock_price[idx]
            shares, trading_fee = self.system.buy(price,amount_per_stock)

            self.shares[ticker] = shares
            self.holding.add(idx)
            total_trading_fee += trading_fee

        self.cash = 0
        total_trading_fee = round(total_trading_fee,4)
        return total_trading_fee
        
    def sell(self, sell_list, stock_price):
        """
        Args:
            sell_list : numpy array shape (50,) | list
            stock_price: numpy array shape (N_stocks,)
        Return:
            total_trading_fee: total trading fee by buying stocks
        """
        #rmb to edit the self.holding, self.cash, self.shares
        num_stocks = len(sell_list)
        total_trading_fee = 0

        if num_stocks == 0:
            return total_trading_fee
        
        for idx in sell_list:
            ticker = self.tickers[idx]
            shares = self.shares[ticker]
            price = stock_price[idx]
            received, trading_fee = self.system.sell(price,shares)

            self.shares[ticker] = 0
            self.holding.remove(idx)
            self.cash += received
            total_trading_fee += trading_fee

        self.cash = round(self.cash,4)
        total_trading_fee = round(total_trading_fee,4)
        return total_trading_fee
        
    def sell_and_buy(self,selected_stock_idx,stock_price):
        #sell
        sell_list = []
        for idx in self.holding:
            if idx in selected_stock_idx:
                continue
            else:
                sell_list.append(idx)

        self.sell(sell_list,stock_price)

        #buy
        buy_list = []
        for idx in selected_stock_idx:
            if idx in self.holding:
                continue
            else:
                buy_list.append(idx)
        
        self.buy(buy_list,stock_price)
