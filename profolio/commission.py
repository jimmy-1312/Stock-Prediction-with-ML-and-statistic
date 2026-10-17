class Commission_System():
    """
    This system simulated the real world trading under the tiered--階梯式 pricing rule from IB(Interactive Broker)
    For official commission fee calculation, please refer to: https://www.interactivebrokers.com.hk/cn/pricing/commissions-stocks.php?region=americas
    """
    def __init__(self):
        """
        Fee Initialization
        """
        #Commission Fee 佣金
        self.commission_fee = 0.0035 #per shares (buy | sell)
        self.minimum_commission_fee = 0.35 #每筆訂單最低佣金

        #Regulation Fee 監管費用
        self.FINRA_trail_fee = 0.000003 #per shares (buy | sell)
        self.SEC_transaction_fee = 0.0000206 #per shares*price (sell)
        
        #Exchange Fees 交易所費用
        self.exchange_fee = 0.003 #per shares (buy | sell) --> we assume the order is always 市價單,in real world the fee will be lower

        #Clearing Fees 清算費用
        self.clearing_fee = 0.0002 #per shares (buy | sell)


    def buy(self, price, value):
        """
        Args:
            price: the price of stock per shares
            value: the total value I am spending in this order
        Return:
            shares: the shares I got
            trading fee: the trading fee for this order

        Equations:
            value = max(minimum_commission_fee, commission_fee*shares) + FINRA_trail_fee*shares + exchange_fee*shares + clearing_fee*shares + price*shares
                  = max(minimum_commission_fee, commission_fee*shares) + (FINRA_trail_fee+exchange_fee+clearing_fee+price)*shares
            
            c1 = minimum_commission_fee, c2 = commission_fee, c3 = FINRA_trail_fee+exchange_fee+clearing_fee+price
            value = max(c1,c2*shares) + c3*shares
        
        --> if value < c1*(c2+c3)/c2
                shares = (value-c1)/c3
            if value >= c1*(c2+c3)/c2
                shares = value/(c2+c3)
        """
        c1 = self.minimum_commission_fee
        c2 = self.commission_fee
        c3 = self.FINRA_trail_fee + self.exchange_fee + self.clearing_fee + price

        if value < c1*(c2+c3)/c2:
            shares = (value - c1)/c3
        else:
            shares = value/(c2+c3) 
        trading_fee = value - shares*price

        trading_fee = round(trading_fee,4)
        shares = round(shares,4)

        return shares,trading_fee

    def sell(self, price, shares):
        """
        Args:
            stock_price: float, the price of stock per shares
            shares: the number of shares i want to sell
        Return:
            received: the cash i received by selling these shares
            trading fee: the trading fee for this order
        Equations:
            fee = max(minimum_commission_fee, commission_fee*shares) + FINRA_trail_fee*shares + SEC_transaction_fee*shares*price + exchange_fee*shares + clearing_fee*shares
                = max(minimum_commission_fee, commission_fee*shares) + (FINRA_trail_fee+SEC_transaction_fee*price+exchange_fee+clearing_fee+price)*shares
            received = shares*price - fee
        """
        commission_fee = max(self.minimum_commission_fee,self.commission_fee*shares)
        regulation_fee = self.FINRA_trail_fee*shares + self.SEC_transaction_fee*shares*price
        exchange_fee = self.exchange_fee*shares
        clearing_fee = self.clearing_fee*shares

        trading_fee = commission_fee + regulation_fee + exchange_fee + clearing_fee
        received = shares*price - trading_fee

        trading_fee = round(trading_fee,4)
        received = round(received,4)

        return received,trading_fee