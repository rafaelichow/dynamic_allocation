import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from dateutil import parser
import context
import sys
import os


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class DynamicAllocation(object):
    def __init__(self):
        self.initial_portfolio_value = 1000000.0
        self.return_limit = 0.06
        self.band_limit = 0.08
        self.min_asset_weight = -0.1
        self.max_asset_weight = 0.25
        self.allocation_step_change = 0.025
        self.initial_weights = None
        self.risk_free_return = 0.065 # TODO model forward Curve

    def _set_initial_asset_weights(self):
        number_of_assets = len(DynamicAllocation._prices_data(self).columns)
        self.initial_weights = 1.0 / number_of_assets
        return self.initial_weights

    def _prices_data(self):
        ## TODO: CHANGE READ CSV
        df = pd.read_csv('../data/prices.csv', sep=';', index_col='Date')
        return df

    def _pct_change(self):
        """
        :return: returns pct_change of the price series
        """
        df = self._prices_data().pct_change()
        return df

    def _create_dfs(self):
        df = self._pct_change()
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        # Creates df that stores alloc decisions
        self.df_alter_alloc = pd.DataFrame(
            0, index=df.index, columns=df.columns)

        weights = self._set_initial_asset_weights()
        self.df_portfolio_weights = pd.DataFrame(
            weights, index=df.index, columns=df.columns)

    def _partial_cummulative_return(self):
        df_pct_change = self._pct_change()
        # stores dates
        date_series = df_pct_change.index
        # Cleans df_pct_change
        df_pct_change.dropna(inplace=True)
        df_pct_change.reset_index(drop=True, inplace=True)
        DynamicAllocation._create_dfs(self)
        # Initial DF subset
        rel_init_row = 0
        # Final DF subsert
        final_init_row = 0
        # Loops over events
        while final_init_row <= df_pct_change.shape[0]:
            # Cuummulative sum
            df_subset = df_pct_change.loc[rel_init_row:final_init_row].cumsum().reset_index(drop=True)
            # Conditional
            if not any(df_subset.loc[df_subset.shape[0] - 1].abs() > self.return_limit):
            # if not any(df_subset.loc[df_subset.shape[0] - 1].abs() > self.band_limit):
                # Basically, pass
                final_init_row = final_init_row + 1

            else:
                # case a cummulative variations of an asset(s)
                # exceeds the imposed limit
                vals = df_subset.loc[df_subset.shape[0] - 1]
                # -1 == sell more
                # 0 == hold
                # 1 ++ buy more
                decision = [-1 if val < -self.return_limit else 1 if val > self.return_limit else 0 for val in vals]
                # Compute the decision in df alter allocation decision
                DynamicAllocation._alter_asset_allocation(self, final_init_row, decision)
                # Reset relative initial row
                rel_init_row = final_init_row
                # sum 1 to final relative initial row
                final_init_row = final_init_row + 1
        # Fixes indexing
        self.df_alter_alloc.index = date_series[1:]
        self.df_portfolio_weights.index = date_series[1:]

        return self.df_alter_alloc

    def _alter_asset_allocation(self, row, decision):
        # Updates decision table
        self.df_alter_alloc.loc[row] = decision
        # Updates asset allocation pct table
        self.df_portfolio_weights.loc[row] = self.df_portfolio_weights.loc[row] + \
                                         self.allocation_step_change * self.df_alter_alloc.loc[row]
                            
        self.df_portfolio_weights.loc[row] = self.df_portfolio_weights.loc[row] / self.df_portfolio_weights.loc[row].sum()
        self.df_portfolio_weights.loc[row:, :] = self.df_portfolio_weights.loc[row].values

    def performance(self):
        df_pct_change = self._pct_change()

        df_pct_change.dropna(inplace=True)
        DynamicAllocation._partial_cummulative_return(self)
        # Calculate performance
        df_asset_performance = self.df_portfolio_weights * df_pct_change.shift(periods=1)
        # clean data
        df_asset_performance.dropna(inplace=True)
        # Sums columns
        df_portfolio_performance = df_asset_performance.sum(axis=1)
        # Returns
        return df_asset_performance, df_portfolio_performance

    def naive_performance(self):
        df_pct_change = self._pct_change()
        weights = self._set_initial_asset_weights()
        df_weights = pd.DataFrame(
            weights, index=df_pct_change.index, columns=df_pct_change.columns)

        df_naive_asset_performance = df_weights * df_pct_change
        df_naive_asset_performance = df_naive_asset_performance.shift(periods=1)
        df_naive_portfolio_performance = df_naive_asset_performance.sum(axis=1)
        return df_naive_asset_performance, df_naive_portfolio_performance

    def cummulative_performance(self, df):
        """
        return: cummulative performace given a return series
        """
        # adds 1 to the df
        df = df + 1
        # makes cummulative product
        df = df.cumprod()
        
        return df


    def analyse_results(self):
        df_asset_performance, df_portfolio_performance = self.performance()
        df_naive_asset_performance, df_naive_portfolio_performance = self.naive_performance()
        # print(df_naive_portfolio_performance)
        df_final = pd.DataFrame(df_portfolio_performance, columns=['DynamicPortfolio'])
        df_final['NaivePortfolio'] = df_naive_portfolio_performance.loc[df_final.index.values]


        df_final = self.cummulative_performance(df=df_final)
    
        print(df_final)
        df_final = df_final.reset_index()

        # # sns.set()
        # ax1 = sns.lineplot(x="Date", y="NaivePortfolio", data=df_final)
        # ax2 = sns.lineplot(x="Date", y="DynamicPortfolio", data=df_final)


        df_final.plot()
        plt.show()


    @staticmethod
    def markovitz_formula():
        pass

    @staticmethod
    def var():
        pass

    @staticmethod
    def sharpe(portfolio_return, portfolio_std_dev, risk_free_rate, number_of_days):
        # Cal
        accum_rf_ret = (1.0 + risk_free_rate) ** (number_of_days / 252) - 1
        # Calculates sharpe ratio
        sharpe = (portfolio_return - accum_rf_ret) / portfolio_std_dev

        return sharpe
        

    def bootstrap(self, backtest_step=10):

        df_asset_performance, df_portfolio_performance = self.performance()
        df_portfolio_performance = self.cummulative_performance(df_portfolio_performance)

        df_portfolio_performance.index = np.vectorize(parser.parse)(df_portfolio_performance.index)

        dates_array = []
        
        for i in range(0, len(df_portfolio_performance.index), backtest_step):
            dates_array.append(df_portfolio_performance.index[i])

        index_forward = dates_array
        index_backward = dates_array

        dict_returns = dict()

        for date_forward in index_forward:
            date_forward += pd.Timedelta(timedelta(days=backtest_step))

            for date_backward in reversed(index_backward):
                date_backward += pd.Timedelta(timedelta(days=-backtest_step))

                dict_outputs = dict()

                delta_days = (date_backward - date_forward).days

                if delta_days > 0:

                    df_port = df_portfolio_performance.loc[date_forward:date_backward]
                    ret = df_port.loc[df_port.index[-1]] / df_port.loc[df_port.index[0]] -1
                    sigma = df_port.std()
                    
                    dict_outputs['number_of_days'] = delta_days
                    dict_outputs['return'] = ret
                    dict_outputs['sigma'] = sigma
                    dict_outputs['sharpe'] = DynamicAllocation.sharpe(ret, sigma, self.risk_free_return, delta_days)

                    

                    key = "{}_{}".format(str(date_forward), str(date_backward))
                    dict_returns[key] = dict_outputs

                    print(dict_outputs)

        return dict_returns

    def main(self):
        # self.analyse_results()
        self.bootstrap()
    

if __name__ == '__main__':
    model = DynamicAllocation()
    model.main()





