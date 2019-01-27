import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class DynamicAllocation(object):
    def __init__(self):
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
        df = pd.read_csv('prices.csv', sep=';', index_col='Date')
        return df

    def _pct_change(self):
        """
        :return: returns pct_change of the price series
        """
        df = DynamicAllocation._prices_data(self).pct_change()
        return df

    def _create_dfs(self):
        df = DynamicAllocation._pct_change(self)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        # Creates df that stores alloc decisions
        self.df_alter_alloc = pd.DataFrame(
            0, index=df.index, columns=df.columns)

        weights = DynamicAllocation._set_initial_asset_weights(self)
        self.df_portfolio_weights = pd.DataFrame(
            weights, index=df.index, columns=df.columns)

    def _partial_cummulative_return(self):
        df_pct_change = DynamicAllocation._pct_change(self)
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
                # print('if')
                # Basically, pass
                final_init_row = final_init_row + 1

            else:
                # print('else')
                # case a cum variations of an asset(s)
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
        # TODO: adicionar derivada como velocidade de acelarcao para alocacao de ativos

        self.df_portfolio_weights.loc[row] = self.df_portfolio_weights.loc[row] + \
                                         self.allocation_step_change * self.df_alter_alloc.loc[row]
        # TODO double check marretada
        self.df_portfolio_weights.loc[row] = self.df_portfolio_weights.loc[row] / self.df_portfolio_weights.loc[row].sum()
        # In case we are short in all assets
        # if self.df_portfolio_weights.loc[row, ~self.df_portfolio_weights]:
        # In case we are long long in all assets
        self.df_portfolio_weights.loc[row:, :] = self.df_portfolio_weights.loc[row].values

    def performance(self):
        df_pct_change = DynamicAllocation._pct_change(self)

        df_pct_change.dropna(inplace=True)
        DynamicAllocation._partial_cummulative_return(self)
        # Calculate performance
        df_asset_performance = self.df_portfolio_weights * df_pct_change.shift(periods=1)
        # clean data
        df_asset_performance.dropna(inplace=True)
        df_portfolio_performance = df_asset_performance.sum(axis=1)
        # Returns
        return df_asset_performance, df_portfolio_performance

    def naive_performance(self):
        df_pct_change = DynamicAllocation._pct_change(self)
        weights = DynamicAllocation._set_initial_asset_weights(self)
        df_weights = pd.DataFrame(
            weights, index=df_pct_change.index, columns=df_pct_change.columns)

        df_naive_asset_performance = df_weights * df_pct_change
        df_naive_asset_performance = df_naive_asset_performance.shift(periods=1)
        df_naive_portfolio_performance = df_naive_asset_performance.sum(axis=1)
        return df_naive_asset_performance, df_naive_portfolio_performance

    def main(self):
        # DynamicAllocation._prices_data(self)
        # DynamicAllocation._set_initial_asset_weights(self)
        # DynamicAllocation._pct_change(self)
        # DynamicAllocation._partial_cummulative_return(self)
        # DynamicAllocation.performance(self)
        # DynamicAllocation.naive_performance(self)
        DynamicAllocation.analyse_results(self)

    def analyse_results(self):
        df_asset_performance, df_portfolio_performance = DynamicAllocation.performance(self)
        df_naive_asset_performance, df_naive_portfolio_performance = DynamicAllocation.naive_performance(self)
        # print(df_naive_portfolio_performance)
        df_final = pd.DataFrame(df_portfolio_performance, columns=['DynamicPortfolio'])
        df_final['NaivePortfolio'] = df_naive_portfolio_performance.loc[df_final.index.values]


        sns.set()
        ax1 = sns.lineplot(x="Date", y="NaivePortfolio", data=df_final)
        ax2 = sns.lineplot(x="Date", y="DynamicPortfolio", data=df_final)
        plt.show()


if __name__ == '__main__':
    model = DynamicAllocation()
    model.main()





