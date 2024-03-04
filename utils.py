import pandas as pd
import numpy as np
import os
import re
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from stargazer.stargazer import Stargazer, LineLocation
from IPython.display import display, HTML

plotly_colors = px.colors.qualitative.Plotly


def plot_r2(pca_res, pca_expl):
    # plot pca results and save to pdf
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pca_res.index, y=pca_res['r2_in'], name="IS"))
    fig.add_trace(go.Bar(x=pca_res.index, y=pca_res['r2_oos'], name="OOS"),)
    fig.add_trace(go.Scatter(x=pca_expl.index, y=pca_expl, name="lambda", mode="lines", yaxis='y2', line=dict(dash='dash')))
    fig.update_layout(
        template='plotly_white',
        title='In-sample and out-of-sample R-squared of predictive regressions',
        width=900, height=500,
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=0.6),
        yaxis=dict(tickvals=[round(-0.01+i/100,2) for i in range(7)], ticktext=[str(round(-0.01+i/100,2)) for i in range(7)], range=[-0.01,0.05], title='Reg. R-squared'),
        yaxis2=dict(tickvals=[round(-0.05+i/20,2) for i in range(7)], ticktext=[str(round(-0.05+i/20,2)) for i in range(7)], range=[-0.05,0.25] ,overlaying='y', side='right', title='PC Explained Variance'),
    )
    fig.show()
    pio.full_figure_for_development(fig, warn=False)
    fig.write_image("./result_export/r2_pca_in_oos.pdf",engine="kaleido")
    fig.write_image("./result_export/r2_pca_in_oos.png",engine="kaleido")

def create_latex_table_from_reg(res_list:list, title:str):
    """
    create a latex table from a list of regression results for statsmodels regressions
    """
    stargazer = Stargazer(res_list)
    stargazer.title(title)
    file_name = title + ".tex" #Include directory path if needed
    tex_file = open('./result_export/' + file_name, "w" ) #This will overwrite an existing file
    tex_file.write(stargazer.render_latex())
    tex_file.close()

def create_latex_table_from_df(df:pd.DataFrame, filename:str, title:str, description:str, precision:int=2, centering:bool=False, index_bool=True):
    """
    create a latex table from a dataframe
    """
    print('Create latex table for: ')
    display(HTML(df.to_html()))

    if index_bool:
        s = df.style.format(na_rep='', precision=precision)
    else:
        s = df.style.format(na_rep='', precision=precision).hide()
    #s = s.format_index(escape="latex")
    s.caption = title
    tex_string = s.to_latex(hrules=True)
    if centering:
        tex_string = tex_string.replace('caption{'+title+'}', 'caption{'+title+'}\n \caption*{'+description+'}')
        tex_string = tex_string.replace('end{tabular}', 'end{tabularx}')
        tex_string = re.sub(r'begin{tabular}{lr*}', r'begin{tabularx}{\\linewidth}{l *'+str(df.shape[1])+r'{>{\\centering\\arraybackslash}X}}', tex_string)
    file_name = filename + ".tex" #Include directory path if needed
    tex_file = open('./result_export/' + file_name, "w" ) #This will overwrite an existing file
    tex_file.write(tex_string)
    tex_file.close()

def plot_real_and_predicted_returns(df:pd.DataFrame, columns:list, title:str, filename:str):
    """
    plotting real and predicted return trajectories for the different predictive regressions
    """
    df.columns = columns
    df = df.rolling(12).mean() * 100
    df_min = df.min().min() - 1
    df_max = df.max().max() + 1
    fig = px.line(df, title=title, labels={'value':'Returns (%)', 'variable':'', 'date':'Time'})
    fig.update_layout(template='plotly_white',
                      legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=0.7),
                      showlegend=True,
                      width=900, height=500,
                      margin=dict(l=10, r=10))
    fig.update_traces(line=dict(color=plotly_colors[0], dash='dot'), selector=dict(name=columns[0]))
    fig.update_traces(line=dict(color=plotly_colors[1], dash='solid'), selector=dict(name=columns[1]))
    fig.update_traces(line=dict(color=plotly_colors[9], dash='solid'), selector=dict(name=columns[2]))
    fig.update_yaxes(range=[df_min, df_max])
    fig.show()
    pio.full_figure_for_development(fig, warn=False)
    fig.write_image("./result_export/"+filename+".pdf",engine="kaleido")
    fig.write_image("./result_export/"+filename+".png",engine="kaleido")

def load_data(parameters):

    all_ret = []
    all_bm = []
    all_n = []
    all_me = []
    all_ret_daily = []

    for f in parameters['factors']:
        all_ret.append(pd.read_csv(
            parameters['data_path']+f'ret10_{f}.csv', 
            parse_dates=['date'], index_col='date', 
            date_format=parameters['date_format']).add_prefix(f'{f}_')) # load returns
        all_bm.append(pd.read_csv(
            parameters['data_path']+f'bmc10_{f}.csv', 
            parse_dates=['date'], 
            index_col='date', 
            date_format=parameters['date_format']).add_prefix(f'{f}_')) # load book/market
        all_n.append(pd.read_csv(
            parameters['data_path']+f'n10_{f}.csv', 
            parse_dates=['date'], 
            index_col='date', 
            date_format=parameters['date_format_day']).add_prefix(f'{f}_')) # load number of firms
        all_me.append(pd.read_csv(
            parameters['data_path']+f'totalme10_{f}.csv', 
            parse_dates=['date'], 
            index_col='date', 
            date_format=parameters['date_format']).add_prefix(f'{f}_')) # load market cap
        all_ret_daily.append(pd.read_csv(
            parameters['data_path_daily']+f'ret10_{f}.csv', 
            parse_dates=['date'], 
            index_col='date', 
            date_format=parameters['date_format_day']).add_prefix(f'{f}_')) # load daily returns
        
    # concat data to one dataframe per datatype
    all_ret = pd.concat(all_ret, axis=1)
    all_n = pd.concat(all_n, axis=1)
    all_me = pd.concat(all_me, axis=1)
    all_bm = pd.concat(all_bm, axis=1)
    all_ret_daily = pd.concat(all_ret_daily, axis=1)
    all_n = all_n.groupby(pd.Grouper(freq='M')).median()
    all_n.index = all_ret.index
    all_book = all_bm.multiply(all_me)

    # cut data to the desired time period
    all_ret = all_ret.loc[parameters['start_date']:parameters['end_date']]
    all_ret_daily = all_ret_daily.loc[parameters['start_date']:parameters['end_date']]
    all_n = all_n.loc[parameters['start_date']:parameters['end_date']]
    all_me = all_me.loc[parameters['start_date']:parameters['end_date']]
    all_bm = all_bm.loc[parameters['start_date']:parameters['end_date']]
    all_book = all_book.loc[parameters['start_date']:parameters['end_date']]

    # sum the number of firms per anomaly to check for variations in basket construction
    all_n_sum = []
    for f in parameters['factors']:
        n_sum = all_n.loc[:,[f+'_p'+str(i) for i in range(1,11)]].sum(axis=1)
        n_sum.name = f
        all_n_sum.append(n_sum)
    all_n_sum = pd.concat(all_n_sum, axis=1)
    all_n_sum_m = all_n_sum.mean()

    # load one month risk free rate and set to monthly
    rf_rate = pd.read_csv('data/FED-SVENY.csv', index_col='date', parse_dates=True)['SVENY01']
    rf_rate_daily = rf_rate / 100
    rf_rate_daily = (rf_rate_daily+1)**(1/252)-1
    rf_rate_daily = rf_rate_daily.reindex(all_ret_daily.index).fillna(method='ffill')
    rf_rate = rf_rate.resample('M').mean() / 100
    rf_rate.index = rf_rate.index + pd.DateOffset(days=1)
    rf_rate = (rf_rate+1)**(1/12)-1

    # load inflation rate
    inf_rate = pd.read_csv('data/inflation.csv', index_col='date', parse_dates=True)
    inf_rate.loc[inf_rate['inflation']=='.',:] = np.nan
    inf_rate = inf_rate.astype(float)
    inf_rate = inf_rate.loc[parameters['start_date']:parameters['end_date']]

    return all_ret, all_ret_daily, all_n, all_me, all_bm, all_book, all_n_sum, all_n_sum_m, rf_rate, rf_rate_daily, inf_rate

def create_value_weights(all_me, factors):
    # create value weights per anomaly
    all_me_sum = []
    for f in factors:
        me_sum = all_me.loc[:,[f+'_p'+str(i) for i in range(1,11)]].sum(axis=1)
        me_sum.name = f+'_sum'
        all_me_sum.append(me_sum)
    all_me_sum = pd.concat(all_me_sum, axis=1)
    return all_me_sum

def build_anomaly_returns(all_ret, all_ret_daily, all_me, all_bm, parameters):
    """
    This function builds anomaly returns and book/market log differences based on the provided parameters.
    It operates differently depending on whether 'quintiles' is set to True or False in the parameters.

    Parameters:
    all_ret (DataFrame): DataFrame containing return data
    all_me (DataFrame): DataFrame containing market equity data
    all_bm (DataFrame): DataFrame containing book/market data
    parameters (dict): Dictionary containing various parameters for the function

    Returns:
    abnormal_ret (DataFrame): DataFrame containing anomaly returns
    abnormal_bm (DataFrame): DataFrame containing anomaly book/market log differences
    """
    # build anomaly returns: subtract returns p10 - p1
    if parameters['quintiles']:
        all_ret = all_ret.fillna(0)
        # weighted sum of p1 + p2 and p9 + p10
        all_me_weight = all_me.copy().fillna(0)
        all_me_12 = all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p1')].add(all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p2')].to_numpy())
        all_me_910 = all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p9')].add(all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p10')].to_numpy())
        all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p1')] = all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p1')].divide(all_me_12.to_numpy())
        all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p2')] = all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p2')].divide(all_me_12.to_numpy())
        all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p9')] = all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p9')].divide(all_me_910.to_numpy())
        all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p10')] = all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p10')].divide(all_me_910.to_numpy())

        # build anomaly returns
        abnormal_ret = all_ret.loc[:, all_ret.columns.str.endswith('_p10')].multiply(all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p10')].to_numpy()).add(
                        all_ret.loc[:, all_ret.columns.str.endswith('_p9')].multiply(all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p9')].to_numpy()).to_numpy()).subtract(
                        all_ret.loc[:, all_ret.columns.str.endswith('_p1')].multiply(all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p1')].to_numpy()).to_numpy()).subtract(
                        all_ret.loc[:, all_ret.columns.str.endswith('_p2')].multiply(all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p2')].to_numpy()).to_numpy())
        abnormal_ret.columns = abnormal_ret.columns.str.replace('_p10','')

        # build anomaly book/market log difference log(p10) - log(p1)
        abnormal_bm = np.log(all_bm.copy()).fillna(0)
        abnormal_bm = abnormal_bm.loc[:, abnormal_bm.columns.str.endswith('_p10')].multiply(all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p10')].to_numpy()).add(
                        abnormal_bm.loc[:, abnormal_bm.columns.str.endswith('_p9')].multiply(all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p9')].to_numpy()).to_numpy()).subtract(
                        abnormal_bm.loc[:, abnormal_bm.columns.str.endswith('_p1')].multiply(all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p1')].to_numpy()).to_numpy()).subtract(
                        abnormal_bm.loc[:, abnormal_bm.columns.str.endswith('_p2')].multiply(all_me_weight.loc[:, all_me_weight.columns.str.endswith('_p2')].to_numpy()).to_numpy())
        abnormal_bm.columns = abnormal_bm.columns.str.replace('_p10','')
    else:
        # build anomaly returns
        abnormal_ret = all_ret.loc[:, all_ret.columns.str.endswith('_p10')].subtract(all_ret.loc[:, all_ret.columns.str.endswith('_p1')].to_numpy())
        abnormal_ret.columns = abnormal_ret.columns.str.replace('_p10','')
        abnormal_ret_daily = all_ret_daily.loc[:, all_ret_daily.columns.str.endswith('_p10')].subtract(all_ret_daily.loc[:, all_ret_daily.columns.str.endswith('_p1')].to_numpy())
        abnormal_ret_daily.columns = abnormal_ret_daily.columns.str.replace('_p10','')

        # build anomaly book/market log difference log(p10) - log(p1)
        abnormal_bm = np.log(all_bm.copy())
        abnormal_bm = abnormal_bm.loc[:, abnormal_bm.columns.str.endswith('_p10')].subtract(abnormal_bm.loc[:, abnormal_bm.columns.str.endswith('_p1')].to_numpy())
        abnormal_bm.columns = abnormal_bm.columns.str.replace('_p10','')

    # change holding period from 1 to 6 months
    if parameters['holding_period'] == 6:
        res_ret = {}
        res_bm = {}
        # calculate 6 month returns
        for year in range(1974,2017):
            res_ret[str(year)+'-01-01'] = (abnormal_ret.loc[str(year)+'-01-01':str(year)+'-06-30',:]+1).prod()-1
            res_ret[str(year)+'-07-01'] = (abnormal_ret.loc[str(year)+'-07-01':str(year)+'-12-31',:]+1).prod()-1
            res_bm[str(year)+'-01-01'] = abnormal_bm.loc[str(year)+'-01-01':str(year)+'-06-30',:].mean()
            res_bm[str(year)+'-07-01'] = abnormal_bm.loc[str(year)+'-07-01':str(year)+'-12-31',:].mean()
        
        # convert to dataframe
        abnormal_ret = pd.DataFrame(res_ret).T
        abnormal_ret.index = pd.to_datetime(abnormal_ret.index)
        abnormal_bm = pd.DataFrame(res_bm).T
        abnormal_bm.index = pd.to_datetime(abnormal_bm.index)
    
    return abnormal_ret, abnormal_bm, abnormal_ret_daily


def build_market(all_ret, all_ret_daily, me_weights, rf_rate, rf_rate_daily, all_book, all_me_sum, parameters):
    """
    """

    # create market portfolio returns
    weighted_ret = all_ret.multiply(me_weights)
    market_ret = weighted_ret.loc[:,[parameters['market_var']+'_p'+str(i) for i in range(1,11)]].sum(axis=1)
    market_ret = market_ret - rf_rate.loc[market_ret.index] * int(parameters['market_excess_returns'])
    market_ret.name = 'market_ret'

    # create daily market portfolio returns
    me_weights_daily = me_weights.reindex(all_ret_daily.index).fillna(method='ffill').fillna(method='bfill')
    weighted_ret_daily = all_ret_daily.multiply(me_weights_daily)
    market_ret_daily = weighted_ret_daily.loc[:,[parameters['market_var']+'_p'+str(i) for i in range(1,11)]].sum(axis=1)
    market_ret_daily = market_ret_daily - rf_rate_daily.loc[market_ret_daily.index]
    market_ret_daily.name = 'market_ret'

    # create market portfolio book-to-market
    market_book = all_book.loc[:,[parameters['market_var']+'_p'+str(i) for i in range(1,11)]].sum(axis=1)
    market_bm = np.log(market_book.divide(all_me_sum[parameters['market_var']+'_sum'].to_numpy()))
    market_bm.name = 'market_bm'

    if parameters['holding_period'] == 6:
        # calculate 6 month returns for the market variables
        res_market_ret = {}
        res_market_bm = {}
        for year in range(1974,2017):
            res_market_ret[str(year)+'-01-01'] = (market_ret.loc[str(year)+'-01-01':str(year)+'-06-30']+1).prod()-1
            res_market_ret[str(year)+'-07-01'] = (market_ret.loc[str(year)+'-07-01':str(year)+'-12-31']+1).prod()-1
            res_market_bm[str(year)+'-01-01'] = market_bm.loc[str(year)+'-01-01':str(year)+'-06-30'].mean()
            res_market_bm[str(year)+'-07-01'] = market_bm.loc[str(year)+'-07-01':str(year)+'-12-31'].mean()
        
        # convert to series
        market_ret = pd.Series(res_market_ret)
        market_ret.index = pd.to_datetime(market_ret.index)
        market_ret.name = 'market_ret'
        market_bm = pd.Series(res_market_bm).T
        market_bm.name = 'market_bm'
        market_bm.index = pd.to_datetime(market_bm.index)

    return market_ret, market_bm, market_ret_daily

def total_r2(returns_pred:pd.DataFrame, returns_true:pd.DataFrame):
    """
    calculate the total r2 of the predicted returns
    """
    prediction_errors = returns_pred - returns_true.loc[returns_pred.index,:]
    cov_pred = prediction_errors.cov()
    cov_true = returns_true.loc[returns_pred.index,:].cov()
    trace_cov_pred = np.trace(cov_pred)
    trace_cov_true = np.trace(cov_true)
    return 1-trace_cov_pred/trace_cov_true, np.sum(np.diag(cov_pred)/np.diag(cov_true)<1)


def expected_utility(expected_ret, expected_risk, name, risk_aversion=0.5, dynamic=False):
    """
    calculates expected utility for a given set of expected returns and expected risks
    """
    if dynamic:
        exp_ut = 1/(2*risk_aversion) * np.diag(expected_ret @ pd.DataFrame(np.linalg.inv(expected_risk), index=expected_risk.columns, columns=expected_risk.columns) @ expected_ret.T)
        exp_ut = exp_ut.mean()
        exp_ut = pd.DataFrame(exp_ut, index=['Expected utility'], columns=[name])
    else:
        exp_ut = 1/(2*risk_aversion) * expected_ret.T @ pd.DataFrame(np.linalg.inv(expected_risk), index=expected_risk.columns, columns=expected_risk.columns) @ expected_ret
        exp_ut = exp_ut.mean()
        exp_ut.name = name
        exp_ut.index = ['Expected utility']
        exp_ut = pd.DataFrame(exp_ut)
    return exp_ut

def eval_strategy(weights_in_, weights_oos_, raw_z_in, raw_z_out, name, dynamic=False, weights_compare_in=None, weights_compare_oos=None):
    """
    evaluates a given strategy based on the weights and the raw returns
    it calculates the expected return, sharpe ratio and information ratio
    for the information ratio it uses the weights of weights_compare_in and weights_compare_oos
    """
    if dynamic:
        weights_in_ = weights_in_.T.reindex(raw_z_in.index).T
        strategy_ret_in = pd.DataFrame(np.diag(raw_z_in @ weights_in_), index=raw_z_in.index, columns=['return'])
        strategy_ret_in_m = strategy_ret_in.mean() * 12
        strategy_sharpe_in_m = strategy_ret_in.mean()/strategy_ret_in.std() * np.sqrt(12)

        weights_oos_ = weights_oos_.T.reindex(raw_z_out.index).T
        strategy_ret_oos = pd.DataFrame(np.diag(raw_z_out @ weights_oos_), index=raw_z_out.index, columns=['return'])
        strategy_ret_oos_m = strategy_ret_oos.mean() * 12
        strategy_sharpe_oos_m = strategy_ret_oos.mean()/strategy_ret_oos.std() * np.sqrt(12)

        strategy_compare_ret_in = raw_z_in @ weights_compare_in
        strategy_compare_ret_oos = raw_z_out @ weights_compare_oos
        strategy_compare_ret_in.columns = ['return']
        strategy_compare_ret_oos.columns = ['return']
        strategy_compare_in = strategy_ret_in - strategy_compare_ret_in
        strategy_compare_oos = strategy_ret_oos - strategy_compare_ret_oos
        strategy_info_in = strategy_compare_in.mean()/ strategy_compare_in.std() * np.sqrt(12)
        strategy_info_oos = strategy_compare_oos.mean() / strategy_compare_oos.std() * np.sqrt(12)
    else:
        strategy_ret_in = raw_z_in @ weights_in_
        strategy_ret_in_m = strategy_ret_in.mean() * 12
        strategy_sharpe_in_m = strategy_ret_in.mean()/strategy_ret_in.std() * np.sqrt(12)

        strategy_ret_oos = raw_z_out @ weights_oos_
        strategy_ret_oos_m = strategy_ret_oos.mean() * 12
        strategy_sharpe_oos_m = strategy_ret_oos.mean()/strategy_ret_oos.std() * np.sqrt(12)
    if weights_compare_in is not None:
        strategy_ret_in_m.name = 'IS Return'
        strategy_ret_oos_m.name = 'OOS Return'
        strategy_sharpe_in_m.name = 'IS Sharpe ratio'
        strategy_sharpe_oos_m.name = 'OOS Sharpe ratio'
        strategy_info_in.name = 'IS Inf. ratio'
        strategy_info_oos.name = 'OOS Inf. ratio'
        out = pd.concat([strategy_ret_in_m, strategy_ret_oos_m, strategy_sharpe_in_m, strategy_sharpe_oos_m, strategy_info_in, strategy_info_oos],axis=1)
        out.index = [name]
        return out.T
    else:
        strategy_ret_in_m.name = 'IS Return'
        strategy_ret_oos_m.name = 'OOS Return'
        strategy_sharpe_in_m.name = 'IS Sharpe ratio'
        strategy_sharpe_oos_m.name = 'OOS Sharpe ratio'
        out = pd.concat([strategy_ret_in_m, strategy_ret_oos_m, strategy_sharpe_in_m, strategy_sharpe_oos_m], axis=1)
        out.index = [name]
        return out.T

def create_weights(sigma, pred_returns, normalize=2):
        """
        calculate weights for a maximal sharpe ratio portfolio with possibly normalized weights
        """
        weights_strat = np.linalg.inv(sigma) @ pred_returns.T
        if normalize !=0 :
            weights_strat = weights_strat / np.linalg.norm(weights_strat, ord=normalize ,axis=0)
        if len(weights_strat.shape) == 1:
            weights_strat = weights_strat.rename(columns={'return':'weights'})
        weights_strat.index = sigma.index
        return weights_strat

def create_sigma(df_z_variance, errors_z, parameters):
    """
    create sigma matrix an basis of monthly variance and errors 
    -> this is an appendix approach thats not used in the paper
    """
    predictions = []
    error_z_sq = errors_z.loc[df_z_variance.index]**2
    for col in df_z_variance.columns:
        X = sm.add_constant(df_z_variance[col])
        model = sm.OLS(error_z_sq.loc[:parameters['oos_split_date'], col], X.loc[:parameters['oos_split_date']])
        results = model.fit()
        predictions.append(results.predict(X))
    predictions = pd.concat(predictions, axis=1)
    sigma = pd.DataFrame(np.diag(predictions.mean()), index=df_z_variance.columns, columns=df_z_variance.columns)
    return sigma


def plot_weights(weights, weights_in_ft, title):
    """
    plot changing weights over time for a given set of weights
    """
    if len(weights.columns) == 1:
        weights_tmp = pd.DataFrame(columns=weights_in_ft.columns, index=weights_in_ft.index)
        weights_tmp = weights_tmp.mul(weights['return'].to_numpy(), axis=0)
        weights = weights_tmp
    idx = weights.mean(axis=1).sort_values().index
    weights = weights.loc[idx,:]
    fig4 = go.Figure()
    for pc in idx:
        fig4.add_trace(go.Scatter(
                x=weights.columns, y=weights.loc[pc,:],
                name = pc if pc != 'market_ret' else 'MKT',
                mode = 'lines',
                stackgroup = 'one'))
    fig4.update_xaxes(
            title_text = 'Year')
    fig4.update_yaxes(
            title_text = "Weight", range = (-5, 22))
    fig4.update_layout(
            title = title,
            template = 'plotly_white',
            height=600, width=900,
            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),
            )
    fig4.show()


def calc_stambaugh_bias(market_bm, pc_bm, reg_market, pc_ret):
    """
    calculate the stambaugh bias for the given data
    """
    # AR(1) model to predict parameter bias in a parametric bootrsrep
    parameter_bias = {}
    persistance = {}
    # market bias
    X = sm.add_constant(market_bm.shift(1).dropna())
    Y = market_bm.loc[market_bm.index[1]:]
    res_beta_dev = []
    for i in range(1000):
        rand_idx = np.random.randint(0, len(market_bm)-1, len(market_bm))
        mod = sm.OLS(Y.iloc[rand_idx], X.iloc[rand_idx])
        res = mod.fit()
        res_beta_dev.append(res.params['market_bm'])
    parameter_bias['market'] = np.std(res_beta_dev)
    persistance['market'] = np.mean(res_beta_dev)

    for pc in ['PC1','PC2','PC3','PC4','PC5']:
        X = sm.add_constant(pc_bm.loc[:, pc].shift(1).dropna())
        Y = pc_bm.loc[pc_bm.index[1]:, pc]
        res_beta_dev = []
        for i in range(1000):
            rand_idx = np.random.randint(0, len(pc_bm)-1, len(pc_bm))
            mod = sm.OLS(Y.iloc[rand_idx], X.iloc[rand_idx])
            res = mod.fit()
            res_beta_dev.append(res.params[pc])
        parameter_bias[pc] = np.std(res_beta_dev)
        persistance[pc] = np.mean(res_beta_dev)

    persistance = pd.Series(persistance)**12
    parameter_std = pd.Series(parameter_bias)
    parameter_bias_estimate = -(1+3*persistance)/len(pc_bm/12)
    
    # build VAR(1) model for returns and bm to predict error correlation
    error_cov = {}
    error_bm_var = {}
    # market error correlation
    var_data = reg_market.copy()
    model_bm = sm.OLS(var_data.iloc[1:,1], sm.add_constant(var_data.iloc[:,1].shift(1).dropna())).fit()
    model_ret = sm.OLS(var_data.iloc[1:,0], sm.add_constant(var_data.iloc[:,1].shift(1).dropna())).fit()
    predictors = var_data.shift(1).dropna()
    solutions = var_data.loc[predictors.index,:]
    errors_bm = model_bm.predict(sm.add_constant(predictors['market_bm']))-solutions['market_bm']
    errors_ret = model_ret.predict(sm.add_constant(predictors['market_bm']))-solutions['market_ret']
    error_cov['market'] = pd.concat([errors_ret,errors_bm],axis=1).cov().iloc[0,1]
    error_bm_var['market'] = errors_bm.var()

    for pc in ['PC1','PC2','PC3','PC4','PC5']:
        var_data = pd.concat([pc_ret.add_suffix('_ret')[pc+'_ret'],pc_bm.add_suffix('_bm')[pc+'_bm']],axis=1)
        model_bm = sm.OLS(var_data.iloc[1:,1], sm.add_constant(var_data.iloc[:,1].shift(1).dropna())).fit()
        model_ret = sm.OLS(var_data.iloc[1:,0], sm.add_constant(var_data.iloc[:,1].shift(1).dropna())).fit()
        predictors = var_data.shift(1).iloc[rand_idx+1,:]
        solutions = var_data.iloc[rand_idx+1,:]
        errors_bm = model_bm.predict(sm.add_constant(predictors[pc+'_bm']))-solutions[pc+'_bm']
        errors_ret = model_ret.predict(sm.add_constant(predictors[pc+'_bm']))-solutions[pc+'_ret']
        error_cov[pc] = pd.concat([errors_ret,errors_bm],axis=1).cov().iloc[0,1]
        error_bm_var[pc] = errors_bm.var()

    error_factor = pd.Series(error_cov) / pd.Series(error_bm_var)
    stambaugh_bias = error_factor * parameter_bias_estimate
    return stambaugh_bias
