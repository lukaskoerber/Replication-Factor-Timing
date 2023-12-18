#%%
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

plotly_colors = px.colors.qualitative.Plotly

# data variables
data_path = './FT-17APR19-monthly/FT-17APR19-monthly/'
data_path_daily = './FT-17APR19-daily/FT-17APR19-daily/'
date_format = '%m/%Y'
date_format_day = '%m/%d/%Y'
start_date = '1974-01-01'
end_date = '2017-12-31'

# hyperparameters for processing and analysis
market_var = 'size' # which variable to use as market portfolio
market_bm_logged = True # whether to log the market b/m or not
market_excess_returns = True # whether to use excess returns or not
num_pca = 5 # number of pca components to use for the regression
quintiles = False # whether to use quintiles or deciles
subtract_market = True # whether to subtract the market from the returns
scale_for_market_var = True # whether to scale returns variance to market variance
holding_period = 1 # holding period in months
export_results = True # whether to export results to latex and pdf or not
log_entry = False # whether to log the entry of the program or not

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
    pio.full_figure_for_development(fig, warn=False)
    fig.write_image("./result_export/"+filename+".pdf",engine="kaleido")
    fig.write_image("./result_export/"+filename+".png",engine="kaleido")

# load data from the directory and exclude some factors
list_of_files = os.listdir(data_path)
name_map = pd.read_csv('./data/name_map.csv')
name_map = name_map.set_index('short_name')
# bmc10 = book/market, n10 = number of firms, ret10 = returns, totalme10 = market cap
datatypes = ['bmc10', 'n10', 'ret10', 'totalme10'] 
# extract the factor names from the file names in the data directory
factors = []
import re
for t in list_of_files:
    txt = re.split('_|\\.',t)[1]
    if txt not in factors and txt in name_map.index:
        factors.append(txt)

# %% load data
all_ret = []
all_ret_daily = []
all_bm = []
all_n = []
all_me = []
for f in factors:
    all_ret.append(pd.read_csv(
        data_path+f'ret10_{f}.csv', 
        parse_dates=['date'], index_col='date', 
        date_format=date_format).add_prefix(f'{f}_')) # laod returns
    all_bm.append(pd.read_csv(
        data_path+f'bmc10_{f}.csv', 
        parse_dates=['date'], 
        index_col='date', 
        date_format=date_format).add_prefix(f'{f}_')) # load book/market
    all_n.append(pd.read_csv(
        data_path+f'n10_{f}.csv', 
        parse_dates=['date'], 
        index_col='date', 
        date_format=date_format_day).add_prefix(f'{f}_')) # load number of firms
    all_me.append(pd.read_csv(
        data_path+f'totalme10_{f}.csv', 
        parse_dates=['date'], 
        index_col='date', 
        date_format=date_format).add_prefix(f'{f}_')) # load market cap
    all_ret_daily.append(pd.read_csv(
        data_path_daily+f'ret10_{f}.csv', 
        parse_dates=['date'], 
        index_col='date', 
        date_format=date_format_day).add_prefix(f'{f}_')) # load daily returns

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
all_ret = all_ret.loc[start_date:end_date]
all_ret_daily = all_ret_daily.loc[start_date:end_date]
all_n = all_n.loc[start_date:end_date]
all_me = all_me.loc[start_date:end_date]
all_bm = all_bm.loc[start_date:end_date]
all_book = all_book.loc[start_date:end_date]

# sum the number of firms per anomaly to check for variations in basket construction
all_n_sum = []
for f in factors:
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
inf_rate = inf_rate.loc[start_date:end_date]

# %%

# create value weights per anomaly
all_me_sum = []
for f in factors:
    me_sum = all_me.loc[:,[f+'_p'+str(i) for i in range(1,11)]].sum(axis=1)
    me_sum.name = f+'_sum'
    all_me_sum.append(me_sum)
all_me_sum = pd.concat(all_me_sum, axis=1)

# create weightings for the market portfolio returns
me_weights = all_me.copy()
for f in factors:
    me_weights.loc[:,[f+'_p'+str(i) for i in range(1,11)]] = all_me.loc[:,[f+'_p'+str(i) for i in range(1,11)]].divide(all_me_sum.loc[:,f+'_sum'], axis=0)

# build anomaly returns: subtract returns p10 - p1
if quintiles:
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
if holding_period == 6:
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

# create market portfolio returns
weighted_ret = all_ret.multiply(me_weights)
market_ret = weighted_ret.loc[:,[market_var+'_p'+str(i) for i in range(1,11)]].sum(axis=1)
market_ret = market_ret - rf_rate.loc[market_ret.index] * int(market_excess_returns)
market_ret.name = 'market_ret'

# create daily market portfolio returns
me_weights_daily = me_weights.reindex(all_ret_daily.index).fillna(method='ffill').fillna(method='bfill')
weighted_ret_daily = all_ret_daily.multiply(me_weights_daily)
market_ret_daily = weighted_ret_daily.loc[:,[market_var+'_p'+str(i) for i in range(1,11)]].sum(axis=1)
market_ret_daily = market_ret_daily - rf_rate_daily.loc[market_ret_daily.index]
market_ret_daily.name = 'market_ret'

# create market portfolio book-to-market
market_book = all_book.loc[:,[market_var+'_p'+str(i) for i in range(1,11)]].sum(axis=1)
market_bm = np.log(market_book.divide(all_me_sum[market_var+'_sum'].to_numpy()))
market_bm.name = 'market_bm'

if holding_period == 6:
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

# %% demarket anomaly returns

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

# regress anomaly returns on market returns and multiply beta times market return
mid_date = market_ret.index[len(market_ret)//2]
betas = {}
fullsample_ret = []
for f in factors:
    X = sm.add_constant(market_ret.loc[:mid_date])
    model = sm.OLS(abnormal_ret.loc[:mid_date][f], X) # regression of anomaly returns on market returns
    results = model.fit() # fit the model
    betas[f] = results.params['market_ret'] # save the beta
    tmp_ret = abnormal_ret[f] - market_ret * betas[f] * int(subtract_market)
    tmp_ret.name = f
    fullsample_ret.append(tmp_ret)
ret_betas = pd.Series(betas)
fullsample_ret = pd.concat(fullsample_ret, axis=1)

# scale with standard deviation
insample_ret_std = fullsample_ret.loc[:mid_date].std().copy()
insample_market_ret_std = market_ret.loc[:mid_date].std()
if scale_for_market_var:
    fullsample_ret = fullsample_ret.divide(insample_ret_std) * insample_market_ret_std
else:
    fullsample_ret = fullsample_ret.divide(insample_ret_std)

# regress anomaly bm on market bm and multiply beta times market bm
betas = {}
fullsample_bm = []
if market_bm_logged:
    market_bm_reg = market_bm
else:
    market_bm_reg = market_bm.copy()
    market_bm_reg = np.exp(market_bm_reg)

# regress anomaly bm on market bm and multiply beta times market bm
for f in factors:
    X = sm.add_constant(market_bm_reg.loc[:mid_date])
    model = sm.OLS(abnormal_bm.loc[:mid_date][f], X) # regression of anomaly returns on market returns
    results = model.fit() # fit the model
    betas[f] = results.params['market_bm'] # save the beta
    # dont subtract market bm from anomaly bm as unclear in the paper
    tmp_bm = abnormal_bm[f] - market_bm_reg * betas[f] * int(subtract_market) * 0 
    tmp_bm.name = f
    fullsample_bm.append(tmp_bm)
bm_betas = pd.Series(betas)
fullsample_bm = pd.concat(fullsample_bm, axis=1)

# different bm regressions, scale with standard deviation
insample_bm_std = fullsample_bm.loc[:mid_date].std().copy()
insample_market_bm_std = market_bm_reg.loc[:mid_date].std()
if scale_for_market_var:
    fullsample_bm = fullsample_bm.divide(insample_bm_std) * insample_market_bm_std
else:
    fullsample_bm = fullsample_bm.divide(insample_bm_std)

#%% pca

# run pca on insample returns
pca = PCA()
pca.fit(fullsample_ret.loc[:mid_date])
pca_expl = pd.Series(pca.explained_variance_ratio_)
pca_expl.index = ['PC'+str(i+1) for i in range(len(pca_expl))]
pca_comp = pd.DataFrame(pca.components_.T, index=factors, columns=['PC'+str(i+1) for i in range(len(fullsample_ret.columns))])

# flip pca components to match the paper results -> flipping because of negativ mean returns
pca_comp['PC2'] = -pca_comp['PC2']
pca_comp['PC4'] = -pca_comp['PC4']

# for scale approximation PC1: 0.3, PC2: 0.3, PC3: 0.4, PC4: 0.5, PC5: 0.5
# investigate the properties of the pca components -> not used in the paper
pc_properties = pd.DataFrame(index=pca_comp.iloc[:,:].columns)
pc_properties['mean'] = pca_comp.iloc[:,:].mean()
pc_properties['expos'] = pca_comp.iloc[:,:].abs().sum()
pc_properties['s-leg'] = (pca_comp*(pca_comp<0).astype(int)).iloc[:,:].sum()
pc_properties['l-leg'] = (pca_comp*(pca_comp>0).astype(int)).iloc[:,:].sum()
pc_properties['s-leg%'] = pc_properties['s-leg'] / pc_properties['expos']
pc_properties['l-leg%'] = pc_properties['l-leg'] / pc_properties['expos']
pc_properties['sum'] = pc_properties['s-leg'] + pc_properties['l-leg']

if export_results:
    # export pca properties to latex
    pca_table = pd.concat([pca_expl.iloc[:10],pca_expl.iloc[:10].cumsum()],axis=1).rename(columns={0:'\% var. explained',1:'Cumulative'}).T * 100
    create_latex_table_from_df(pca_table,
                            'pca_table',
                            'Percentage of variance explained by anomaly PCs', 
                            'Percentage of variance explained by each PC of the 50 anomaly strategies.', 
                            1)

# %% run predictive regressions
    
# run predictive regressions
pc_ret = fullsample_ret.dot(pca_comp)
pc_bm = fullsample_bm.dot(pca_comp)
pc_ret_daily = abnormal_ret_daily.dot(pca_comp)

# run predictive regressions
# market regressions
if market_excess_returns:
    reg_market = pd.concat([market_ret,market_bm.shift(1)], axis=1)
else:
    tmp_market_ret = market_ret - rf_rate.loc[market_ret.index]
    tmp_market_ret.name = 'market_ret'
    reg_market = pd.concat([tmp_market_ret, market_bm.shift(1)], axis=1)
reg_market = reg_market.dropna()
reg_market_in = reg_market.loc[:mid_date]
reg_market_out = reg_market.loc[mid_date:]

# market regressions oos
model = smf.ols('market_ret ~ market_bm', data=reg_market_in)
results = model.fit()
t_test_oos_m = results.t_test('market_bm=0').summary_frame()['t'][0]

# market regressions is
model_full = smf.ols('market_ret ~ market_bm', data=reg_market)
results_full = model_full.fit()
t_test_in_m = results.t_test('market_bm=0').summary_frame()['t'][0]

# market regressions results and predictions
full_pred_market = results_full.predict(reg_market['market_bm'])
full_pred_market.name = 'market_ret'
oos_pred_market = results.predict(reg_market['market_bm'])
oos_pred_market.name = 'market_ret'
r2_oos_m = 1-((reg_market_out['market_ret']-oos_pred_market.loc[mid_date:]).var()/(reg_market_out['market_ret'].var()))
r2_in_m = 1-((reg_market['market_ret']-full_pred_market).var()/(reg_market['market_ret'].var()))

# pca regressions
tmp_data = pd.concat([pc_ret.add_suffix('_ret'),pc_bm.shift(1).add_suffix('_bm')], axis=1)
tmp_data = tmp_data.dropna()

# start with market results
pca_res = {'r2_in':{'market':r2_in_m},
           'r2_oos':{'market':r2_oos_m},
           'beta_in':{'market':results_full.params['market_bm']},
           'beta_oos':{'market':results.params['market_bm']},
           't_test_in':{'market':t_test_in_m},
           't_test_oos':{'market':t_test_oos_m}}

# run predictive regressions for the pca components
in_preds = []
oos_preds = []
for i in range(len(pca_expl)):
    # run predictive regressions for the pca components is
    model = smf.ols('PC'+str(i+1)+'_ret ~ '+'PC'+str(i+1)+'_bm', data=tmp_data)
    results = model.fit()
    in_pred = results.predict(tmp_data['PC'+str(i+1)+'_bm'])
    in_pred.name = 'PC'+str(i+1)  

    # run predictive regressions for the pca components oos
    model_in = smf.ols('PC'+str(i+1)+'_ret ~ '+'PC'+str(i+1)+'_bm', data=tmp_data.loc[:mid_date])
    results_in = model_in.fit()
    oos_pred = results_in.predict(tmp_data.loc[:,'PC'+str(i+1)+'_bm'])
    oos_pred.name = 'PC'+str(i+1)
    
    # save results
    in_preds.append(in_pred)
    oos_preds.append(oos_pred)
    r2_oos = 1-((tmp_data.loc[mid_date:,'PC'+str(i+1)+'_ret']-oos_pred.loc[mid_date:]).var()/tmp_data.loc[mid_date:,'PC'+str(i+1)+'_ret'].var())
    pca_res['r2_in']['PC'+str(i+1)] = results.rsquared
    pca_res['r2_oos']['PC'+str(i+1)] = r2_oos
    pca_res['beta_in']['PC'+str(i+1)] = results.params['PC'+str(i+1)+'_bm']
    pca_res['beta_oos']['PC'+str(i+1)] = results_in.params['PC'+str(i+1)+'_bm']
    pca_res['t_test_in']['PC'+str(i+1)] = results.t_test('PC'+str(i+1)+'_bm=0').summary_frame()['t'][0]
    pca_res['t_test_oos']['PC'+str(i+1)] = results_in.t_test('PC'+str(i+1)+'_bm=0').summary_frame()['t'][0]

# convert to dataframe
pca_res = pd.DataFrame(pca_res)
in_pred = pd.concat(in_preds, axis=1)
oos_pred = pd.concat(oos_preds, axis=1)
pca_res = pca_res.rename(index={'market':'MKT'})

if export_results:
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
    pio.full_figure_for_development(fig, warn=False)
    fig.write_image("./result_export/r2_pca_in_oos.pdf",engine="kaleido")
    fig.write_image("./result_export/r2_pca_in_oos.png",engine="kaleido")

    # collect regression results and save to tex file
    reg_table = pca_res.loc[:'PC5',['beta_in','t_test_in','beta_oos','t_test_oos','r2_in','r2_oos']]
    reg_table.loc[:,['beta_in','beta_oos','r2_in','r2_oos']] = reg_table.loc[:,['beta_in','beta_oos','r2_in','r2_oos']] *100
    reg_table = reg_table.rename(columns={'beta_in':'Own $bm$ (IS)','beta_oos':'Own $bm$ (OOS)', 't_test_in':'t-stat. (IS)','t_test_oos':'t-stat. (OOS)','r2_in':'$R^2$','r2_oos':'OOS $R^2$'}, index={'market':'MKT'})
    reg_table = reg_table.T
    create_latex_table_from_df(reg_table, 
                            'reg_table',
                            'Predicting dominant equity components with BE/ME ratios', 
                            'We report results from predictive regressions of excess market returns and five PCs of long-short anomaly returns. The market is forecasted using the log of the aggregate book-to-market ratio. The anomaly PCs are forecasted using a restricted linear combination of anomalies log book-to-market ratios with weights given by the corresponding eigenvector of pooled long-short strategy returns. The first row shows the coefficient estimate. The second row shows asymptotic t-statistics estimated using the method of Newey and West (1987). The third and fourth rows show the bias and p-value from a parametric bootstrap. The fifth and sixth rows shows the in-sample and out-of-sample monthly R2. The last three rows give critical values of the OOS R2 based on the placebo test in Kelly and Pruitt (2013).', 
                            precision=2, centering=True)

    # plot real and predicted returns for the market and the pca components
    market_plot = pd.concat([reg_market['market_ret'],full_pred_market,oos_pred_market.loc[mid_date:]],axis=1)
    plot_real_and_predicted_returns(market_plot, ['Realized', 'Predicted(IS)', 'Predicted(OOS)'], '(a) Market', 'market_returns')

    pc1_plot = pd.concat([pc_ret['PC1'],in_pred['PC1'],oos_pred.loc[mid_date:,'PC1']],axis=1)
    plot_real_and_predicted_returns(pc1_plot, ['Realized', 'Predicted(IS)', 'Predicted(OOS)'], '(b) PC1', 'pc1_returns')

    pc2_plot = pd.concat([pc_ret['PC2'],in_pred['PC2'],oos_pred.loc[mid_date:,'PC2']],axis=1)
    plot_real_and_predicted_returns(pc2_plot, ['Realized', 'Predicted(IS)', 'Predicted(OOS)'], '(c) PC2', 'pc2_returns')

    pc3_plot = pd.concat([pc_ret['PC3'],in_pred['PC3'],oos_pred.loc[mid_date:,'PC3']],axis=1)
    plot_real_and_predicted_returns(pc3_plot, ['Realized', 'Predicted(IS)', 'Predicted(OOS)'], '(d) PC3', 'pc3_returns')

    pc4_plot = pd.concat([pc_ret['PC4'],in_pred['PC4'],oos_pred.loc[mid_date:,'PC4']],axis=1)
    plot_real_and_predicted_returns(pc4_plot, ['Realized', 'Predicted(IS)', 'Predicted(OOS)'], '(e) PC4', 'pc4_returns')

    pc5_plot = pd.concat([pc_ret['PC5'],in_pred['PC5'],oos_pred.loc[mid_date:,'PC5']],axis=1)
    plot_real_and_predicted_returns(pc5_plot, ['Realized', 'Predicted(IS)', 'Predicted(OOS)'], '(f) PC5', 'pc5_returns')
else:
    px.bar(pca_res, x=pca_res.index, y=['r2_in','r2_oos'], barmode='group', title='In-sample and out-of-sample R-squared', template='plotly_white').show()

if log_entry:
    # create log entry, important for the different robustness tests
    total_r2_res, significant = total_r2(oos_pred.loc[mid_date:,'PC'+str(1):'PC'+str(num_pca)], pc_ret.loc[:,'PC'+str(1):'PC'+str(num_pca)])
    total_r2_log = pd.read_csv('./result_export/total_r2_log.csv', index_col=0)
    setting_idx = str(num_pca)+str(int(quintiles))+str(int(subtract_market))+str(int(scale_for_market_var))+str(holding_period)
    total_r2_log.loc[setting_idx,:] = pd.Series(
        [quintiles, holding_period, num_pca, holding_period==1, subtract_market, scale_for_market_var, round(total_r2_res * 100,2), significant],
        index=['anom_port_sort','holding_period','num_pc','monthly_pc','market_adj_ret','scaled_var','total_r2','num_sign_pc'])
    total_r2_log.to_csv('./result_export/total_r2_log.csv')

# back to factor projection
ch_pred_in = in_pred.loc[:,'PC'+str(1):'PC'+str(num_pca)] @ pca_comp.loc[:,'PC'+str(1):'PC'+str(num_pca)].T
ch_pred_oos = oos_pred.loc[mid_date:,'PC'+str(1):'PC'+str(num_pca)] @ pca_comp.loc[:,'PC'+str(1):'PC'+str(num_pca)].T

# calculate factor predictability after back to factor projection
r2 = {'in':{},'oos':{}}
for factor in ch_pred_in.columns:
    r2['in'][factor] = r2_score(y_true=fullsample_ret.loc[ch_pred_in.index,factor], y_pred=ch_pred_in[factor])
    r2['oos'][factor] = r2_score(y_true=fullsample_ret.loc[ch_pred_oos.index,factor], y_pred=ch_pred_oos[factor])
factor_predictability = pd.DataFrame(r2).sort_values('oos', ascending=False)

if export_results:
    # summarize factor predictability and save to tex file
    name_map = pd.read_csv('./data/name_map.csv')
    name_map = name_map.set_index('short_name')
    factor_predictability_table = pd.concat([name_map, factor_predictability * 100], axis=1)
    factor_predictability_table = factor_predictability_table.rename(columns={'index':'#','name':'Characteristic','in':'IS', 'oos':'OOS'})
    factor_predictability_table = factor_predictability_table.set_index('#')
    create_latex_table_from_df(factor_predictability_table,
                               'factor_predictability_table',
                               'Predicting individual anomaly returns: $R^2$ (%)',
                               'Monthly predictive $R^2$ of individual anomalies returns using implied fitted values based on PC forecasts. Column 1 (IS) provides estimates in full sample. Column 2 (OOS) shows out-of-sample $R^2$',
                               precision=1, centering=True)
    
    # plot factor predictability and save to pdf
    fig = px.bar(factor_predictability_table, x=factor_predictability_table['Characteristic'], y=['IS','OOS'], barmode='group', title='In-sample and out-of-sample R-squared', template='plotly_white')
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=0.6),
                    showlegend=True,
                    width=900, height=700,
                    margin=dict(l=10, r=10),
                    yaxis=dict(title='R-squared (%)', range=[-5,5]))
    pio.full_figure_for_development(fig, warn=False)
    fig.write_image("./result_export/factor_predictability.pdf",engine="kaleido")
    fig.write_image("./result_export/factor_predictability.png",engine="kaleido")

    # calculate summary statistics of factor predictability
    factor_pred_table_desc = factor_predictability_table.describe().loc[['mean','50%','std','min','max'],:]
    factor_pred_table_desc = factor_pred_table_desc.rename(index={'mean':'Mean','std':'Std. Dev.','min':'Min.','max':'Max.','50%':'Median'})

    # save robustness log file to tex table
    total_r2_log = pd.read_csv('./result_export/total_r2_log.csv', index_col=0)
    total_r2_log['Anomaly portfolio sort'] = 'Deciles'
    total_r2_log.loc[total_r2_log['anom_port_sort']==True,'Anomaly portfolio sort'] = 'Quintiles'
    total_r2_log['Holding period'] = total_r2_log['holding_period'].astype(int)
    total_r2_log['# of PCs'] = total_r2_log['num_pc'].astype(int)
    total_r2_log['Monthly PC'] = 'X'
    total_r2_log.loc[total_r2_log['monthly_pc']==False,'Monthly PC'] = ''
    total_r2_log['Market adjusted returns'] = 'X'
    total_r2_log.loc[total_r2_log['market_adj_ret']==False,'Market adjusted returns'] = ''
    total_r2_log['Scaled market variance'] = 'X'
    total_r2_log.loc[total_r2_log['scaled_var']==False,'Scaled market variance'] = ''
    total_r2_log['OOS Total $R^2$'] = total_r2_log['total_r2']
    total_r2_log['Significant PCs'] = total_r2_log['num_sign_pc'].astype(int)
    total_r2_log = total_r2_log.drop(['anom_port_sort','holding_period','num_pc','monthly_pc','market_adj_ret','scaled_var','total_r2','num_sign_pc'], axis=1)
    create_latex_table_from_df(total_r2_log,
                               'various_data_choices',
                               'Various data choices',
                               'The table reports summary statistics of predictive regressions in Table 2 for various data construction choices. Specifically, we report the OOS total R2 and the number of PC portfolios for which the OOS R2 is statistically significant using the placebo test of Kelly and Pruitt (2013). The first column reports the number of portfolios used for the underlying characteristic sorts. The second column reports the holding period in months. For holding periods longer than one month, the third column reports whether principal components are estimated using monthly or holding period returns. The fourth column reports whether the anomaly returns are orthogonalized relative to the aggregate market. The fifth column reports whether the anomaly returns and book-to-market values are normalized to have equal variance.',
                               precision=2, centering=True, index_bool=False)


# expected returns calculated from the pca components
if export_results:
    # combine the four factors to a plot using make subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=('(a) Size', '(b) Value', '(c) Momentum', '(d) Profitability'))
    fig.add_trace(go.Scatter(x=ch_pred_in.index, y=ch_pred_in['size'].rolling(6, center=True).mean(), name='(a) Size'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ch_pred_in.index, y=ch_pred_in['value'].rolling(6, center=True).mean(), name='(b) Value'), row=1, col=2)
    fig.add_trace(go.Scatter(x=ch_pred_in.index, y=ch_pred_in['mom'].rolling(6, center=True).mean(), name='(c) Momentum'), row=2, col=1)
    fig.add_trace(go.Scatter(x=ch_pred_in.index, y=ch_pred_in['roa'].rolling(6, center=True).mean(), name='(d) Profitability'), row=2, col=2)
    fig.update_layout(height=600, width=800, title_text="Anomaly expected returns", showlegend=False, template='plotly_white')
    fig.update_traces(line=dict(color=plotly_colors[0], width=2))
    pio.full_figure_for_development(fig, warn=False)
    fig.write_image("./result_export/anomaly_exp_rets.pdf",engine="kaleido")
    fig.write_image("./result_export/anomaly_exp_rets.png",engine="kaleido")

# %%

if not quintiles and holding_period==1:
    # build portfolios for static factor investing
    # crate evaluation data
    tmp_portfolio_ret = abnormal_ret.copy()
    portfolio_ret = tmp_portfolio_ret @ pca_comp.loc[:,'PC1':'PC'+str(num_pca)]
    portfolio_ret_in = portfolio_ret
    portfolio_ret_oos = portfolio_ret.loc[mid_date:,:]

    # calculate market excess returns, if market returns are not excess returns subtract risk free rate
    market_ret_eval = market_ret - rf_rate.loc[market_ret.index] * (1-int(market_excess_returns))
    market_ret_eval.name = 'market_ret'

    # combine market returns and portfolio returns
    raw_z_in = pd.concat([market_ret_eval, portfolio_ret_in], axis=1, join='inner')
    raw_z_out = pd.concat([market_ret_eval, portfolio_ret_oos], axis=1, join='inner')

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

    def eval_strategy(weights_in_, weights_oos_, name, dynamic=False, weights_compare_in=None, weights_compare_oos=None):
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

    if scale_for_market_var:
        market_ret_z = reg_market['market_ret'].copy()
    else:
        market_ret_z = reg_market['market_ret'].copy() / insample_market_ret_std
    norm_weights = 0

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

    def create_sigma(df_z_variance, errors_z):
        """
        create sigma matrix an basis of monthly variance and errors 
        -> this is an appendix approach thats not used in the paper
        """
        predictions = []
        error_z_sq = errors_z.loc[df_z_variance.index]**2
        for col in df_z_variance.columns:
            X = sm.add_constant(df_z_variance[col])
            model = sm.OLS(error_z_sq.loc[:mid_date, col], X.loc[:mid_date])
            results = model.fit()
            predictions.append(results.predict(X))
        predictions = pd.concat(predictions, axis=1)
        sigma = pd.DataFrame(np.diag(predictions.mean()), index=df_z_variance.columns, columns=df_z_variance.columns)
        return sigma

    # create the z vectors over time
    df_z_daily = pd.concat([market_ret_daily, pc_ret_daily.loc[:,'PC1':'PC'+str(num_pca)]], axis=1)
    df_z_monthly_var = df_z_daily.groupby(pd.Grouper(freq='M')).var().iloc[:-1]
    df_z_monthly_var.index = df_z_monthly_var.index + pd.DateOffset(days=1)
    df_z_real = pd.concat([market_ret_z, pc_ret.loc[:,'PC1':'PC'+str(num_pca)]], axis=1)
    df_z_pred =  pd.concat([oos_pred_market, oos_pred.loc[:, 'PC1':'PC'+str(num_pca)]], axis=1)
    errors_z = df_z_real.subtract(df_z_pred)
    # construct sigma matrix
    sigma = (df_z_real-df_z_pred).loc[:mid_date].cov()

    # create weights
    pred_z_in = pd.DataFrame(pd.concat([market_ret_z, pc_ret.loc[:,'PC1':'PC'+str(num_pca)]], axis=1).mean(),columns=['return']).T
    pred_z_oos = pd.DataFrame(pd.concat([market_ret_z, pc_ret.loc[:mid_date, 'PC1':'PC'+str(num_pca)]], axis=1, join='inner').mean(),columns=['return']).T
    # l2 norm weights
    weights_in_fi = create_weights(sigma,pred_z_in,normalize=norm_weights)
    weights_oos_fi = create_weights(sigma,pred_z_oos,normalize=norm_weights)
    # evaluate strategy
    res_strat_fi = eval_strategy(weights_in_fi, weights_oos_fi, 'Factor investing')
    res_strat_fi = pd.concat([res_strat_fi,expected_utility(pd.DataFrame(pred_z_in.mean(), columns=['weights']),sigma, 'Factor investing')])

    # build portfolios for market timing
    pred_z_in = pd.concat([full_pred_market,pc_ret.loc[:,'PC1':'PC'+str(num_pca)]], axis=1, join='inner')
    pred_z_in.loc[:,'PC1':'PC'+str(num_pca)] = np.ones((len(pred_z_in),num_pca)) * pc_ret.loc[:,'PC1':'PC'+str(num_pca)].mean().to_numpy()
    pred_z_oos = pd.concat([oos_pred_market, pc_ret.loc[mid_date:, 'PC1':'PC'+str(num_pca)]], axis=1, join='inner')
    pred_z_oos.loc[:,'PC1':'PC'+str(num_pca)] = np.ones((len(pred_z_oos),num_pca)) * pc_ret.loc[:mid_date,'PC1':'PC'+str(num_pca)].mean().to_numpy()
    weights_in_mt = create_weights(sigma,pred_z_in,normalize=norm_weights)
    weights_oos_mt = create_weights(sigma,pred_z_oos,normalize=norm_weights)
    # evaluate strategy
    res_strat_mt = eval_strategy(weights_in_mt, weights_oos_mt, 'Market timing', dynamic=True, weights_compare_in=weights_in_fi, weights_compare_oos=weights_in_fi)
    res_strat_mt = pd.concat([res_strat_mt,expected_utility(pred_z_in, sigma, 'Market timing', dynamic=True)])

    # build portfolios for factor timing
    pred_z_in = pd.concat([full_pred_market,in_pred.loc[:,'PC1':'PC'+str(num_pca)]], axis=1)
    pred_z_oos = pd.concat([oos_pred_market, oos_pred.loc[:, 'PC1':'PC'+str(num_pca)]], axis=1, join='inner')
    weights_in_ft = create_weights(sigma,pred_z_in,normalize=norm_weights)
    weights_oos_ft = create_weights(sigma,pred_z_oos,normalize=norm_weights)
    # evaluate strategy
    res_strat_ft = eval_strategy(weights_in_ft, weights_oos_ft, 'Factor timing' ,dynamic=True, weights_compare_in=weights_in_fi, weights_compare_oos=weights_in_fi)
    res_strat_ft = pd.concat([res_strat_ft,expected_utility(pred_z_in, sigma, 'Factor timing', dynamic=True)])

    # build portfolios for anomaly timing
    pred_z_in = pd.concat([market_ret_z,in_pred.loc[:,'PC1':'PC'+str(num_pca)]], axis=1)
    pred_z_in.loc[:,'market_ret'] = market_ret_z.mean()
    pred_z_oos = pd.concat([market_ret_z, oos_pred.loc[:, 'PC1':'PC'+str(num_pca)]], axis=1, join='inner')
    pred_z_oos.loc[:,'market_ret'] = market_ret_z.loc[:mid_date].mean()
    weights_in_at = create_weights(sigma,pred_z_in,normalize=norm_weights)
    weights_oos_at = create_weights(sigma,pred_z_oos,normalize=norm_weights)
    # evaluate strategy
    res_strat_at = eval_strategy(weights_in_at, weights_oos_at, 'Anomaly timing', dynamic=True, weights_compare_in=weights_in_fi, weights_compare_oos=weights_in_fi)
    res_strat_at = pd.concat([res_strat_at,expected_utility(pred_z_in, sigma, 'Anomaly timing', dynamic=True)])

    # build portfolios for pure anomaly timing
    pred_z_in = pd.concat([market_ret_z, in_pred.loc[:,'PC1':'PC'+str(num_pca)]-in_pred.loc[:,'PC1':'PC'+str(num_pca)].mean()], axis=1)
    pred_z_in.loc[:,'market_ret'] = 0
    pred_z_oos = pd.concat([market_ret_z, oos_pred.loc[:,'PC1':'PC'+str(num_pca)]-oos_pred.loc[:mid_date,'PC1':'PC'+str(num_pca)].mean()], axis=1, join='inner')
    pred_z_oos.loc[:,'market_ret'] = 0
    weights_in_pat = create_weights(sigma,pred_z_in,normalize=norm_weights)
    weights_oos_pat = create_weights(sigma,pred_z_oos,normalize=norm_weights)
    # evaluate strategy
    res_strat_pat = eval_strategy(weights_in_pat, weights_oos_pat, 'Pure anom. timing', dynamic=True, weights_compare_in=weights_in_fi, weights_compare_oos=weights_in_fi)
    res_strat_pat = pd.concat([res_strat_pat,expected_utility(pred_z_in, sigma, 'Pure anom. timing', dynamic=True)])

    if export_results:
        res_strat_table = pd.concat([res_strat_fi, res_strat_mt, res_strat_ft, res_strat_at, res_strat_pat], axis=1)
        res_strat_table = res_strat_table.loc[['IS Return', 'OOS Return', 'IS Sharpe ratio', 'OOS Sharpe ratio', 'IS Inf. ratio', 'OOS Inf. ratio', 'Expected utility'],:]
        create_latex_table_from_df(res_strat_table,
                                'strat_table',
                                'Performance of various portfolio strategies',
                                'The table reports the unconditional Sharpe ratio, information ratio, and average mean-variance utility of five strategies: (i) static factor investing strategy, based on unconditional estimates of E [Zt]; (ii) market timing strategy which uses forecasts of the market return based on Table 2 but sets expected returns on the PC equal to unconditional values; (iii) full factor timing strategy including predictability of the PCs and the market; (iv) anomaly timing strategy which uses forecasts of the PCs based on Table 2 but sets expected returns on the market to unconditional values; and (v) pure anomaly timing strategy sets the weight on the market to zero and invests in anomalies proportional to the deviation of their forecast to its unconditional average, Et[Zt+1] - E [Zt]. All strategies assume a homoskedastic conditional covariance matrix, estimated as the covariance of forecast residuals. Information ratios are calculated relative to the static strategy. Out-of-sample(OOS) values are based on split-sample analysis with all parameters estimated using the first half of the data', 
                                )

    def plot_weights(weights, title):
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

    # variation of market timing and factor timing
    var_sdf = pd.DataFrame(columns=['market_timing', 'factor_timing'], index=weights_in_mt.columns)
    var_sdf_fi = (weights_in_fi.T @ sigma @ weights_in_fi).values[0,0]*12
    var_sdf['market_timing'] = np.diag(weights_in_mt.T @ sigma @ weights_in_mt)
    var_sdf['factor_timing'] = np.diag(weights_in_ft.T @ sigma @ weights_in_ft)
    var_sdf = var_sdf * 12
    if export_results:
        # plot variation of market timing and factor timing
        fig = px.line(var_sdf.rolling(6).mean(), title='Conditional variance of SDFs', labels={'value':'SDF Variance', 'variable':'', 'date':'Time'})
        fig.update_layout(template='plotly_white',
                            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=0.6),
                            showlegend=True,
                            width=900, height=500,
                            margin=dict(l=10, r=10))
        pio.full_figure_for_development(fig, warn=False)
        fig.write_image("./result_export/sdf_variance.pdf",engine="kaleido")
        fig.write_image("./result_export/sdf_variance.png",engine="kaleido")

        # plot variation offactor timing with inflation
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=var_sdf.index, y=var_sdf['factor_timing'], name="SDF Variance", mode="lines"))
        fig.add_trace(go.Scatter(x=inf_rate.index, y=inf_rate['inflation'], name="Inflation", mode="lines", yaxis='y2', line=dict(dash='dash')))
        fig.update_layout(
            template='plotly_white',
            title='Conditional variance of SDFs and inflation',
            width=900, height=500,
            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=0.7),
            yaxis=dict(range=[0,14], title='SDF Variance'),
            yaxis2=dict(range=[0,14] , title='Inflation (%)', overlaying='y', side='right'))
        pio.full_figure_for_development(fig, warn=False)
        fig.write_image("./result_export/sdf_var_inflation.pdf",engine="kaleido")

        # create summary table for variance of sdf
        var_sdf_plot = pd.concat([pd.Series([var_sdf_fi,np.nan],index=['mean','std']),var_sdf.describe().loc[['mean','std']]],axis=1)
        var_sdf_plot.columns = ['Factor Investing','Market Timing','Factor Timing']
        var_sdf_plot.index = ['E[var(m)]','std[var(m)]']
        create_latex_table_from_df(var_sdf_plot,
                                'sdf_variance_table',
                                'Variance of the SDF',
                                'We report the average conditional variance of the SDF and its standard deviation constructed under various sets of assumptions. “Factor timing” is our full estimate, which takes into account variation in the means of the PCs and the market. “Factor investing” imposes the assumption of no factor timing: conditional means are replaced by their unconditional counterpart. “Market timing” only allows for variation in the mean of the market return', 
                                centering=True)

    # correlation of factor investing and factor timing
    var_sdf_fi = (weights_in_fi.T @ sigma @ weights_in_fi).values[0,0]
    var_sdf_ft = pd.DataFrame(columns=['factor_timing'], index=weights_in_ft.columns)
    var_sdf_ft['factor_timing'] = np.diag(weights_in_ft.T @ sigma @ weights_in_ft)
    cov_sdf = pd.DataFrame(columns=['covariance'], index=weights_in_ft.columns)
    cov_sdf['covariance'] = weights_in_ft.T @ sigma @ weights_in_fi
    corr_sdf = pd.DataFrame(columns=['correlation'], index=weights_in_ft.columns)
    corr_sdf['correlation'] = cov_sdf['covariance'] / np.sqrt(var_sdf_fi * var_sdf_ft['factor_timing'])
    if export_results:
        # plot correlation of factor investing and factor timing
        fig = px.line(corr_sdf.rolling(6).mean(), title='Correlation of factor investing and factor timing', labels={'value':'Conditional correlation', 'variable':'', 'date':'Time'})
        fig.update_layout(template='plotly_white',
                        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=0.6),
                        showlegend=False,
                        width=900, height=500,
                        margin=dict(l=10, r=10))
        fig.update_yaxes(range=[0.2, 1.0])
        pio.full_figure_for_development(fig, warn=False)
        fig.write_image("./result_export/strategie_correlation.pdf",engine="kaleido")
        fig.write_image("./result_export/strategie_correlation.png",engine="kaleido")

    # different methods for predicting anomaly returns
    # regression of all factor valuation ratios on all factor returns
    r2_methods = {'raw_factors':{}}
    predictions = {}
    for f in factors:
        X = sm.add_constant(fullsample_bm.loc[:mid_date].shift(1).dropna())
        model = sm.OLS(fullsample_ret.loc[fullsample_ret.index[1]:mid_date,f], X)
        results = model.fit()
        X2 = sm.add_constant(fullsample_bm.shift(1).loc[mid_date:])
        predictions[f] = results.predict(X2)
        r2_methods['raw_factors'][f] = r2_score(y_true=fullsample_ret.loc[mid_date:,f], y_pred=results.predict(X2))
    predictions = pd.DataFrame(predictions)
    total_r2_raw_factors = total_r2(predictions, fullsample_ret.loc[mid_date:])[0]

    # regression of own pc valuation ratios on own pc returns
    r2_methods['pc_and_bm'] = {}
    predictions = {}
    pcs = ['PC1','PC2','PC3','PC4','PC5']
    for pc in pcs:
        X = sm.add_constant(pc_bm.loc[:mid_date,pc].shift(1).dropna())
        model = sm.OLS(pc_ret.loc[pc_ret.index[1]:mid_date,pc], X)
        results = model.fit()
        X2 = sm.add_constant(pc_bm.shift(1).loc[mid_date:,pc])
        predictions[pc] = results.predict(X2)
        r2_methods['pc_and_bm'][pc] = r2_score(y_true=pc_ret.loc[mid_date:,pc], y_pred=results.predict(X2))
    predictions = pd.DataFrame(predictions)
    total_r2_pc_and_bm = total_r2(predictions, pc_ret.loc[mid_date:,pcs])[0]

    # rigde regression of all pc valuation ratios on all pc returns
    r2_methods['pc_and_pc_bm_ridge'] = {}
    predictions = {}
    for pc in pcs:
        X = sm.add_constant(pc_bm.loc[:mid_date,'PC1':'PC5'].shift(1).dropna())
        model = sm.OLS(pc_ret.loc[pc_ret.index[1]:mid_date,pc], X)
        results = model.fit_regularized(L1_wt=0, alpha=0.2)
        X2 = sm.add_constant(pc_bm.shift(1).loc[mid_date:,'PC1':'PC5'])
        predictions[pc] = results.predict(X2)
        r2_methods['pc_and_pc_bm_ridge'][pc] = r2_score(y_true=pc_ret.loc[mid_date:,pc], y_pred=results.predict(X2))
    predictions = pd.DataFrame(predictions)
    total_r2_pc_and_pc_bm_ridge = total_r2(predictions, pc_ret.loc[mid_date:,pcs])[0]

    # lasso regression of all pc valuation ratios on all pc returns
    r2_methods['pc_and_pc_bm_lasso'] = {}
    predictions = {}
    for pc in pcs:
        X = sm.add_constant(pc_bm.loc[:mid_date,'PC1':'PC5'].shift(1).dropna())
        model = sm.OLS(pc_ret.loc[pc_ret.index[1]:mid_date,pc], X)
        results = model.fit_regularized(L1_wt=1, alpha=0.004)
        X2 = sm.add_constant(pc_bm.shift(1).loc[mid_date:,'PC1':'PC5'])
        predictions[pc] = results.predict(X2)
        r2_methods['pc_and_pc_bm_lasso'][pc] = r2_score(y_true=pc_ret.loc[mid_date:,pc], y_pred=results.predict(X2))
    predictions = pd.DataFrame(predictions)
    total_r2_pc_and_pc_bm_lasso = total_r2(predictions, pc_ret.loc[mid_date:,pcs])[0]

    # regression of each factor valuation ratio on each factor return
    r2_methods['raw_factors_own'] = {}
    predictions = {}
    for f in factors:
        X = sm.add_constant(fullsample_bm.loc[:mid_date,f].shift(1).dropna())
        model = sm.OLS(fullsample_ret.loc[fullsample_ret.index[1]:mid_date,f], X)
        results = model.fit()
        X2 = sm.add_constant(fullsample_bm.shift(1).loc[mid_date:,f])
        predictions[f] = results.predict(X2)
        r2_methods['raw_factors_own'][f] = r2_score(y_true=fullsample_ret.loc[mid_date:,f], y_pred=results.predict(X2))
    predictions = pd.DataFrame(predictions)
    total_r2_raw_factors_own = total_r2(predictions, fullsample_ret.loc[mid_date:])[0]

    # create summary table for r2 of various methods
    r2_methods_means = {}
    for k in r2_methods.keys():
        r2_methods_means[k] = pd.Series(r2_methods[k]).describe()
    r2_methods_means = pd.DataFrame(r2_methods_means).T
    r2_methods_means['OOS total R2'] = [total_r2_raw_factors, total_r2_pc_and_bm, total_r2_pc_and_pc_bm_ridge, total_r2_pc_and_pc_bm_lasso, total_r2_raw_factors_own]

    if export_results:
        # collect results for r2 of various methods and export to latex
        r2_various_methods = r2_methods_means.loc[:,['OOS total R2','mean','50%','std']] * 100
        r2_various_methods.columns = ['OOS total $R^2$','Mean', 'Median', 'Std.']
        r2_various_methods.index = ['50 Anom, BM of Anom, OLS', '5 PCs, Own BM', '5 PCs, BM of PCs, Ridge 1DoF', '5 PCs, BM of PCs, Lasso-OLS 1DoF', '50 Anom, Own BM']
        create_latex_table_from_df(r2_various_methods,
                                'r2_various_methods', 
                                'Out-of-sample R2 of various forecasting methods',
                                'The table reports the monthly OOS total R2 as well as mean, median, and standard deviation of OOS R2 for individual anomaly portfolios for various forecasting methods. The first column gives the set of assets which are directly forecast, the predictive variables used, and the forecasting method. When omitted, the method is ordinary least squares',
                                )
    
    def calc_stambaugh_bias():
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

# %%
