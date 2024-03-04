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
from sklearn.metrics import r2_score
import statsmodels.api as sm

plotly_colors = px.colors.qualitative.Plotly

parameters = {
    # data variables
    'data_path': './FT-17APR19-monthly/FT-17APR19-monthly/',
    'data_path_daily': './FT-17APR19-daily/FT-17APR19-daily/',
    'date_format': '%m/%Y',
    'date_format_day': '%m/%d/%Y',
    'start_date': '1974-01-01',
    'oos_split_date': '1996-01-01', # '1996-01-01'
    'end_date': '2017-12-31',

    # hyperparameters for processing and analysis
    'market_var': 'size',
    'market_bm_logged': True,
    'market_excess_returns': True,
    'num_pca': 5,
    'quintiles': False,
    'subtract_market': True,
    'scale_for_market_var': True,
    'holding_period': 1,
    'export_results': True,
    'log_entry': False,

    # bmc10 = book/market, n10 = number of firms, ret10 = returns, totalme10 = market cap
    'datatypes': ['bmc10', 'n10', 'ret10', 'totalme10'],
    'factors': None, 
}

# load data from the directory and exclude some factors
list_of_files = os.listdir(path=parameters['data_path'])
name_map = pd.read_csv('./data/name_map.csv')
name_map = name_map.set_index('short_name')

# extract the factor names from the file names in the data directory
parameters['factors'] = []
import re
for t in list_of_files:
    txt = re.split('_|\\.',t)[1]
    if txt not in parameters['factors'] and txt in name_map.index:
        parameters['factors'].append(txt)

# %% load data

from utils import load_data

all_ret, all_ret_daily, all_n, all_me, all_bm, all_book, all_n_sum, all_n_sum_m, rf_rate, rf_rate_daily, inf_rate = load_data(parameters)

# %%
from utils import create_value_weights, build_anomaly_returns, build_market, plot_real_and_predicted_returns, create_latex_table_from_df

# create value weights 
all_me_sum = create_value_weights(all_me, parameters['factors'])

# create weightings for the market portfolio returns
me_weights = all_me.copy()
for f in parameters['factors']:
    me_weights.loc[:,[f+'_p'+str(i) for i in range(1,11)]] = all_me.loc[:,[f+'_p'+str(i) for i in range(1,11)]].divide(all_me_sum.loc[:,f+'_sum'], axis=0)

# build anomaly returns
abnormal_ret, abnormal_bm, abnormal_ret_daily = build_anomaly_returns(all_ret, all_ret_daily, all_me, all_bm, parameters)

# build market returns
market_ret, market_bm, market_ret_daily = build_market(all_ret, all_ret_daily, me_weights, rf_rate, rf_rate_daily, all_book, all_me_sum, parameters)

# %% demarket anomaly returns

from utils import total_r2

# regress anomaly returns on market returns and multiply beta times market return
betas = {}
fullsample_ret = []
for f in parameters['factors']:
    X = sm.add_constant(market_ret.loc[:parameters['oos_split_date']])
    model = sm.OLS(abnormal_ret.loc[:parameters['oos_split_date']][f], X) # regression of anomaly returns on market returns
    results = model.fit() # fit the model
    betas[f] = results.params['market_ret'] # save the beta
    tmp_ret = abnormal_ret[f] - market_ret * betas[f] * int(parameters['subtract_market'])
    tmp_ret.name = f
    fullsample_ret.append(tmp_ret)
ret_betas = pd.Series(betas)
fullsample_ret = pd.concat(fullsample_ret, axis=1)

# scale with standard deviation
insample_ret_std = fullsample_ret.loc[:parameters['oos_split_date']].std().copy()
insample_market_ret_std = market_ret.loc[:parameters['oos_split_date']].std()
if parameters['scale_for_market_var']:
    fullsample_ret = fullsample_ret.divide(insample_ret_std) * insample_market_ret_std
else:
    fullsample_ret = fullsample_ret.divide(insample_ret_std)

# regress anomaly bm on market bm and multiply beta times market bm
betas = {}
fullsample_bm = []
if parameters['market_bm_logged']:
    market_bm_reg = market_bm
else:
    market_bm_reg = market_bm.copy()
    market_bm_reg = np.exp(market_bm_reg)

# regress anomaly bm on market bm and multiply beta times market bm
for f in parameters['factors']:
    X = sm.add_constant(market_bm_reg.loc[:parameters['oos_split_date']])
    model = sm.OLS(abnormal_bm.loc[:parameters['oos_split_date']][f], X) # regression of anomaly returns on market returns
    results = model.fit() # fit the model
    betas[f] = results.params['market_bm'] # save the beta
    # dont subtract market bm from anomaly bm as unclear in the paper
    tmp_bm = abnormal_bm[f] - market_bm_reg * betas[f] * int(parameters['subtract_market']) * 0 
    tmp_bm.name = f
    fullsample_bm.append(tmp_bm)
bm_betas = pd.Series(betas)
fullsample_bm = pd.concat(fullsample_bm, axis=1)

# different bm regressions, scale with standard deviation
insample_bm_std = fullsample_bm.loc[:parameters['oos_split_date']].std().copy()
insample_market_bm_std = market_bm_reg.loc[:parameters['oos_split_date']].std()
if parameters['scale_for_market_var']:
    fullsample_bm = fullsample_bm.divide(insample_bm_std) * insample_market_bm_std
else:
    fullsample_bm = fullsample_bm.divide(insample_bm_std)

#%% pca

# run pca on insample returns
pca = PCA()
pca.fit(fullsample_ret.loc[:parameters['oos_split_date']])
pca_expl = pd.Series(pca.explained_variance_ratio_)
pca_expl.index = ['PC'+str(i+1) for i in range(len(pca_expl))]
pca_comp = pd.DataFrame(pca.components_.T, index=parameters['factors'], columns=['PC'+str(i+1) for i in range(len(fullsample_ret.columns))])

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

if parameters['export_results']:
    # export pca properties to latex
    pca_table = pd.concat([pca_expl.iloc[:10],pca_expl.iloc[:10].cumsum()],axis=1).rename(columns={0:'\% var. explained',1:'Cumulative'}).T * 100
    create_latex_table_from_df(pca_table,
                            'pca_table',
                            'Percentage of variance explained by anomaly PCs', 
                            'Percentage of variance explained by each PC of the 50 anomaly strategies.', 
                            1)

# %% market: predictive regressions

# run predictive regressions
# market regressions
if parameters['market_excess_returns']:
    reg_market = pd.concat([market_ret,market_bm.shift(1)], axis=1)
else:
    tmp_market_ret = market_ret - rf_rate.loc[market_ret.index]
    tmp_market_ret.name = 'market_ret'
    reg_market = pd.concat([tmp_market_ret, market_bm.shift(1)], axis=1)
reg_market = reg_market.dropna()
reg_market_in = reg_market.loc[:parameters['oos_split_date']]
reg_market_out = reg_market.loc[parameters['oos_split_date']:]

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
r2_oos_m = 1-((reg_market_out['market_ret']-oos_pred_market.loc[parameters['oos_split_date']:]).var()/(reg_market_out['market_ret'].var()))
r2_in_m = 1-((reg_market['market_ret']-full_pred_market).var()/(reg_market['market_ret'].var()))

#%% pcs: predictive regressions

# construct the returns and bm ratios of pc portfolios
pc_ret = fullsample_ret.dot(pca_comp)
pc_bm = fullsample_bm.dot(pca_comp)
pc_ret_daily = abnormal_ret_daily.dot(pca_comp)

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
    model_in = smf.ols('PC'+str(i+1)+'_ret ~ '+'PC'+str(i+1)+'_bm', data=tmp_data.loc[:parameters['oos_split_date']])
    results_in = model_in.fit()
    oos_pred = results_in.predict(tmp_data.loc[:,'PC'+str(i+1)+'_bm'])
    oos_pred.name = 'PC'+str(i+1)
    
    # save results
    in_preds.append(in_pred)
    oos_preds.append(oos_pred)
    r2_oos = 1-((tmp_data.loc[parameters['oos_split_date']:,'PC'+str(i+1)+'_ret']-oos_pred.loc[parameters['oos_split_date']:]).var()/tmp_data.loc[parameters['oos_split_date']:,'PC'+str(i+1)+'_ret'].var())
    
    pca_res['r2_in']['PC'+str(i+1)] = results.rsquared
    pca_res['r2_oos']['PC'+str(i+1)] = r2_oos
    pca_res['beta_in']['PC'+str(i+1)] = results.params['PC'+str(i+1)+'_bm']
    pca_res['beta_oos']['PC'+str(i+1)] = results_in.params['PC'+str(i+1)+'_bm']
    pca_res['t_test_in']['PC'+str(i+1)] = results.t_test('PC'+str(i+1)+'_bm=0').summary_frame()['t'].iloc[0]
    pca_res['t_test_oos']['PC'+str(i+1)] = results_in.t_test('PC'+str(i+1)+'_bm=0').summary_frame()['t'].iloc[0]

# convert to dataframe
pca_res = pd.DataFrame(pca_res)
in_pred = pd.concat(in_preds, axis=1)
oos_pred = pd.concat(oos_preds, axis=1)
pca_res = pca_res.rename(index={'market':'MKT'})

# %% 
from utils import plot_r2

if parameters['export_results']:
    # plot pca results and save to pdf
    plot_r2(pca_res, pca_expl)

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
    market_plot = pd.concat([reg_market['market_ret'],full_pred_market,oos_pred_market.loc[parameters['oos_split_date']:]],axis=1)
    plot_real_and_predicted_returns(market_plot, ['Realized', 'Predicted(IS)', 'Predicted(OOS)'], '(a) Market', 'market_returns')

    pc1_plot = pd.concat([pc_ret['PC1'],in_pred['PC1'],oos_pred.loc[parameters['oos_split_date']:,'PC1']],axis=1)
    plot_real_and_predicted_returns(pc1_plot, ['Realized', 'Predicted(IS)', 'Predicted(OOS)'], '(b) PC1', 'pc1_returns')

    pc2_plot = pd.concat([pc_ret['PC2'],in_pred['PC2'],oos_pred.loc[parameters['oos_split_date']:,'PC2']],axis=1)
    plot_real_and_predicted_returns(pc2_plot, ['Realized', 'Predicted(IS)', 'Predicted(OOS)'], '(c) PC2', 'pc2_returns')

    pc3_plot = pd.concat([pc_ret['PC3'],in_pred['PC3'],oos_pred.loc[parameters['oos_split_date']:,'PC3']],axis=1)
    plot_real_and_predicted_returns(pc3_plot, ['Realized', 'Predicted(IS)', 'Predicted(OOS)'], '(d) PC3', 'pc3_returns')

    pc4_plot = pd.concat([pc_ret['PC4'],in_pred['PC4'],oos_pred.loc[parameters['oos_split_date']:,'PC4']],axis=1)
    plot_real_and_predicted_returns(pc4_plot, ['Realized', 'Predicted(IS)', 'Predicted(OOS)'], '(e) PC4', 'pc4_returns')

    pc5_plot = pd.concat([pc_ret['PC5'],in_pred['PC5'],oos_pred.loc[parameters['oos_split_date']:,'PC5']],axis=1)
    plot_real_and_predicted_returns(pc5_plot, ['Realized', 'Predicted(IS)', 'Predicted(OOS)'], '(f) PC5', 'pc5_returns')
else:
    px.bar(pca_res, x=pca_res.index, y=['r2_in','r2_oos'], barmode='group', title='In-sample and out-of-sample R-squared', template='plotly_white').show()

if parameters['log_entry']:
    # create log entry, important for the different robustness tests
    total_r2_res, significant = total_r2(oos_pred.loc[parameters['oos_split_date']:,'PC'+str(1):'PC'+str(parameters['num_pca'])], pc_ret.loc[:,'PC'+str(1):'PC'+str(parameters['num_pca'])])
    total_r2_log = pd.read_csv('./result_export/total_r2_log.csv', index_col=0)
    setting_idx = str(parameters['num_pca'])+str(int(parameters['quintiles']))+str(int(parameters['subtract_market']))+str(int(parameters['scale_for_market_var']))+str(parameters['holding_period'])
    total_r2_log.loc[setting_idx,:] = pd.Series(
        [parameters['quintiles'], parameters['holding_period'], parameters['num_pca'], parameters['holding_period']==1, parameters['subtract_market'], parameters['scale_for_market_var'], round(total_r2_res * 100,2), significant],
        index=['anom_port_sort','holding_period','num_pc','monthly_pc','market_adj_ret','scaled_var','total_r2','num_sign_pc'])
    total_r2_log.to_csv('./result_export/total_r2_log.csv')


# %% back to implied prediction of individual factor returns
    
ch_pred_in = in_pred.loc[:, 'PC'+str(1):'PC'+str(parameters['num_pca'])] @ pca_comp.loc[:, 'PC'+str(1):'PC'+str(parameters['num_pca'])].T
ch_pred_oos = oos_pred.loc[parameters['oos_split_date']:, 'PC'+str(1):'PC'+str(parameters['num_pca'])] @ pca_comp.loc[:, 'PC'+str(1):'PC'+str(parameters['num_pca'])].T

# calculate factor predictability after back to factor projection
r2 = {'in':{},'oos':{}}
for factor in ch_pred_in.columns:
    r2['in'][factor] = r2_score(y_true=fullsample_ret.loc[ch_pred_in.index,factor], y_pred=ch_pred_in[factor])
    r2['oos'][factor] = r2_score(y_true=fullsample_ret.loc[ch_pred_oos.index,factor], y_pred=ch_pred_oos[factor])
factor_predictability = pd.DataFrame(r2).sort_values('oos', ascending=False)

if parameters['export_results']:
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
    fig.show()
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
if parameters['export_results']:
    # combine the four factors to a plot using make subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=('(a) Size', '(b) Value', '(c) Momentum', '(d) Profitability'))
    fig.add_trace(go.Scatter(x=ch_pred_in.index, y=ch_pred_in['size'].rolling(6, center=True).mean(), name='(a) Size'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ch_pred_in.index, y=ch_pred_in['value'].rolling(6, center=True).mean(), name='(b) Value'), row=1, col=2)
    fig.add_trace(go.Scatter(x=ch_pred_in.index, y=ch_pred_in['mom'].rolling(6, center=True).mean(), name='(c) Momentum'), row=2, col=1)
    fig.add_trace(go.Scatter(x=ch_pred_in.index, y=ch_pred_in['roa'].rolling(6, center=True).mean(), name='(d) Profitability'), row=2, col=2)
    fig.update_layout(height=600, width=800, title_text="Anomaly expected returns", showlegend=False, template='plotly_white')
    fig.update_traces(line=dict(color=plotly_colors[0], width=2))
    fig.show()
    pio.full_figure_for_development(fig, warn=False)
    fig.write_image("./result_export/anomaly_exp_rets.pdf",engine="kaleido")
    fig.write_image("./result_export/anomaly_exp_rets.png",engine="kaleido")

# %%
    
from utils import create_weights, eval_strategy, expected_utility

if not parameters['quintiles'] and parameters['holding_period']==1:
    # build portfolios for static factor investing
    # crate evaluation data
    tmp_portfolio_ret = abnormal_ret.copy()
    portfolio_ret = tmp_portfolio_ret @ pca_comp.loc[:,'PC1':'PC'+str(parameters['num_pca'])]
    portfolio_ret_in = portfolio_ret
    portfolio_ret_oos = portfolio_ret.loc[parameters['oos_split_date']:,:]

    # calculate market excess returns, if market returns are not excess returns subtract risk free rate
    market_ret_eval = market_ret - rf_rate.loc[market_ret.index] * (1-int(parameters['market_excess_returns']))
    market_ret_eval.name = 'market_ret'

    # combine market returns and portfolio returns
    raw_z_in = pd.concat([market_ret_eval, portfolio_ret_in], axis=1, join='inner')
    raw_z_out = pd.concat([market_ret_eval, portfolio_ret_oos], axis=1, join='inner')

    if parameters['scale_for_market_var']:
        market_ret_z = reg_market['market_ret'].copy()
    else:
        market_ret_z = reg_market['market_ret'].copy() / insample_market_ret_std
    norm_weights = 0

    # create the z vectors over time
    df_z_daily = pd.concat([market_ret_daily, pc_ret_daily.loc[:,'PC1':'PC'+str(parameters['num_pca'])]], axis=1)
    df_z_monthly_var = df_z_daily.groupby(pd.Grouper(freq='M')).var().iloc[:-1]
    df_z_monthly_var.index = df_z_monthly_var.index + pd.DateOffset(days=1)
    df_z_real = pd.concat([market_ret_z, pc_ret.loc[:,'PC1':'PC'+str(parameters['num_pca'])]], axis=1)
    df_z_pred =  pd.concat([oos_pred_market, oos_pred.loc[:, 'PC1':'PC'+str(parameters['num_pca'])]], axis=1)
    errors_z = df_z_real.subtract(df_z_pred)
    # construct sigma matrix
    sigma = (df_z_real-df_z_pred).loc[:parameters['oos_split_date']].cov()

    # create weights
    pred_z_in = pd.DataFrame(pd.concat([market_ret_z, pc_ret.loc[:,'PC1':'PC'+str(parameters['num_pca'])]], axis=1).mean(),columns=['return']).T
    pred_z_oos = pd.DataFrame(pd.concat([market_ret_z, pc_ret.loc[:parameters['oos_split_date'], 'PC1':'PC'+str(parameters['num_pca'])]], axis=1, join='inner').mean(),columns=['return']).T
    # l2 norm weights
    weights_in_fi = create_weights(sigma,pred_z_in,normalize=norm_weights)
    weights_oos_fi = create_weights(sigma,pred_z_oos,normalize=norm_weights)
    # evaluate strategy
    res_strat_fi = eval_strategy(weights_in_fi, weights_oos_fi, raw_z_in, raw_z_out, 'Factor investing')
    res_strat_fi = pd.concat([res_strat_fi,expected_utility(pd.DataFrame(pred_z_in.mean(), columns=['weights']),sigma, 'Factor investing')])

    # build portfolios for market timing
    pred_z_in = pd.concat([full_pred_market,pc_ret.loc[:,'PC1':'PC'+str(parameters['num_pca'])]], axis=1, join='inner')
    pred_z_in.loc[:,'PC1':'PC'+str(parameters['num_pca'])] = np.ones((len(pred_z_in),parameters['num_pca'])) * pc_ret.loc[:,'PC1':'PC'+str(parameters['num_pca'])].mean().to_numpy()
    pred_z_oos = pd.concat([oos_pred_market, pc_ret.loc[parameters['oos_split_date']:, 'PC1':'PC'+str(parameters['num_pca'])]], axis=1, join='inner')
    pred_z_oos.loc[:,'PC1':'PC'+str(parameters['num_pca'])] = np.ones((len(pred_z_oos),parameters['num_pca'])) * pc_ret.loc[:parameters['oos_split_date'],'PC1':'PC'+str(parameters['num_pca'])].mean().to_numpy()
    weights_in_mt = create_weights(sigma,pred_z_in,normalize=norm_weights)
    weights_oos_mt = create_weights(sigma,pred_z_oos,normalize=norm_weights)
    # evaluate strategy
    res_strat_mt = eval_strategy(weights_in_mt, weights_oos_mt, raw_z_in, raw_z_out, 'Market timing', dynamic=True, weights_compare_in=weights_in_fi, weights_compare_oos=weights_in_fi)
    res_strat_mt = pd.concat([res_strat_mt,expected_utility(pred_z_in, sigma, 'Market timing', dynamic=True)])

    # build portfolios for factor timing
    pred_z_in = pd.concat([full_pred_market,in_pred.loc[:,'PC1':'PC'+str(parameters['num_pca'])]], axis=1)
    pred_z_oos = pd.concat([oos_pred_market, oos_pred.loc[:, 'PC1':'PC'+str(parameters['num_pca'])]], axis=1, join='inner')
    weights_in_ft = create_weights(sigma,pred_z_in,normalize=norm_weights)
    weights_oos_ft = create_weights(sigma,pred_z_oos,normalize=norm_weights)
    # evaluate strategy
    res_strat_ft = eval_strategy(weights_in_ft, weights_oos_ft, raw_z_in, raw_z_out, 'Factor timing' ,dynamic=True, weights_compare_in=weights_in_fi, weights_compare_oos=weights_in_fi)
    res_strat_ft = pd.concat([res_strat_ft,expected_utility(pred_z_in, sigma, 'Factor timing', dynamic=True)])

    # build portfolios for anomaly timing
    pred_z_in = pd.concat([market_ret_z,in_pred.loc[:,'PC1':'PC'+str(parameters['num_pca'])]], axis=1)
    pred_z_in.loc[:,'market_ret'] = market_ret_z.mean()
    pred_z_oos = pd.concat([market_ret_z, oos_pred.loc[:, 'PC1':'PC'+str(parameters['num_pca'])]], axis=1, join='inner')
    pred_z_oos.loc[:,'market_ret'] = market_ret_z.loc[:parameters['oos_split_date']].mean()
    weights_in_at = create_weights(sigma,pred_z_in,normalize=norm_weights)
    weights_oos_at = create_weights(sigma,pred_z_oos,normalize=norm_weights)
    # evaluate strategy
    res_strat_at = eval_strategy(weights_in_at, weights_oos_at, raw_z_in, raw_z_out, 'Anomaly timing', dynamic=True, weights_compare_in=weights_in_fi, weights_compare_oos=weights_in_fi)
    res_strat_at = pd.concat([res_strat_at,expected_utility(pred_z_in, sigma, 'Anomaly timing', dynamic=True)])

    # build portfolios for pure anomaly timing
    pred_z_in = pd.concat([market_ret_z, in_pred.loc[:,'PC1':'PC'+str(parameters['num_pca'])]-in_pred.loc[:,'PC1':'PC'+str(parameters['num_pca'])].mean()], axis=1)
    pred_z_in.loc[:,'market_ret'] = 0
    pred_z_oos = pd.concat([market_ret_z, oos_pred.loc[:,'PC1':'PC'+str(parameters['num_pca'])]-oos_pred.loc[:parameters['oos_split_date'],'PC1':'PC'+str(parameters['num_pca'])].mean()], axis=1, join='inner')
    pred_z_oos.loc[:,'market_ret'] = 0
    weights_in_pat = create_weights(sigma,pred_z_in,normalize=norm_weights)
    weights_oos_pat = create_weights(sigma,pred_z_oos,normalize=norm_weights)
    # evaluate strategy
    res_strat_pat = eval_strategy(weights_in_pat, weights_oos_pat, raw_z_in, raw_z_out, 'Pure anom. timing', dynamic=True, weights_compare_in=weights_in_fi, weights_compare_oos=weights_in_fi)
    res_strat_pat = pd.concat([res_strat_pat,expected_utility(pred_z_in, sigma, 'Pure anom. timing', dynamic=True)])

    if parameters['export_results']:
        res_strat_table = pd.concat([res_strat_fi, res_strat_mt, res_strat_ft, res_strat_at, res_strat_pat], axis=1)
        res_strat_table = res_strat_table.loc[['IS Return', 'OOS Return', 'IS Sharpe ratio', 'OOS Sharpe ratio', 'IS Inf. ratio', 'OOS Inf. ratio', 'Expected utility'],:]
        create_latex_table_from_df(res_strat_table,
                                'strat_table',
                                'Performance of various portfolio strategies',
                                'The table reports the unconditional Sharpe ratio, information ratio, and average mean-variance utility of five strategies: (i) static factor investing strategy, based on unconditional estimates of E [Zt]; (ii) market timing strategy which uses forecasts of the market return based on Table 2 but sets expected returns on the PC equal to unconditional values; (iii) full factor timing strategy including predictability of the PCs and the market; (iv) anomaly timing strategy which uses forecasts of the PCs based on Table 2 but sets expected returns on the market to unconditional values; and (v) pure anomaly timing strategy sets the weight on the market to zero and invests in anomalies proportional to the deviation of their forecast to its unconditional average, Et[Zt+1] - E [Zt]. All strategies assume a homoskedastic conditional covariance matrix, estimated as the covariance of forecast residuals. Information ratios are calculated relative to the static strategy. Out-of-sample(OOS) values are based on split-sample analysis with all parameters estimated using the first half of the data', 
                                )

    # variation of market timing and factor timing
    var_sdf = pd.DataFrame(columns=['market_timing', 'factor_timing'], index=weights_in_mt.columns)
    var_sdf_fi = (weights_in_fi.T @ sigma @ weights_in_fi).values[0,0]*12
    var_sdf['market_timing'] = np.diag(weights_in_mt.T @ sigma @ weights_in_mt)
    var_sdf['factor_timing'] = np.diag(weights_in_ft.T @ sigma @ weights_in_ft)
    var_sdf = var_sdf * 12
    if parameters['export_results']:
        # plot variation of market timing and factor timing
        fig = px.line(var_sdf.rolling(6).mean(), title='Conditional variance of SDFs', labels={'value':'SDF Variance', 'variable':'', 'date':'Time'})
        fig.update_layout(template='plotly_white',
                            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=0.6),
                            showlegend=True,
                            width=900, height=500,
                            margin=dict(l=10, r=10))
        fig.show()
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
        fig.show()
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
    if parameters['export_results']:
        # plot correlation of factor investing and factor timing
        fig = px.line(corr_sdf.rolling(6).mean(), title='Correlation of factor investing and factor timing', labels={'value':'Conditional correlation', 'variable':'', 'date':'Time'})
        fig.update_layout(template='plotly_white',
                        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=0.6),
                        showlegend=False,
                        width=900, height=500,
                        margin=dict(l=10, r=10))
        fig.update_yaxes(range=[0.2, 1.0])
        fig.show()
        pio.full_figure_for_development(fig, warn=False)
        fig.write_image("./result_export/strategie_correlation.pdf",engine="kaleido")
        fig.write_image("./result_export/strategie_correlation.png",engine="kaleido")

    # different methods for predicting anomaly returns
    # regression of all factor valuation ratios on all factor returns
    r2_methods = {'raw_factors':{}}
    predictions = {}
    for f in parameters['factors']:
        X = sm.add_constant(fullsample_bm.loc[:parameters['oos_split_date']].shift(1).dropna())
        model = sm.OLS(fullsample_ret.loc[fullsample_ret.index[1]:parameters['oos_split_date'],f], X)
        results = model.fit()
        X2 = sm.add_constant(fullsample_bm.shift(1).loc[parameters['oos_split_date']:])
        predictions[f] = results.predict(X2)
        r2_methods['raw_factors'][f] = r2_score(y_true=fullsample_ret.loc[parameters['oos_split_date']:,f], y_pred=results.predict(X2))
    predictions = pd.DataFrame(predictions)
    total_r2_raw_factors = total_r2(predictions, fullsample_ret.loc[parameters['oos_split_date']:])[0]

    # regression of own pc valuation ratios on own pc returns
    r2_methods['pc_and_bm'] = {}
    predictions = {}
    pcs = ['PC1','PC2','PC3','PC4','PC5']
    for pc in pcs:
        X = sm.add_constant(pc_bm.loc[:parameters['oos_split_date'],pc].shift(1).dropna())
        model = sm.OLS(pc_ret.loc[pc_ret.index[1]:parameters['oos_split_date'],pc], X)
        results = model.fit()
        X2 = sm.add_constant(pc_bm.shift(1).loc[parameters['oos_split_date']:,pc])
        predictions[pc] = results.predict(X2)
        r2_methods['pc_and_bm'][pc] = r2_score(y_true=pc_ret.loc[parameters['oos_split_date']:,pc], y_pred=results.predict(X2))
    predictions = pd.DataFrame(predictions)
    total_r2_pc_and_bm = total_r2(predictions, pc_ret.loc[parameters['oos_split_date']:,pcs])[0]

    # rigde regression of all pc valuation ratios on all pc returns
    r2_methods['pc_and_pc_bm_ridge'] = {}
    predictions = {}
    for pc in pcs:
        X = sm.add_constant(pc_bm.loc[:parameters['oos_split_date'],'PC1':'PC5'].shift(1).dropna())
        model = sm.OLS(pc_ret.loc[pc_ret.index[1]:parameters['oos_split_date'],pc], X)
        results = model.fit_regularized(L1_wt=0, alpha=0.2)
        X2 = sm.add_constant(pc_bm.shift(1).loc[parameters['oos_split_date']:,'PC1':'PC5'])
        predictions[pc] = results.predict(X2)
        r2_methods['pc_and_pc_bm_ridge'][pc] = r2_score(y_true=pc_ret.loc[parameters['oos_split_date']:,pc], y_pred=results.predict(X2))
    predictions = pd.DataFrame(predictions)
    total_r2_pc_and_pc_bm_ridge = total_r2(predictions, pc_ret.loc[parameters['oos_split_date']:,pcs])[0]

    # lasso regression of all pc valuation ratios on all pc returns
    r2_methods['pc_and_pc_bm_lasso'] = {}
    predictions = {}
    for pc in pcs:
        X = sm.add_constant(pc_bm.loc[:parameters['oos_split_date'],'PC1':'PC5'].shift(1).dropna())
        model = sm.OLS(pc_ret.loc[pc_ret.index[1]:parameters['oos_split_date'],pc], X)
        results = model.fit_regularized(L1_wt=1, alpha=0.004)
        X2 = sm.add_constant(pc_bm.shift(1).loc[parameters['oos_split_date']:,'PC1':'PC5'])
        predictions[pc] = results.predict(X2)
        r2_methods['pc_and_pc_bm_lasso'][pc] = r2_score(y_true=pc_ret.loc[parameters['oos_split_date']:,pc], y_pred=results.predict(X2))
    predictions = pd.DataFrame(predictions)
    total_r2_pc_and_pc_bm_lasso = total_r2(predictions, pc_ret.loc[parameters['oos_split_date']:,pcs])[0]

    # regression of each factor valuation ratio on each factor return
    r2_methods['raw_factors_own'] = {}
    predictions = {}
    for f in parameters['factors']:
        X = sm.add_constant(fullsample_bm.loc[:parameters['oos_split_date'],f].shift(1).dropna())
        model = sm.OLS(fullsample_ret.loc[fullsample_ret.index[1]:parameters['oos_split_date'],f], X)
        results = model.fit()
        X2 = sm.add_constant(fullsample_bm.shift(1).loc[parameters['oos_split_date']:,f])
        predictions[f] = results.predict(X2)
        r2_methods['raw_factors_own'][f] = r2_score(y_true=fullsample_ret.loc[parameters['oos_split_date']:,f], y_pred=results.predict(X2))
    predictions = pd.DataFrame(predictions)
    total_r2_raw_factors_own = total_r2(predictions, fullsample_ret.loc[parameters['oos_split_date']:])[0]

    # create summary table for r2 of various methods
    r2_methods_means = {}
    for k in r2_methods.keys():
        r2_methods_means[k] = pd.Series(r2_methods[k]).describe()
    r2_methods_means = pd.DataFrame(r2_methods_means).T
    r2_methods_means['OOS total R2'] = [total_r2_raw_factors, total_r2_pc_and_bm, total_r2_pc_and_pc_bm_ridge, total_r2_pc_and_pc_bm_lasso, total_r2_raw_factors_own]

    if parameters['export_results']:
        # collect results for r2 of various methods and export to latex
        r2_various_methods = r2_methods_means.loc[:,['OOS total R2','mean','50%','std']] * 100
        r2_various_methods.columns = ['OOS total $R^2$','Mean', 'Median', 'Std.']
        r2_various_methods.index = ['50 Anom, BM of Anom, OLS', '5 PCs, Own BM', '5 PCs, BM of PCs, Ridge 1DoF', '5 PCs, BM of PCs, Lasso-OLS 1DoF', '50 Anom, Own BM']
        create_latex_table_from_df(r2_various_methods,
                                'r2_various_methods', 
                                'Out-of-sample R2 of various forecasting methods',
                                'The table reports the monthly OOS total R2 as well as mean, median, and standard deviation of OOS R2 for individual anomaly portfolios for various forecasting methods. The first column gives the set of assets which are directly forecast, the predictive variables used, and the forecasting method. When omitted, the method is ordinary least squares',
                                )
    
# %%
