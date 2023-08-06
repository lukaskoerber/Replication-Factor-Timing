# Replication: Factor Timing
This repo is replicating the following paper:

**Valentin Haddad, Serhiy Kozak, and Shrihari Santosh**, *Factor timing*, 2020, The Review of Financial Studies 33.5, p. 1980-2018. Available at SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2945667.

> The optimal factor timing portfolio is equivalent to the stochastic discount factor. We propose and implement a method to characterize both empirically. Our approach imposes restrictions on the dynamics of expected returns which lead to an economically plausible SDF. Market-neutral equity factors are strongly and robustly predictable. Exploiting this predictability leads to substantial improvement in portfolio performance relative to static factor investing. The variance of the corresponding SDF is larger, more variable over time, and exhibits different cyclical behavior than estimates ignoring this fact. These results pose new challenges for theories that aim to match the cross-section of stock returns.

The dataset used in the paper is available at Serhiy Kozak's [webpage](https://www.serhiykozak.com/data).

---

**Main.py**: Code to obtain the graphics in the results section.

**FT-17APR19-monthly**: Data used.

---


# Result:

### Conditional Variance of the SDF
![sdf_variance](results_export/sdf_variance.png "SDF Variance")
### Conditional Variance of the SDF and Inflation
![sdf_var_inflation](results_export/sdf_var_inflation.png "sdf_var_inflation")

### Correlation of Factor Investing and Factor Timing
![strategie_correlation](results_export/strategie_correlation.png "strategie_correlation")

## Return Predictions
### Predictability of Principal Components
![r2_pca_in_oos](results_export/r2_pca_in_oos.png "PC Predictability")
### PC1-PC5 Prediction 
![pc1_returns](results_export/pc1_returns.png "Factor Predictability")
![pc2_returns](results_export/pc2_returns.png "Factor Predictability")
![pc3_returns](results_export/pc3_returns.png "Factor Predictability")
![pc4_returns](results_export/pc4_returns.png "Factor Predictability")
![pc5_returns](results_export/pc5_returns.png "Factor Predictability")

