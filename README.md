# Forecasting 2022 U.S. Unemployment Rate (U-3) using Classical Time Series Models

This project utilizes 2022 data from the Federal Reserve Economic Data (FRED) to forecast US unemployment rate (U-3). Data coverage includes all major areas of macroeconomic analysis: growth, inflation, employment, interest rates, exchange rates, etc. Independent variables are monthly supply of new housing and interest rate.

The data can be accessed at: https://fred.stlouisfed.org/

<details><summary>Decomposition and ACF Analysis:</summary>
</br>  

  
| Code | Description of variable measures, frequency, source | Time Period |
|:------------------|:----------------------------------------------------|:------------|
| MSACSRNSA     | U.S. Census Bureau, Monthly<br>The months' supply is the ratio of new houses for sale to new houses sold. |1963:1-2022:12|
| UNRATENSA     | U.S. Bureau of Labor Statistics, Monthly<br>16+ years old, reside in 50 states or the District of Columbia, do not reside in institutions or are on active duty in the Armed Forces. |1948:1-2022:12|
| FEDFUNDS      | Board of Governors of the Federal Reserve System (US), Monthly<br>Interest rate at which depository institutions trade federal funds with each other overnight. |1954:7-2022:12|

**1. Decomposition for Unemployment Training Data**

Because the seasonality does not increase with trend, additive decomposition (seasonal variation is relatively constant over time) is more suitable for this time series than multiplicative decomposition (seasonal variation increases over time).

<p align="center">
  <img src="https://github.com/user-attachments/assets/da8a63ab-defe-4b97-b96a-9d421552078b" width="500"/>
</p>

**2. Autocorrelation Feature Analysis**

The ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots identify the autocorrelation structure of a time series dataset. These plots help in determining the appropriate parameters for autoregressive integrated moving average (ARIMA) models.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8216d207-d92f-45f2-805d-7aff16354de0" width="500"/>
  <img src="https://github.com/user-attachments/assets/3f00e933-bf8e-44a6-a2af-1b22c69442d0" width="500"/>
</p>

From the ACF plot, the peak cuts off at lag 2, signifying an `MA(2)` process; in the PACF, the peak cuts off at lag 4 — `AR(4)`, while the seasonal component appears as an `MA(3)` model.
An autoregressive process with an order of 1 and a parameter of 0.4 is optimal.

</details>

## Model 1: Holt-Winters Exponential Smoothing

Leveraging [fpp2](https://otexts.com/fpp2/holt-winters.html), the additive Holt-Winters prediction function for time series with period length p is defined as:

$$\widehat{y_{t+h}} = l_t + hb_t +s_{t+h-m}$$

Finding the optimal values of `alpha` (defined as $l$ in the equation above), `beta`,
and/or `gamma`, the final auto-fitted model is:

$$\widehat{Y_{t+h}} = (a_t + b_t \cdot h) + s_{t + 1 + (h-1) \bmod p}, \quad
\begin{cases}
a_t = 0.8428519 \times (Y_t - s_{t-p}) + (1-0.8428519)(a_{t-1} + b_{t-1}) \newline
b_t = 0.02833799 \times (a_t - a_{t-1}) + (1-0.02833799) \times b_{t-1} \newline
s_t = 1 \times (Y_t - a_t) + (1-1) \times s_{t-p}
\end{cases}$$

The smoothing parameters and coefficients are:

<div align="center">

| Smoothing Parameters | Coefficients        |                     |
|:---------------------|:--------------------|:--------------------|
| `alpha` = 0.8429     | `a` = 4.07894326    | `s6` = 0.32706991   |
| `beta` = 0.0283      | `b` = -0.04468121   | `s7` = -0.01516064  |
| `gamma` = 1          | `s1` = 0.40741155   | `s8` = -0.40081732  |
|                      | `s2` = 0.25960977   | `s9` = -0.52094787  |
|                      | `s3` = 0.21651062   | `s10` = -0.52039932 |
|                      | `s4` = -0.21651062  | `s11` = -0.35669888 |
|                      | `s5` = 0.30746959   | `s12` = -0.42105674 |

</div>

The forecasted values are about 0.1 from the actual model in the first half of 2018, but the difference increases to more than 0.4 by January 2019.

<div align="center">
  
||      | Jan  | Feb  | Mar  | Apr  | May  | Jun  | Jul  | Aug  | Sep  | Oct  | Nov  | Dec  |
|:-|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|
|Predicted| 2018 |      | 4.44 | 4.24 | 3.72 | 3.77 | 4.16 | 4.13 | 3.78 | 3.32 | 3.15 | 3.11 | 3.23 |
|| 2019 | 3.96 |      |      |      |      |      |      |      |      |      |      |      |
|Validation Set| 2018 |      | 4.4  | 4.1  | 3.7  | 3.6  | 4.2  | 4.1  | 3.9  | 3.6  | 3.5  | 3.5  | 3.7  |
|| 2019 | 4.4  |      |      |      |      |      |      |      |      |      |      |      |

</div>

## Model 2: Auto Regressive Integrated Moving Average (ARIMA)

**1. Assessment of Mean Stationarity**

The ACF and PACF of the model, with regular and/or seasonal differencing, help identify which method minimizes trend in the mean. With the seasonal differencing of the regular differencing, the training data achieves mean stationarity.

<p align="center">
  <img src="https://github.com/user-attachments/assets/866ca40f-6ff5-4ced-90dc-bbe295677402" width="800"/>
</p>

From the ACF and PACF plots of regular and seasonal differencing, the ACF cuts off at approximately lag 1 (corresponding to an `MA(1)`), while the PACF cuts off at approximately lag 3 (corresponding to an `AR(3)`). Because there is seasonality, the model requires stationarity at low lags and at seasonal lags. In the seasonal plots, the seasonal differencing can be expressed as `AR(1)` and `MA(1)`. 
* The model—in ARIMA notation ARIMA(p,d,q)(P,D,Q)[F]—is: `ARIMA(3,1,1)(1,1,1)[12]`

**2. Ljung-Box Test for Residuals**

Applying the Ljung-Box test with 50 lags:

$$H_0: \rho_1 = \rho_2 = ... = \rho_{50} = 0$$

$$H_1: \text{not all } \rho_k \text{ up to lag } k \text{ are } 0$$

This tests whether or not the detrended and seasonally adjusted unemployment data is white noise. Assuming $\alpha=0.05$, the test returns `P-value: 0.82536`. Because P-value > $\alpha=0.05$, we do not reject the null hypothesis that the residuals are white noise. 

**3. Final Model**

The model in polynomial form—with all AR terms and differencing on the left hand side of the equal (=) sign and all the MA terms on the right hand side—is:

$$(1-0.6052 \times B-0.1334 \times B^2-0.0921 \times B^3) \times (1-B^{12}) \times (1-B) \times y_t = (1-0.5824 \times B+0.0495 \times B^2) \times (1-0.8786 \times B^{12}) \times w_t$$

Expanding the polynomial model, the final forecasting equation, with all the independent variables on the right hand side and only the dependent variable at time t on the left-hand-side, is:

$$\downarrow$$

$$(1-0.6052 \times B-0.1334 \times B^2-0.0921 \times B^3) \times (1-B^{12}) \times (1-B) \times y_t = (1-0.5824 \times B+0.0495 \times B^2) \times (1-0.8786 \times B^{12}) \times w_t$$

$$\downarrow$$

$$y_t \times (-0.0921 \times B^{16}-0.0413 \times B^{15}-0.4718 \times B^{14}+1.6052 \times B^{13}-B^{12}+ 0.0921 \times B^4+0.0413 \times B^3+0.4718 \times B^2-1.6052 \times B+1) = w_t \times (1-0.8786 \times B^{12}-0.5824 \times B+0.51169664 \times B^{13}+0.0495 \times B^2-0.0434907 \times B^{14})$$

$$\downarrow$$

$$-0.0921 \times y_{t-16}-0.0413 \times y_{t-15}-0.4718 \times y_{t-14}+1.6052 \times y_{t-13}- y_{t-12}+0.0921 \times y_{t-4}+0.0413 \times y_{t-3}+0.4718 \times y_{t-2}-1.6052 \times y_{t-1} + y_t = w_t-0.8786 \times w_{t-12}-0.5824 \times w_{t-1}+0.51169664 \times w_{t-13}+ 0.0495 \times w_{t-2}-0.0434907 \times w_{t-14}$$

$$\downarrow$$

$$y_t = 0.0921 \times y_{t-16}+0.0413 \times y_{t-15}+0.4718 \times y_{t-14}-1.6052 \times y_{t-13}+y_{t-12}-0.0921 \times y_{t-4}-0.0413 \times y_{t-3}-0.4718 \times y_{t-2}+1.6052 \times y_{t-1}-0.8786 \times w_{t-12}-0.5824 \times w_{t-1}+0.51169664 \times w_{t-13}+0.0495 \times w_{t-2}-0.0434907 \times w_{t-14}$$

**4. Stationarity (AR roots) and Invertibility (MA roots)**

A stationary time series is one whose statistical properties—such as variance and autocorrelation—remain constant over time. An invertible model ensures that the estimated coefficients provide meaningful insights.

* To check whether the model is stationary and invertible, find the modulus of the roots of the polynomial in B: `(1,-0.6052,-0.1334,-0.0921)`. Because the modulus of the roots (`1.140201 3.085883 3.085883`) are greater than 1, the process is stationary.

* Finding the modulus of the roots of the polynomial in B: `(1, -0.5824, 0.0495)`. Because the modulus of the roots (`2.0873509 9.6783056`) are greater than 1, the process is invertible.

In order to confirm that the coefficients generating the data are not 0, we utilize t-tests. The hypothesis are shown as follows:

$$H_0: \alpha_1 = \alpha_2 = \alpha_3 = \alpha_4 = \alpha_5 = \alpha_6 = 0$$

$$H_1: \alpha_1, \alpha_2, \alpha_3, \alpha_4, \alpha_5, \text{ and/or } \alpha_6 \text{ not equal to } 0$$

<div align="center">
  
|      | $\alpha_1$ | $\alpha_1$ | $\alpha_1$ | $\alpha_1$ | $\alpha_1$ | $\alpha_1$ |
|:-----|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|
| value | 0.6052 | 0.1334 | 0.0921 | -0.5824 | 0.0495 | -0.8786 |
| se   | 0.1353 | 0.0558 | 0.0669 | 0.1293 | 0.0570 | 0.0285 |   
|$t_{n-k}$ | 4.47302 | 2.39068 | 1.37668 | 4.50425 | 0.86842 | 30.82807 |
</div>

Because the t-statistics are more than 2 standard errors away from the center, we reject the null hypotheses in all cases. Thus, there is statistically evidence suggesting that the coefficients of the AR and MA models generating the data—labeled $\alpha$—are not 0.

**5. Forecasting**

After confirming that the residuals are white noise, the model is stationary and invertible, and the model coefficients are significantly different from 0, the model is acceptable for forecasting future values of the series. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/75c5469c-0aa4-48fc-b665-6d25e67604f7" width="800"/>
</p>

<div align="center">
  
| Test | CI Low         | Forecast Value | CI High        | Forecast SE    |
|:-----|:---------------|:---------------|:---------------|:---------------|
| 4.4  | 3.9885357      | 4.3393331      | 4.6901305      | 0.17897825     |
| 4.1  | 3.5827438      | 4.0845305      | 4.5863172      | 0.25601362     |
| 3.7  | 2.8913789      | 3.5396347      | 4.1878905      | 0.33074274     |
| 3.6  | 2.7654800      | 3.5691688      | 4.3728577      | 0.41004533     |
| 4.2  | 3.0167901      | 3.9750120      | 4.9332340      | 0.48888873     |
| 4.1  | 2.9065643      | 4.0181760      | 5.1297878      | 0.56714886     |
| 3.9  | 2.5231823      | 3.7867982      | 5.0504140      | 0.64470195     |
| 3.6  | 2.0609218      | 3.4743871      | 4.8878525      | 0.72115578     |
| 3.5  | 1.7703723      | 3.3311271      | 4.8918819      | 0.79630347     |
| 3.5  | 1.5895574      | 3.2947801      | 5.0000029      | 0.87001161     |
| 3.7  | 1.4740907      | 3.3207743      | 5.1674579      | 0.94218553     |
| 4.4  | 2.0286240      | 4.0136551      | 5.9986862      | 1.01277097     |
</div>

The final calculated RMSE for the ARIMA(3,1,1)(1,1,1)[12] model is: **0.56367369**.

## Model 3: Multiple Regression with ARMA Residuals

Multiple regression with ARMA (AutoRegressive Moving Average) residuals combines elements of multiple linear regression with ARMA time series modeling, specifically for when the residuals from a multiple regression model exhibit autocorrelation.

**1. ARMA Causal Model Fit**

In a causal model, one variable (the independent or predictor variable) is hypothesized to directly influence another variable (the dependent or response variable), while controlling for other potential confounding variables. Fitting a causal model involves a systematic process of evaluating causal relationships between variables.

<p align="center">
  <img src="https://github.com/user-attachments/assets/68c73298-b159-4470-a0a4-8513a67f1d51" width="800"/>
</p>

The residuals are not white noise; by fitting an ARMA model to get the coefficients to feed into a GLS model:

<p align="center">
  <img src="https://github.com/user-attachments/assets/f3967045-b4a0-4ce2-82b4-00bb4e5202af" width="450"/>
  <img src="https://github.com/user-attachments/assets/c1606a10-c37c-4a5f-b6b7-747a2ed6d12f" width="450"/>
</p>

**2. Final Model**

With unemployment rate as the dependent variable, and housing supply ratio ($x_{1t}$, time $t$) and federal funds effective rate ($x_{2t}$, time $t$) as the two independent variables:

$$y_t = 6.480278 - 0.036687 \times x_{1t} - 0.024086 \times x_{2t} + e_t$$  

where:

$$e_t = 0.94761954 \times e_{t-1} + 0.136519195 \times e_{t-2} + 0.005222809 \times e_{t-3} - 0.064356943 \times e_{t-4} + 0.048716685 \times e_{t-5} - 0.133601897 \times e_{t-6} + 0.101303293 \times e_{t-7} - 0.027352147 \times e_{t-8} - 0.149792849 \times e_{t-9} + 0.089271534 \times e_{t-10} + 0.035347094 \times e_{t-11} + 0.687864584 \times e_{t-12} - 0.643179964 \times e_{t-13} - 0.142164182 \times e_{t-14} - 0.127062788 \times e_{t-15} + 0.217739528 \times e_{t-16}$$

We have calculated the intercept standard error as: **0.5313121**

* $x_{1t}$ standard error: **0.0142313**

* $x_{2t}$ standard error: **0.0125550**

The RMSE of the GLS model on the test data is **2.328397**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0b3eec7e-9f02-4158-a7f0-7b8fc69c9159" width="700"/>
</p>


## Model 4: Vector Autoregression

Vector Autoregression (VAR) is a multivariate time series forecasting model used to analyze the dynamic relationships among multiple time series variables. Unlike traditional univariate time series models that focus on predicting a single variable, VAR models jointly model the behavior of multiple variables over time.

**1. CCF and Degree for Vector Autoregression**

The cross-correlation function (CCF) is a statistical tool used to measure the relationship between two time series variables by calculating the correlation between their lagged values. It helps to identify the extent and direction of the linear relationship between two series, including any time lags in the relationship. 

Because the data is not mean-stationary, it suggests that the mean of the time series is not constant over time, violating one of the key assumptions of time series analysis. In this case, it is important to address the non-stationarity before proceeding with any further analysis or modeling. By applying these differencing techniques, non-stationary time series data can be transformed into a stationary form:

* Regular differencing involves taking the difference between consecutive observations in the time series (effective for removing trend components from the data)
* Seasonal differencing involves taking the difference between observations separated by the seasonal period (helps remove seasonal patterns from the data)
* The seasonal difference of the regular difference combines regular differencing and seasonal differencing (data exhibits both trend and seasonality that need to be removed) 

For the unemployment training data, after applying seasonal differences and regular differences: 

$$\text{unemployment}_t = (1-B^{12}) \times (1-B) \times \text{unemployment}_t$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/f040b59f-e6f5-4e57-9515-c4229a1feb24" width="700"/>
</p>

For housing supply ratio training data, after applying seasonal differences and regular differences: 

$$\text{housing.supply.ratio}_t = (1-B^{12})\times(1-B) \times\text{housing.supply.ratio}_t$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/f743bf71-787d-4969-98a8-3e1b5ce164c0" width="700"/>
</p>

For Federal Reserve Effective Rate training data, after applying only regular differences: 

$${fed.reserve.rate}_t = (1-B) \times \text{fed.reserve.rate}_t$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/b3e40137-93c1-402a-a877-cb1ec33da9eb" width="700"/>
</p>

Combining all of the differenced data into a single object, the cross-correlation plot is:

<p align="center">
  <img src="https://github.com/user-attachments/assets/f2141403-d828-44bd-b314-0355fa3e278c" width="600"/>
  <img src="https://github.com/user-attachments/assets/071a2edf-fa69-45ca-93fe-48f932a301b2" width="600"/>
</p>

Unemployment at time $t$ depends on housing supply ratio at time ($t-6$). For how unemployment relates to itselt, the CCFs die away in a damped sine-wave fashion, so unemployment at time $t$ depends on unemployment at time $t-1$ and unemployment at time $t-2$. Writing this as a formula:

$$y_t^{\ast} = a_1 \times x_{{1,t-6}}^{\ast} + a_2 \times y_{t-1}^{\ast} + a_3 \times y_{t-2}^{\ast}, \quad \begin{cases} y_t^{\ast} = \text{differenced unemployment data} \newline x_{{1,t}}^{\ast} = \text{differenced housing supply ratio data} \end{cases}$$

Analyzing how the housing supply ratio is affected by unemployment, the first significant spike occurs at lag 1 (excluding lag 0), which decays right away. Thus, housing supply ratio at time $t$ depends on unemployment at time ($t-1$). For how the housing supply ratio relates to itself, the CCFs die away in a damped sine-wave fashion, so housing supply ratio at time $t$ depends on itself at time $t-1$ and $t-2$. Writing this in formula form:

$${x_{1,t}}^* = a_1 \times {y_{t-1}}^* + a_2 \times {x_{1,t-1}}^* + a_3 \times {x_{1,t-2}}^* , \quad \begin{cases} {x_{1,t}}^* = a_1 \times {y_{t-1}}^* + a_2 \times {x_{1,t-1}}^* + a_3 \times {x_{1,t-2}}^* \newline {y_t}^* = a_1 \times {x_{1,t-6}}^* + a_2 \times {y_{t-1}}^* + a_3 \times {y_{t-2}}^* \end{cases}$$

In the cross-correlation between unemployment and federal reserve effective rate, the first significant spike occurs at lag 1 (excluding lag 0). In addition, that significant spike decays right away, so unemployment at time $t$ depends on federal reserve effective rate at time ($t-1$). For how unemployment relates to itself, the CCFs die away in a damped sine-wave fashion, so unemployment at time $t$ depends on itself at time $t-1$ and $t-2$. Writing this in formula form:

$${y_t}^* = a_1 \times {x_{2,t-1}}^* + a_2 \times {y_{t-1}}^* + a_3 \times {y_{t-2}}^*$$

Analyzing how the federal reserve rate is affected by unemployment, the first significant spike occurs at lag 1 (excluding lag 0). In addition, that significant spike decays right away, so the reserve rate at time $t$ depends on unemployment at time $t-1$. For how the federal reserve rate is related to itself, the CCFs die away in a damped sine-wave fashion, so federal reserve rate at time $t$ depends on itself at time $t-1$ and time $t-2$. Writing this in formula form:

$${x_{2,t}}^* = a_1 \times {y_{t-1}}^* + a_2 \times {x_{2,t-1}}^* + a_3 \times {x_{2,t-2}}^*$$ 

where ${x_{2,t}}^*$ refers to differenced federal reserve rate training data. This means that VAR(2) model is optimal for between unemployment and federal reserve effective rate. Unemployment is leading since it is significant at lag 1, while housing is not significant until lag 6.

**2. Vector Autoregression Model**

Fitting the VAR(6):

<div align="center">

| Unemployment Rate      | Estimate     |                         | Estimate    | 
|:-----------------------|:-------------|:------------------------|:------------|
|Unemployment.Rate.l1    | -0.057276934 | Unemployment.Rate.l4    | 0.149502297 |
|Housing.Supply.Raio.l1  | 0.036779918  | Housing.Supply.Ratio.l4 | 0.028874474 |
|Unemployment.Rate.l2    | 0.138770027  | Unemployment.Rate.l5    | 0.143440787 |
|Housing.Supply.Ratio.l2 | 0.060761740  | Housing.Supply.Ratio.l5 | 0.035424376 |
|Unemployment.Rate.l3    | 0.222121689  | Unemployment.Rate.l6    | 0.052098039 |
|Housing.Supply.Ratio.l3 | 0.040446479  | Housing.Supply.Ratio.l6 | 0.044845204 |
|constant                | 0.001448732  |                         |             |

| Housing Supply Ratio   | Estimate     |                         | Estimate     | 
|:-----------------------|:-------------|:------------------------|:-------------|
|Unemployment.Rate.l1    | -0.628678932 | Unemployment.Rate.l4    | 0.061052072  |
|Housing.Supply.Raio.l1  | -0.247767338 | Housing.Supply.Ratio.l4 | -0.153412526 |
|Unemployment.Rate.l2    | -0.478534543 | Unemployment.Rate.l5    | -0.121499075 |
|Housing.Supply.Ratio.l2 | -0.188153630 | Housing.Supply.Ratio.l5 | -0.048113699 |
|Unemployment.Rate.l3    | 0.058333071  | Unemployment.Rate.l6    | -0.534840301 |
|Housing.Supply.Ratio.l3 | 0.045247464  | Housing.Supply.Ratio.l6 | -0.066892020 |
|constant                | -0.008287217 |                         |              |

</div>

With the information about the model coefficients, the system is:

$${x_{1,t}}^* = -0.2478 \times {x_{1,t-1}}^* - 0.1882 \times {x_{1,t-2}}^* + 0.04524 \times {x_{1,t-3}}^* - 0.1534 \times {x_{1,t-4}}^* - 0.04811 \times {x_{1,t-5}}^* - 0.0669 \times {x_{1,t-6}}^* - 0.6287 \times {y_{t-1}}^* - 0.4785 \times {y_{t-2}}^* + 0.05833 \times {y_{t-3}}^* + 0.06105 \times {y_{t-4}}^* - 0.1215 \times {y_{t-5}}^* - 0.5348 \times {y_{t-6}}^* -0.00829$$

$${y_t}^* = -0.05728 \times {y_{t-1}}^* + 0.13877 \times {y_{t-2}}^* + 0.2221 \times {y_{t-3}}^* + 0.1495 \times {y_{t-4}}^* + 0.14344 \times {y_{t-5}}^* + 0.05210 \times {y_{t-6}}^* + 0.03677 \times {x_{1,t-1}}^* + 0.06076 \times {x_{1,t-2}}^* + 0.04044 \times {x_{1,t-3}}^* + 0.02887 \times {x_{1,t-4}}^* + 0.04811 \times {x_{1,t-5}}^* + 0.04485 \times {x_{1,t-6}}^* + 0.001449$$

Fitting the VAR(2):

<div align="center">
  
| Unemployment Rate      | Estimate     |  Federal Funds Effective Rate  | Estimate     | 
|:-----------------------|:-------------|:-------------------------------|:-------------|
|Unemployment.Rate.l1    | 0.034332129  | Unemployment.Rate.l1           | -0.146505553 |
|Fed.Effective.Rate.l1   | -0.068430632 | Fed.Effective.Rate.l1          | 0.535638790  |
|Unemployment.Rate.l2    | 0.162287671  | Unemployment.Rate.l2           | 0.007378826  |
|Fed.Effective.Rate.l2   | 0.041075519  | Fed.Effective.Rate.l2          | -0.175693686 |
|constant                | -0.003645138 | constant                       | -0.020028198  |

</div>

With the information about the model coefficients, the system is:

$$x_{2,t} = 0.5356 \times x_{2,t-1} - 0.1757 \times x_{2,t-2} - 0.1465 \times y_{t-1} + 0.00738 \times y_{t-2} - -0.02003$$

$$y_t = 0.0343 \times y_{t-1} + 0.1623 \times y_{t-2} - 0.0684 \times x_{2,t-1} + 0.0411 \times x_{2,t-2} - 0.003645$$

**3. Impulse Response Functions**

The impulse response reflects the effect and length of the effect of a shock to the system; it allows for the decomposition of the response of endogenous variables to a shock into orthogonal components

Orthogonal Impulse Response for Unemployment Rate (how Unemployment rate and Housing Supply Ratio respond to an impulse in Unemployment rate). Both have a large response, and the equilibrium is achieved after t = 4. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/cabd3a5c-5f88-4c5a-9f1a-4b6be3ae3941" width="600"/>
</p>

Orthogonal Impulse Response for Housing Supply Ratio (how the Unemployment Rate and Housing Supply Ratio respond to an impulse in Housing Supply Ratio). Unemployment does not respond much to the shock, while the Housing Supply Ratio responds dramatically to the impulse. Therefore, Equilibrium is achieved after t = 4.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6daa1173-2b89-4467-9967-5495780ce81b" width="600"/>
</p>

Orthogonal Impulse Response from Unemployment Rate (how Unemployment Rate and Federal Reserve Effective Rate respond to an impulse in Unemployment rate). The Unemployment Rate responds drastically to the impulse, but the Federal Reserve Effective Rate responds more mildly. In conclusion, both of them reach equilibrium at t = 10.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fa78e7c1-364f-4f98-8c14-1ac9cc4dbfdf" width="600"/>
</p>

Orthogonal Impulse Response for Federal Funds Rate (how the Unemployment rate and Federal Reserve Effective Rate respond to an impulse in Federal Reserve Effective Rate). The Unemployment Rate does not respond much to the impulse, but the Federal Reserve Effective Rate responds quite dramatically. Both of them reach equilibrium at t = 10.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f1c0cee0-0229-497f-9c43-742c579e7bfe" width="600"/>
</p>









