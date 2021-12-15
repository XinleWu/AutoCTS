#  AutoCTS: Automated Correlated Time Series Forecasting

This is the original pytorch implementation of AutoCTS in the following paper: AutoCTS: Automated Correlated Time Series Forecasting.

## Requirements
- python 3
- see `requirements.txt`
## Data Preparation
AutoCTS is implemented on several public correlated time series forecasting datasets.

- **METR-LA** and **PEMS-BAY** from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN)

- **PEMS03**, **PEMS04**, **PEMS07** and **PEMS08** from [STSGCN (AAAI-20)](https://github.com/Davidham3/STSGCN).
Download the data [STSGCN_data.tar.gz](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw) with password: `p72z` and uncompress data file using`tar -zxvf data.tar.gz`


- **Solar-Energy** and **Electricity** datasets from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Uncompress them and move them to the data folder.


## Architecture Search

```
CUDA_VISIBLE_DEVICES=0 python3 train_search.py
```
