#  AutoCTS: Automated Correlated Time Series Forecasting

This is the original pytorch implementation of AutoCTS in the following paper: AutoCTS: Automated Correlated Time Series Forecasting.

This code is based on the implementation of [PC-Darts](https://github.com/yuhuixu1993/PC-DARTS).

## Requirements
- python 3
- see `requirements.txt`
## Data Preparation
AutoCTS is implemented on several public correlated time series forecasting datasets.

- **METR-LA** and **PEMS-BAY** from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN)

- **PEMS03**, **PEMS04**, **PEMS07** and **PEMS08** from [STSGCN (AAAI-20)](https://github.com/Davidham3/STSGCN).
Download the data [STSGCN_data.tar.gz](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw) with password: `p72z` and uncompress data file using`tar -zxvf data.tar.gz`
To run AutoCTS on the PEMS03 dataset, you only need to download the [PEMS03_data.csv](https://drive.google.com/file/d/1rJwVmQTAoOmmR5l6iQGRwT__tHjQeGn6/view?usp=sharing) and put it into the data/pems/PEMS03 folder.

- **Solar-Energy** and **Electricity** datasets from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Uncompress them and move them to the data folder.


## Architecture Search

```
CUDA_VISIBLE_DEVICES=0 python3 train_search.py
```

# Reference

If you use AutoCTS for your research, please cite the following paper. 
<pre>     
@article{wu2022autocts,
  title={AutoCTS: Automated Correlated Time Series Forecasting},
  author={Xinle Wu and Dalin Zhang and Chenjuan Guo and Chaoyang He and Bin Yang and Christian S. Jensen},
  year={2022},
  pages={971--983}
  journal={Proceedings of the VLDB Endowment},
  volume={4}
}
</pre>   
