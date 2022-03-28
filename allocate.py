import numpy as np
import pandas as pd
import scipy

#########################################################################
## Change this code to take in all asset price data and predictions    ##
## for one day and allocate your portfolio accordingly.                ##
#########################################################################

def allocate_portfolio(asset_prices, asset_price_predictions_1\
                       asset_price_predictions_2,\
                       asset_price_predictions_3):
    
    # This simple strategy equally weights all assets every period
    # (called a 1/n strategy).
    
    n_assets = len(asset_prices)
    weights = np.repeat(1 / n_assets, n_assets)
    return weights
