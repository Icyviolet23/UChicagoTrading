import numpy as np
import pandas as pd
import scipy

#########################################################################
## Change this code to take in all asset price data and predictions    ##
## for one day and allocate your portfolio accordingly.                ##
#########################################################################

# CONSTANTS

SHARES_OUTSTANDING = np.array([425000000, 
                               246970000,
                               576250000,
                               4230000000,
                               1930000000,
                               3370000000,
                               16320000000,
                               7510000000,
                               508840000])

# Temporary values
HISTORICAL_COVARIANCE_MATRIX = np.array([[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                                         [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                                         [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                                         [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                                         [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                                         [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                                         [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                                         [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                                         [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]])

def allocate_portfolio(asset_prices, asset_price_predictions_1\
                       asset_price_predictions_2,\
                       asset_price_predictions_3):
    
    # This simple strategy equally weights all assets every period
    # (called a 1/n strategy).
    
    n_assets = len(asset_prices)
    weights = np.repeat(1 / n_assets, n_assets)
    return weights

# Inputs: Numpy Arrays
def getMarketCapWeightsVector(sharesOutstanding, assetPrices):
    totalMarketCap = np.dot(sharesOutstanding, assetPrices)
    marketCapsVector = np.multiply(sharesOutstanding, assetPrices)
    marketCapWeightsVector = np.true_divide(marketCapsVector, totalMarketCap)
    return marketCapWeightsVector
