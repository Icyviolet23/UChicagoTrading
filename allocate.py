import numpy as np
import pandas as pd
import scipy

#########################################################################
## Change this code to take in all asset price data and predictions    ##
## for one day and allocate your portfolio accordingly.                ##
#########################################################################

# ##############################
# CONSTANTS
# ##############################

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

# ##############################
# ALLOCATE_PORTFOLIO 
# ##############################

def allocate_portfolio(asset_prices, 
                       asset_price_predictions_1,
                       asset_price_predictions_2,
                       asset_price_predictions_3):
    # Calculate 3 expected returns E(R) matrices with 3 given analyst predictions
    ER1 = getExpectedReturns(asset_prices, asset_price_predictions_1)
    ER2 = getExpectedReturns(asset_prices, asset_price_predictions_2)
    ER3 = getExpectedReturns(asset_prices, asset_price_predictions_3)
    # TO DO: Get a weighted average of the 3 E(R) values based on accuracies of analysts
    # TO DO: Iterate over some weight matrices and see which has the best Sharpe ratio

# ##############################
# BIG HELPER FUNCTIONS
# ##############################

def getExpectedReturns(currPrices, predPrices):
    Q = getReturns(currPrices, predPrices)


# ##############################
# SMALL HELPER FUNCTIONS
# ##############################

# Inputs: Numpy Arrays
# - Currently using logarithmic returns, to change later
def getReturns(prevPrices, nextPrices):
    return np.log(np.true_divide(nextPrices, prevPrices))

# Inputs: Numpy Arrays
def getMarketCapWeightsVector(sharesOutstanding, assetPrices):
    totalMarketCap = np.dot(sharesOutstanding, assetPrices)
    marketCapsVector = np.multiply(sharesOutstanding, assetPrices)
    marketCapWeightsVector = np.true_divide(marketCapsVector, totalMarketCap)
    return marketCapWeightsVector