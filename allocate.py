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

def allocate_portfolio(asset_prices, 
                       asset_price_predictions_1,
                       asset_price_predictions_2,
                       asset_price_predictions_3):

    Q = getPredictedReturnsMatrix(asset_prices, 
                                  asset_price_predictions_1,
                                  asset_price_predictions_2,
                                  asset_price_predictions_3)


# Inputs: Numpy Arrays
def getPredictedReturnsMatrix(currPrices, 
                              predPrices1, 
                              predPrices2, 
                              predPrices3):
    predReturns1 = getReturns(currPrices, predPrices1)
    predReturns2 = getReturns(currPrices, predPrices2)
    predReturns3 = getReturns(currPrices, predPrices3)
    predReturnsMatrix = np.array([predReturns1,
                                  predReturns2,
                                  predReturns3])
    return predReturnsMatrix

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