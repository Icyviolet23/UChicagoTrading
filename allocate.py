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

# To populate with given historical data (in CSV files)
# To update on each call of allocate_portfolio with new data
# Assume these are populated exactly the same as if using pd.read_csv()
DF_PRICES = pd.DataFrame()
DF_PREDICTED_PRICES_1 = pd.DataFrame()
DF_PREDICTED_PRICES_2 = pd.DataFrame()
DF_PREDICTED_PRICES_3 = pd.DataFrame()

# ##############################
# ALLOCATE_PORTFOLIO 
# ##############################

def allocate_portfolio(asset_prices, 
                       asset_price_predictions_1,
                       asset_price_predictions_2,
                       asset_price_predictions_3):
    
    # Test
    DF_PRICES = pd.read_csv("Acutal Testing Data.csv")
    DF_PREDICTED_PRICES_1 = pd.read_csv("Predicted Testing Data Analyst 1.csv")
    DF_PREDICTED_PRICES_2 = pd.read_csv("Predicted Testing Data Analyst 2.csv")
    DF_PREDICTED_PRICES_3 = pd.read_csv("Predicted Testing Data Analyst 3.csv")

    # Clean up DFs by removing first column if necessary
    # (Should only be run the first time allocate_portfolio is called)
    for DF in [DF_PRICES, 
               DF_PREDICTED_PRICES_1, 
               DF_PREDICTED_PRICES_2, 
               DF_PREDICTED_PRICES_3]:
        if len(DF.columns) == 10:
            DF.drop(DF.columns[0], axis=1, inplace = True)
            print(DF)
    
    # Update DFs with new data
    for (DF, data) in [(DF_PRICES, asset_prices),
                       (DF_PREDICTED_PRICES_1, asset_price_predictions_1),
                       (DF_PREDICTED_PRICES_2, asset_price_predictions_2), 
                       (DF_PREDICTED_PRICES_3, asset_price_predictions_3)]:
        DF.loc[len(DF)] = data
        print(DF)


    # bigSigma = getCovarianceMatrix()
    # # Calculate 3 expected returns E(R) matrices with 3 given analyst predictions
    # ER1 = getExpectedReturns(asset_prices, asset_price_predictions_1)
    # ER2 = getExpectedReturns(asset_prices, asset_price_predictions_2)
    # ER3 = getExpectedReturns(asset_prices, asset_price_predictions_3)
    # TO DO: Get a weighted average of the 3 E(R) values based on accuracies of analysts
    # TO DO: Iterate over some weight matrices and see which has the best Sharpe ratio

def test():
    allocate_portfolio([1,2,3,4,5,6,7,8,9], 
                       [123,2,3,4,5,6,7,8,9],
                       [456,2,3,4,5,6,7,8,9],
                       [789,2,3,4,5,6,7,8,9])

# ##############################
# BIG HELPER FUNCTIONS
# ##############################

# def updateAllDataFrames(currPrices,
#                         predPrices1,
#                         predPrices2,
#                         predPrices3):
#     DF_PRICES.loc[len(DF_PRICES)] = currPrices
#     DF_PREDICTED_PRICES_1[len(DF_PREDICTED_PRICES_1)] = predPrices1

def getCovarianceMatrix():
    pass

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