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

MARKET_CAP_WEIGHTS = np.array([0.00732755, 
                               0.00736718, 
                               0.01086331, 
                               0.0388959, 
                               0.03170921,
                               0.01237441, 
                               0.37441448, 
                               0.3110686 , 
                               0.20597937])

# Risk aversion coefficient
LAMBDA = 2

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

    # Calculate covariance matrix (bigSigma)
    returns = np.log(DF_PRICES).diff() # DataFrame
    bigSigma = returns.cov().to_numpy()
    print(bigSigma)

    # Calculate implied excess equilibrium returns (bigPi)
    bigPi = LAMBDA * (bigSigma @ MARKET_CAP_WEIGHTS)
    print(bigPi)

    # Calculate diagonal matrix representing uncertainty of view (bigOmega)
    # (Since we're using only absolute views, this should be same for all views)
    diagonalValues = [bigSigma[x, x] * LAMBDA for x in range(9)]
    bigOmega = np.diag(diagonalValues)
    print(bigOmega)
    
    # Calculate 3 expected returns E(R) matrices with 3 given analyst predictions
    ER1 = getExpectedReturns(asset_prices,
                             asset_price_predictions_1, 
                             bigSigma, 
                             bigPi, 
                             bigOmega)
    ER2 = getExpectedReturns(asset_prices,
                             asset_price_predictions_2, 
                             bigSigma, 
                             bigPi, 
                             bigOmega)
    ER3 = getExpectedReturns(asset_prices,
                             asset_price_predictions_3, 
                             bigSigma, 
                             bigPi, 
                             bigOmega)
    print("\nExpected Returns:")
    print(ER1)
    print(ER2)
    print(ER3)

    # TO DO: Get a weighted average of the 3 E(R) values based on accuracies of analysts
    # TO DO: Iterate over some weight matrices and see which has the best Sharpe ratio

# ##############################
# HELPER FUNCTIONS
# ##############################

# Inputs: (Various)
def getExpectedReturns(currPrices,  # List
                       predPrices,  # List
                       bigSigma,    # NumPy Array
                       bigPi,       # NumPy Array
                       bigOmega):   # NumPy Array
    Q = getReturns(np.array(currPrices), np.array(predPrices))
    return (np.linalg.inv(np.linalg.inv(LAMBDA * bigSigma) + np.linalg.inv(bigOmega)) 
           @ (np.linalg.inv(LAMBDA * bigSigma) @ bigPi + np.linalg.inv(bigOmega) @ Q))

# Inputs: Numpy Arrays
def getReturns(prevPrices, nextPrices):
    return np.log(np.true_divide(nextPrices, prevPrices))

# ##############################
# TEST
# ##############################

def test():
    global DF_PRICES
    global DF_PREDICTED_PRICES_1
    global DF_PREDICTED_PRICES_2
    global DF_PREDICTED_PRICES_3
    DF_PRICES = pd.read_csv("Acutal Testing Data.csv")
    DF_PREDICTED_PRICES_1 = pd.read_csv("Predicted Testing Data Analyst 1.csv")
    DF_PREDICTED_PRICES_2 = pd.read_csv("Predicted Testing Data Analyst 2.csv")
    DF_PREDICTED_PRICES_3 = pd.read_csv("Predicted Testing Data Analyst 3.csv")
    allocate_portfolio([1,2,3,4,5,6,7,8,9], 
                       [123,2,3,4,5,6,7,8,9],
                       [456,2,3,4,5,6,7,8,9],
                       [789,2,3,4,5,6,7,8,9])