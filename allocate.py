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
            #print(DF)
    
    # Update DFs with new data
    for (DF, data) in [(DF_PRICES, asset_prices),
                       (DF_PREDICTED_PRICES_1, asset_price_predictions_1),
                       (DF_PREDICTED_PRICES_2, asset_price_predictions_2), 
                       (DF_PREDICTED_PRICES_3, asset_price_predictions_3)]:
        DF.loc[len(DF)] = data
        #print(DF)

    # Calculate covariance matrix (bigSigma)
    returns = np.log(DF_PRICES).diff() # DataFrame
    bigSigma = returns.cov().to_numpy()
    #print(bigSigma)

    # Calculate implied excess equilibrium returns (bigPi)
    bigPi = LAMBDA * (bigSigma @ MARKET_CAP_WEIGHTS)
    #print(bigPi)

    # Calculate diagonal matrix representing uncertainty of view (bigOmega)
    # (Since we're using only absolute views, this should be same for all views)
    diagonalValues = [bigSigma[x, x] * LAMBDA for x in range(9)]
    bigOmega = np.diag(diagonalValues)
   #print(bigOmega)
    
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

    #averageReturn = np.mean ([ER1, ER2, ER3], axis = 0)
    corrA1 = DF_PRICES.corrwith(DF_PREDICTED_PRICES_1, axis = 0)
    corrA1 = corrA1.to_numpy()
    corrA2 = DF_PRICES.corrwith(DF_PREDICTED_PRICES_2, axis = 0)
    corrA2 = corrA2.to_numpy()
    corrA3 = DF_PRICES.corrwith(DF_PREDICTED_PRICES_3, axis = 0)
    corrA3 = corrA3.to_numpy()

    (confidenceA1, confidenceA2, confidenceA3) = generateAnalystConfidenceInterval (corrA1, corrA2, corrA3)
    weightedE1 = np.multiply(ER1, confidenceA1)
    weightedE2 = np.multiply(ER2, confidenceA2)
    weightedE3 = np.multiply(ER3, confidenceA3)
    finalER = np.add (np.add (weightedE1, weightedE2), weightedE3)



    finalweights = generateWeights (finalER, bigSigma)


    
    #print("\nExpected Returns:")

    #print(finalER)
    #print(ER2)
    #print(ER3)
    #print(bigSigma)
    #print(averageReturn)
    #print((finalweights))
    #print(sum(finalweights))
    #(confidenceA1, confidenceA2, confidenceA3)print(finalweights)

    # TO DO: Get a weighted average of the 3 E(R) values based on accuracies of analysts
    # TO DO: Iterate over some weight matrices and see which has the best Sharpe ratio
    return finalweights
# ##############################
# HELPER FUNCTIONS
# ##############################

#generate weights on how much we should trust each analyst given past predictions
def generateAnalystConfidenceInterval (corrA1, corrA2, corrA3):
    sum1 = np.add(corrA1,corrA2)
    totalsum = np.add(sum1, corrA3)
    A1weight = np.divide(corrA1, totalsum)
    A2weight = np.divide(corrA2, totalsum)
    A3weight = np.divide(corrA3, totalsum)
    return (A1weight, A2weight, A3weight)




def generatePortfolioStdDev (weights, covMatrix):
    variance = np.transpose(weights) @ covMatrix @ weights
    stdDev = np.sqrt(variance)
    return stdDev

#returns a 9x1 vector of weights based on the analyst predicted return
#predictedReturn must be in the form of a np array (representing a 9x1 vector)
def generateWeights (predictedReturn, covMatrix):
    bestWeight = None
    skip = 1 #we make 5% adjustment each time
    bestSharpeRatio = -1000000
    iterations = 10000
    for i in range(iterations):
        weightsArray = np.random.dirichlet(np.ones(9),size=1)
        weights = np.transpose(np.asmatrix(weightsArray))
        stdDev = generatePortfolioStdDev (weights, covMatrix)
        EReturn = np.dot (predictedReturn, weights)
        sharpeRatio = EReturn/stdDev
        if sharpeRatio > bestSharpeRatio:
            bestWeight = weights
            bestSharpeRatio = sharpeRatio

    """
    for w1 in range(0, 100, skip):
        for w2 in range(0, 100 - w1, skip):
            for w3 in range(0, 100 - w1 - w2, skip):
                for w4 in range(0, 100 - w1 - w2 - w3, skip):
                    for w5 in range(0, 100 - w1 - w2 - w3 - w4, skip):
                        for w6 in range(0, 100 - w1 - w2 - w3 - w4 - w5, skip):
                            for w7 in range(0, 100 - w1 - w2 - w3 - w4 -w5 -w6, skip):
                                for w8 in range(0, 100 - w1 - w2 - w3 - w4 -w5 -w6 -w7, skip):
                                    for w9 in range(0, 100 - w1 - w2 - w3 - w4 -w5 -w6 -w7-w8, skip):
                                        weightsArray = [w1, w2, w3, w4, w5, w6, w7, w8, w9]
                                        print(weightsArray)
                                        if (sum(weightsArray) != 100): continue  
                                        else:
                                            weightsArray = [w1/100, w2/100, w3/100, w4/100, w5/100, w6/100, w7/100, w8/100, w9/100]
                                            weights = np.transpose(np.asmatrix(weightsArray))   
                                            stdDev = generatePortfolioStdDev (weights, covMatrix)
                                            EReturn = np.dot (predictedReturn, weights)
                                            sharpeRatio = EReturn/stdDev
                                            if sharpeRatio > bestSharpeRatio:
                                                bestWeight = weights
                                                bestSharpeRatio = sharpeRatio
    """
    return bestWeight
                







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
    allocate_portfolio([141.9648974,226.3529486,146.4843367,77.53980757,134.9442617,32.98279653,167.4403036,297.0918321,3091.00349], 
                       [151.3342795,	248.634224,	91.98669049,	129.5934306	,75.57864797,	31.2687251,	217.6962327,	244.2895793,	3205.820805],
                       [98.18867628,	257.8457167,	135.9126622,	83.50970224,	123.0132427,	43.28584419,	144.913266,	351.1300353,	3888.316207],
                       [128.0097445,	321.4693291,	205.9085143,	95.20471768,	52.06143928,	39.07667906,	84.49135392,	328.8131346,	2759.588444])

    for i in range (100):
        weight = allocate_portfolio (DF_PRICES.iloc[i], DF_PREDICTED_PRICES_1.iloc[i], DF_PREDICTED_PRICES_2.iloc[i], DF_PREDICTED_PRICES_3.iloc[i])
        print(weight)


test()



