import numpy as np
from allocate import *

# Test getPredictedReturnsMatrix

currPrices = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10])
predPrices1 = np.array([11, 12, 13, 10, 10, 10, 10, 10, 20])
predPrices2 = np.array([12, 13, 14, 10, 10, 10, 10, 10, 20])
predPrices3 = np.array([13, 14, 15, 10, 10, 10, 10, 10, 20])

Q = getPredictedReturnsMatrix(currPrices, predPrices1, predPrices2, predPrices3)

print(Q)
