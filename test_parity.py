import unittest
import pandas as pd
import numpy as np
from parity_monitor import calculate_parity

class TestParityLogic(unittest.TestCase):
    
    def setUp(self):
        """
        This runs before every test. We create 'Fake Data' here.
        Imagine a perfect world where Parity is exact.
        """
        # 1. Define the variables
        self.S = 400.00       # Spot Price (Stock is $400)
        self.K = 400.00       # Strike Price
        self.r = 0.05         # 5% Interest Rate
        self.T = 1.0          # 1 Year to expiration
        
        # 2. Calculate what the Call Price SHOULD be for perfect parity
        # Formula: C = P + S - K*e^(-rT)
        # Let's say Put = $10.00
        self.put_price = 10.00
        
        # Calculate the theoretical Call price manually:
        # PV of Strike = 400 * e^(-0.05 * 1) = 400 * 0.9512 = 380.49
        pv_strike = self.K * np.exp(-self.r * self.T)
        
        # So Call = 10 + 400 - 380.49 = 29.51
        self.theoretical_call = self.put_price + self.S - pv_strike
        
        # 3. Create the Fake Dataframe (like yfinance would give us)
        data = {
            'strike': [self.K],
            'call_price': [self.theoretical_call],
            'put_price': [self.put_price]
        }
        self.fake_chain = pd.DataFrame(data)

    def test_parity_calculation_zero_deviation(self):
        """
        Test that our function calculates 0 deviation when prices are perfect.
        """
        # Run the function we are testing
        result_df = calculate_parity(self.fake_chain, self.S, self.T, self.r)
        
        # Get the calculated deviation
        calculated_deviation = result_df['deviation'].iloc[0]
        
        # Check if it is effectively zero (using almost_equal for floating point math)
        print(f"\nExpected Deviation: 0.0")
        print(f"Actual Deviation:   {calculated_deviation:.10f}")
        
        self.assertAlmostEqual(calculated_deviation, 0.0, places=5)

if __name__ == '__main__':
    unittest.main()