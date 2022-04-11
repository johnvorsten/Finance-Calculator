# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:39:44 2019

@author: z003vrzk
"""

# Python imports
import datetime, unittest, re

# Third party imports
import numpy as np

# Local imports
from finance_graphing import (create_transaction_objects, Income,
                              plot_integral, calculate_gradients)



#%%

class Testing(unittest.TestCase):

    def test_gradients_values(self):

        # 10 dollars a day
        inc1 = Income(10, 1/365, datetime.date(2020,1,1), datetime.date(2020,1,5))
        # Single 10 dollar expense
        inc2 = Income(-20, 1/365, datetime.date(2020,1,2), datetime.date(2020,1,2), one_time=True)

        # Calculate gradients over days
        start_date = datetime.date(2020,1,1)
        end_date = datetime.date(2020,1,5)
        incomes = [inc1, inc2]
        gradients, values = calculate_gradients(incomes, start_date, end_date)

        """Gradients should look like this
        array([[[ 10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.,  10., 10.]],
           [[-10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10.]],
           [[ 10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.,  10., 10.]],
           [[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 0.]]])"""
        test_gradient =np.array([[[10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]],
                     [[-10., -10.,-10.,-10.,-10.,-10.,-10.,-10.,-10.,-10.,-10.]],
                     [[10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]],
                     [[ 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]])

        test_values = np.array([[[10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]],
                               [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]],
                               [[10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]],
                               [[10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]]])

        self.assertTrue(np.array_equal(test_gradient, test_gradient))
        self.assertTrue(np.array_equal(test_values, test_values))

        return None

    def test_dates(self):

        # 10 dollars a day
        inc1 = Income(10, 1/365,
                      datetime.date(2020,1,1), datetime.date(2020,1,5))
        # Single 10 dollar expense
        inc2 = Income(-20, 1/365,
                      datetime.date(2020,1,2), datetime.date(2020,1,2), one_time=True)

        # A range of dates should not be an issue
        self.assertTrue(inc1.start_date == datetime.date(2020,1,1))
        self.assertTrue(inc1.end_date == datetime.date(2020,1,5))

        # One-time expenses should range multiple days even if it occurs on one day
        # This is because a gradient needs to be calculated between days
        self.assertTrue(inc2.start_date == datetime.date(2020,1,2))
        self.assertTrue(inc2.end_date == datetime.date(2020,1,3))

        return None


    def test_income_gradient(self):

        # 10 dollars a day
        inc1 = Income(10, 1/365,
                      datetime.date(2020,1,1), datetime.date(2020,1,5))
        # Single 10 dollar expense
        inc2 = Income(-20, 1/365,
                      datetime.date(2020,1,2), datetime.date(2020,1,2), one_time=True)

        # Test gradient
        gradient1 = inc1.calc_derivative()
        gradient2 = inc2.calc_derivative()

        self.assertTrue(np.array_equal(gradient1, np.array([float(10)] * 11)))
        self.assertTrue(np.array_equal(gradient2, np.array([float(-20)] * 11)))

        return None


    def test_calc_gradients(self):

        # 10 dollars a day
        inc1 = Income(10, 1/365, datetime.date(2020,1,1), datetime.date(2020,1,5))
        # Single 10 dollar expense
        inc2 = Income(-20, 1/365,
                      datetime.date(2020,1,2), datetime.date(2020,1,2), one_time=True)

        # Test gradient
        start_date = datetime.date(2020,1,1)
        end_date = datetime.date(2020,1,5)
        gradients1, values1 = calculate_gradients([inc1,inc2], start_date, end_date)
        gradients_test1=np.array([[[10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]],
                    [[-10.,-10.,-10.,-10.,-10.,-10.,-10.,-10.,-10.,-10.,-10.]],
                    [[10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]],
                    [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]])
        values_test1=np.array([[[10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]],
                               [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]],
                               [[10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]],
                               [[10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]]])

        # Test gradient
        start_date = datetime.date(2019,12,31)
        end_date = datetime.date(2020,1,3)
        gradients2, values2 = calculate_gradients([inc1,inc2], start_date, end_date)
        gradients_test2=np.array([[
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]],
            [[10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]],
            [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]])
        values_test2=np.array([[
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]],
            [[10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]],
            [[10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]]])

        self.assertTrue(np.array_equal(gradients1, gradients_test1))
        self.assertTrue(np.array_equal(values1, values_test1))
        self.assertTrue(np.array_equal(gradients2, gradients_test2))
        self.assertTrue(np.array_equal(values2, values_test2))

        return None


    def test_income_probability(self):

        # 10 dollars a day
        inc1 = Income(10, 1/365, datetime.date(2020,1,1), datetime.date(2020,1,5))

        # Confirm the default probability
        a = np.array([1. , 0.9, 0.8, 0.7, 0.6, 0.5,0.4,0.3, 0.2, 0.1,0.])
        self.assertTrue(np.array_equal(inc1.probability, a))

        # Confirm custom probability
        probability=[1,0.98,0.9,0.85,0.8,0.75,0.70,0.68,0.65,0.5,0.45]
        inc2 = Income(10, 1/365, datetime.date(2020,1,1), datetime.date(2020,1,5),
                      probability=probability)
        self.assertTrue(np.array_equal(inc2.probability, probability))

        return None

    def test_best_worst_case(self):

        # Income of $100 to $50 over 10 days
        period = 10/365
        start_date=datetime.date(2020,1,1)
        end_date=datetime.date(2020,1,2)
        best_case = 100
        worst_case = 50
        inc1 = Income(income=100, period=period,
                      start_date=start_date, end_date=end_date,
                      best_case=best_case, worst_case=worst_case)

        daily_gradient = inc1.calc_derivative()
        value = (best_case - worst_case) * inc1.probability + worst_case
        daily_gradient_test = value / (period * 365)

        self.assertTrue(np.array_equal(daily_gradient,daily_gradient_test))

        return None


    def test_regex_match(self):
        """# TODO
        This regex is weak. Lots of the test cases will output odd numbers.
        I would have to test for lots of conditionals like

        If more that (2) '.' appear in the string sequence
            A number shouldn't be represnted like 2.334.05 (other countries maybe?)
        If numbers are interrupted by characters other than ',' or '.'
            This is not a valid number : 24A33.05
        Even or odd numbers
            This is negative : (5.45)
            This is negative : -5.45
            This is something : -(5.45)
            This is positive : 5.45
            This is positive : +5.45
        Is the data formatted where 5.45 is a transaction out of account or into
        account?
            This is transaction out of account : -5.45
            Or is this? : 5.45

        """

        reg = re.compile('[^0-9.-]')
        s1 = 'name ni4nvi $03,200.40'
        s2 = '(-$32,43.22)'
        s3 = '$4,300.00'
        s4 = '-$4,300.00'
        s5 = '+3i2nndfs!@#$%^&*()2.jnfinn%55'
        s6 = '#4,304'
        s7 = '#4.334455.0040'
        s8 = ''
        s9 = ''
        s10 = ''

        test_case = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10]

        for test in test_case:
            result = re.sub(reg, '', test)
            print(result)

        return None

if __name__ == '__main__':
    unittest.main()


"""
~~ Order Filtering ~~
Domain acts as a mask over each pixel of x
For every non-zero element in the mask of x, a list is constructed
The list is sorted, and the 'rank' element from the list is selected

from scipy import signal
x = np.arange(25).reshape(5, 5)
domain = np.identity(3)
signal.order_filter(x, domain, 0)
signal.order_filter(x, domain, 1)
signal.order_filter(x, domain, 2)

Example 1
rank = 1
domain = [[1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.]]
x = [[ 0,  1,  2,  3,  4],
     [ 5,  6,  7,  8,  9],
     [10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19],
     [20, 21, 22, 23, 24]]

For element [0,0] the domain mask is centered, and the non-masked elements
are [0,0,6] (elements are zero when the mask is off of x).
Since rank is 1, the 1st element from the list is selected (0).

For element [1,2] the domain mask is centered, and the non-masked elements
are [1,7,13].
Since rank is 1, the 1st element from the list is selected (7).

~~ Median filter ~~
Same idea as above, but the median of the list is chosen
from scipy import signal
x = np.arange(25).reshape(5, 5)
kernel_size = 3
signal.medfilt(x, kernel_size=kernel_size)

Element [1,0]
Elements in list are [0, 0, 0, 0, 1, 5, 6, 10, 11]
The median of the list is 1

"""


