# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:41:51 2020

@author: z003vrzk
"""
# Python imports
import datetime

# Third party imports
import pandas as pd
import numpy as np

# Local imports
from finance_graphing import (create_transaction_objects, Income,
                              plot_integral, calculate_gradients)

#%% Graphing

"""Creating income instances
Income instances can either be incomes or expenses. Enter a negative number
for expenses or a positive number for income
"""

# Basic income - $60,000 to $40,000 per year
INC1 = Income(60000*0.7, 365/365, datetime.date(2019,6,5), datetime.date(2023,6,5),
                best_case=60000*0.7, worst_case=40000*0.7) #Income

# Single expense of $4,000, best $3,000 worst $5,000
INC2 = Income(-4000, 1/365,datetime.date(2019,10,1),datetime.date(2020,10,1),
                best_case=-3000, worst_case=-5000, one_time=True)

# Single expense of $7,500
INC3 = Income(-7500,1/365,datetime.date(2021,10,20),datetime.date(2021,10,20),
                one_time=True)

# Monthly expense of $2,000
INC4 = Income(-2000,31/365, datetime.date(2019,6,5), datetime.date(2025,6,5),
                best_case=-1800, worst_case=-2100, one_time=True)

# Other income bi-weekly
INC5 = Income(450, 14/365, datetime.date(2019, 6, 5), datetime.date(2025, 6, 5),
                best_case=500, worst_case=440)

# Single expense of $12,500 yearly over two years ($25,000 total expense)
# Range from $10,000 to $15,000 expense
INC6 = Income(-12500, 365/365, datetime.date(2023,6,5), datetime.date(2025,6,5),
                best_case=-10000,worst_case=-15000)

# Import bank transactions .csv for projection of daily expenses
path = r'../writeup/Transaction_Data.csv'
transaction_dataframe = pd.read_csv(path)
incomes_auto = create_transaction_objects(transaction_dataframe,
                                          value_col='Amount',
                                          date_col='Date')

# Create an iterable of income objects
incomes = [INC1, INC2,INC3,INC4,INC5,INC6]
incomes.extend(incomes_auto)
# Define your plotting period
start_date = min([x.start_date for x in incomes])
end_date = datetime.date(2020,2,1)

# Calculate gradients and values
gradients, values = calculate_gradients(incomes, start_date, end_date)

#Plot the income objects
plot_integral(values, start_date, end_date, smooth=True, smooth_type='LSQ-bspline')
# plot_integral(values, start_date, end_date, smooth=True, smooth_type='bv-bspline')
# plot_integral(values, start_date, end_date, smooth=True, smooth_type='Smooth-bspline')
# plot_integral(values, start_date, end_date, smooth=True, smooth_type='wiener')


#%% Test 2

"""Creating income instances
Income instances can either be incomes or expenses. Enter a negative number
for expenses or a positive number for income
"""

# Simple
INC1 = Income(100, 1/365, datetime.date(2020,1,2), datetime.date(2020,1,3),
                one_time=True)

# Simple
INC2 = Income(-100, 1/365, datetime.date(2020,1,3), datetime.date(2020,1,4),
                one_time=True)

# Simple
INC3 = Income(100, 1/365, datetime.date(2020,1,4), datetime.date(2020,1,5),
                one_time=True)

# Simple
INC4 = Income(100, 1/365, datetime.date(2020,1,5), datetime.date(2020,1,6),
                one_time=True)

# Simple
INC5 = Income(-100, 1/365, datetime.date(2020,1,6), datetime.date(2020,1,7),
                one_time=True)

# Create an iterable of income objects
incomes = [INC1,INC2,INC3,INC4,INC5]
# Define your plotting period
start_date = min([x.start_date for x in incomes])
end_date = max([x.end_date for x in incomes]) + datetime.timedelta(days=1)

# Calculate gradients and values
gradients, values = calculate_gradients(incomes, start_date, end_date)

#Plot the income objects
# plot_integral(values, start_date, end_date, smooth=True, smooth_type='LSQ-bspline')
# plot_integral(values, start_date, end_date, smooth=True, smooth_type='bv-bspline')
plot_integral(values, start_date, end_date, smooth=True, smooth_type='Smooth-bspline')
# plot_integral(values, start_date, end_date, smooth=True, smooth_type='wiener')
# plot_integral(values, start_date, end_date, smooth=False)

#%% Test 3

"""Creating income instances
Income instances can either be incomes or expenses. Enter a negative number
for expenses or a positive number for income
"""

# Basic income
INC1 = Income(40000*0.7, 365/365, datetime.date(2019,6,5), datetime.date(2023,6,5),
                best_case=40000*0.7, worst_case=20000*0.7) #Income

# Single expense of $4,000, best $3,000 worst $5,000
INC2 = Income(-4000, 1/365,datetime.date(2019,10,1),datetime.date(2020,10,1),
                best_case=-3000, worst_case=-5000, one_time=True)

# Monthly expense of $2,000
INC3 = Income(-2000,31/365, datetime.date(2019,6,5), datetime.date(2025,6,5),
                best_case=-1800, worst_case=-2100, one_time=True)


# Generate some random transaction values
_transactions1 = np.random.normal(loc=-30, scale=50, size=(100))
_transactions3 = np.random.normal(loc=-1000, scale=300, size=(25))
_transactions = np.concatenate((_transactions1, _transactions3))
incomes_auto = []
_days = np.linspace(1,28,28, dtype=np.int16)
_months = np.linspace(1,12,12,dtype=np.int16)
for _val in _transactions:
    _year = np.random.choice([2019,2020,2021])
    _month = np.random.choice(_months)
    _day = np.random.choice(_days)
    incomes_auto.append(Income(float(_val), 1/365,datetime.date(_year,_month,_day),None,one_time=True))

# Create an iterable of income objects
incomes = [INC1,INC2,INC3]
incomes.extend(incomes_auto)
# Define your plotting period
start_date = min([x.start_date for x in incomes])
end_date = datetime.date(2021,12,1)

# Calculate gradients and values
gradients, values = calculate_gradients(incomes, start_date, end_date)

#Plot the income objects
plot_integral(values, start_date, end_date, smooth=True, smooth_type='LSQ-bspline')
plot_integral(values, start_date, end_date, smooth=True, smooth_type='bv-bspline')
plot_integral(values, start_date, end_date, smooth=True, smooth_type='Smooth-bspline')
plot_integral(values, start_date, end_date, smooth=True, smooth_type='wiener')
plot_integral(values, start_date, end_date, smooth=False)
