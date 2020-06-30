# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:39:44 2019

@author: z003vrzk
"""

import Personal_Finance_Calculator as pfc
from datetime import date

"""Test Graph"""

#Example income instance
"""my_income1 = pfc.Income(income1, period1, time_start1, time_end1, 
                   best_case=best_in1, worst_case=worst_in1)"""

#Create income instances
inc1=pfc.Income(59000*0.8,365/365,date(2019,6,5),date(2023,6,5),
                best_case=68000*0.84,worst_case=32000*0.8, tax=True) #Income
inc2=pfc.Income(-4500,1/365,date(2020,4,1),date(2020,4,1),
                best_case=-3800,worst_case=-10000,one_time=True) #Ring
inc3=pfc.Income(-7500,1/365,date(2022,2,20),date(2022,2,20),
                best_case=-6800,worst_case=-15000,one_time=True) #wedding
inc4=pfc.Income(-2000,31/365,date(2019,6,5),date(2025,6,5),
                best_case=-1800,worst_case=-2100,one_time=True) #General Expenses
inc5=pfc.Income(450,15/365,date(2019, 6, 5),date(2025, 6, 5),
                best_case=500,worst_case=440) #Other Income
inc6=pfc.Income(-12500,365/365,date(2023, 6, 5),date(2025, 6, 5),
                best_case=-10000,worst_case=-15000) #school

#Import bank .csv
path = r'Transaction_Data.csv'
income_importer = pfc.Income_Import(path)
incomes_auto = income_importer.create_transaction_objects()

incomes = [inc1, inc2,inc3,inc4,inc5,inc6]
incomes.extend(incomes_auto)
#Define your plotting period
time_start = min([x.time_start for x in incomes])
time_end = max([x.time_start for x in incomes])

#Plot the income objects
pfc.plot_integral(incomes, time_start, time_end, smooth=False)
