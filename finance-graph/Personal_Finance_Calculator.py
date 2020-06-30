# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:04:54 2019



@author: z003vrzk
"""

from datetime import date, timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import pandas as pd

class Income():
    
    def __init__(self, income, period, time_start, time_end, best_case=None, 
                 worst_case=None, proba=None, one_time=False, tax=False, 
                 effective_tax_rate=0.2):
        """Inputs
        -------
        income : income you expect - define this per-period
        period : the time over you receive income. Format days/(365 days)
        time_start : start time for receiving this income (datetime.date object)
        time_end : end time for receiving this income (datetime.date object)
        best_case : best case income/expense scenario. income = best_case if 
            best_case is defined.
        worst_case : worst case income/expense scenario
        proba : probability distribution between best and worst case. Type list 
            or np.array of shape (10,). Structure [1,p(n-1)..0]. Intended to give
            the graph a different shape. If best_case is defined and proba=None,
            proba will be a liner gradient between [1 ... 0]. 
            Try np.linspace(0,1,num=11)
        one_time : if this is a one_time expense/income. In this case use 
            time_start to signify the date of expense
        tax : enable tax calculation for income source
        effective_tax_rate : your effective tax rate"""
        
        
        self.best_case = best_case
        if best_case is not None:
            assert worst_case is not None, 'Please enter worst_case'
#            self.income = best_case
            self.worst_case = worst_case
            
            if proba is not None:
                assert len(proba)==11, 'Probabiltiy must be of length 10'
        else:
            self.best_case = income
            self.worst_case = income
        
        if tax:
            self.income = (1-effective_tax_rate)*income
            self.best_case = (1-effective_tax_rate)*best_case
            self.worst_case = (1-effective_tax_rate)*worst_case
        else:
            self.income = income
            
        self.time_start = time_start
        days = period*365
        error_msg_days = 'Enter a period greater than 1/365 and less than 365/365'
        error_msg_type = 'Income must be int or float type'
        assert (days >= 1 and days <= 365) is True, error_msg_days
        assert (type(income)==int or type(income)==float), error_msg_type
        
        if one_time:
            self.period = 1/365
            self.time_end = date(time_start.year, time_start.month, time_start.day) #add 1?
        else:
            self.period = period
            self.time_end = time_end
            
        if proba is None:
            self.proba = np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0])

    def run(self, run_time):
        """Return the derivative for the time period specified.  
        The period is by day by default.  Integrating is the job of
        a different class creating the graph
        run_time : period you want to know derivative. If not between 
        self.time_start and self.time_end return 0"""
        
        if (run_time > self.time_end or run_time < self.time_start):
            return 0
        else:
            return self.calc_derivative()
    
    def calc_derivative(self):
        days = self.period*365
        income_proba = (self.best_case - self.worst_case)*(self.proba) + self.worst_case
        income_rate = income_proba / days
        
        return income_rate


def plot_integral(incomes, start_date, end_date, smooth=True):
    global values
    
    delta = end_date - start_date
    dates = [(start_date + timedelta(days=i)) for i in range(delta.days)]
    m = delta.days
    n = 11
    gradients = np.zeros((m,n))
    values = np.zeros((m,n))
    
    for m, _date in enumerate(dates):
    
        for income in incomes:
            #Iterate through all gradients
            gradient = income.run(_date)
            gradients[m,:] = gradients[m,:] + gradient
            values[m] = np.sum(gradients[:m,:], axis=0) #(m,n) m=days, n=probability
            
    #Start doing plotting
    fig3d = plt.figure(1)
    fig2d = plt.figure(2)
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax2d = fig2d.add_subplot(111)
    
#    X_ = np.arange(0,len(dates), 1)
#    y_ = np.linspace(1,0,num=11)
#    X, y = np.meshgrid(X_, y_)
#    Z3d = values.transpose() #Get integrated values
    Z2d = values.mean(axis=1)
    
    #Apply b-spline fit to surface
    if smooth: 
        m, n = values.shape[0], values.shape[1]
        pts = np.zeros((m*n, 3))
        idx = 0
        
        for _row in range(m):
            for _col in range(n):
                pts[idx, 0] = _row
                pts[idx, 1] = _col
                pts[idx, 2] = values[_row,_col]
                idx += 1
                
    #    sets = 100
    #    _segment = int(min(sets*n, pts.shape[0]*0.5))
    #    tck = interpolate.bisplrep(pts[:_segment,0], pts[:_segment,1], pts[:_segment,2])
        tck = interpolate.bisplrep(pts[:,0], pts[:,1], pts[:,2])
        
        x_grid = np.arange(0, values.shape[0], 1)
        y_grid = np.linspace(1, values.shape[1],num=11)
        Z3d = interpolate.bisplev(x_grid, y_grid, tck).transpose()
        
        X_ = np.arange(0, len(dates), 1)
        y_ = np.linspace(1, 0, num=11)
        X, y = np.meshgrid(X_, y_)
        
    #No surface fit - plot actual data
    else: 
        
        X_ = np.arange(0,len(dates), 1)
        y_ = np.linspace(1,0,num=11)
        X, y = np.meshgrid(X_, y_)
        Z3d = values.transpose()
    
    surf = ax3d.plot_surface(X, y, Z3d, cmap='summer', linewidth=0, 
                           antialiased=False)
    line = ax2d.plot(X_, Z2d, label='Bank Value', linewidth=2)
    
    #Change int series tick marks to date marks
    num_ticks = 4
    tick_index = np.linspace(min(X_), max(X_), num=num_ticks, dtype=np.int16)
    #ticks = X[tick_index]
    ticks = X_[tick_index]
    ax3d.set_xticks(ticks) 
    ax2d.set_xticks(ticks)
    
    label_index = tick_index
    label_date = [dates[idx] for idx in label_index]
    labels = [x.isoformat() for x in label_date]
    ax3d.set_xticklabels(labels) 
    ax2d.set_xticklabels(labels)
    fig3d.colorbar(surf, shrink=0.5, aspect=5)
    
    ax3d.set_xlabel('Date')
    ax3d.set_ylabel('Probability')
    ax3d.set_zlabel('Total Value [$]')
    ax2d.set_xlabel('Date')
    ax2d.set_ylabel('Total Value [$]')
    
    ax3d.view_init(elev=20, azim=125) #elev is in z plane, azim is in x,y plane

    for angle in range(100,125):
        ax3d.view_init(elev=20, azim=angle)
        plt.draw()
        plt.pause(0.003)
    
    plt.show()




class Income_Import():
    
    def __init__(self, path):
        self.tx_df = pd.read_csv(path)

    def create_transaction_objects(self, 
                                   value_col='Amount',
                                   date_col='Date'):
        incomes = []
        date_format = r'%m/%d/%Y'
        period = 1/365
        
        for idx, row in self.tx_df.iterrows():
            income_str = row[value_col]
            income_str = income_str.replace(',', '')
            income = float(income_str)
            date_str = row[date_col]
            my_date = datetime.strptime(date_str, date_format) #datetime
            my_date = my_date.date()
            
            income_obj = Income(income, period, my_date, my_date, 
                                one_time=True)
            incomes.append(income_obj)
        
        return incomes
    
    def categorize_transactions(self, cat_col='Category', 
                                date_col='Date',
                                value_col='Amount',
                                by_year=True):
        """Return a set of categories of expenses and average
        spending on these categories over the timeframe from the imported
        data.
        Returns
        -------
        categories : {} dict of category : expense/year
        categories_year : {} dict of category : expense/year.  This is not
            averaged across the whold data period = returns a category for each
            year"""
        cat_tags = list(set(self.tx_df[cat_col]))
        categories = {}
        categories_year = {}
        
        _dates = self.tx_df[date_col]
        date_format = r'%m/%d/%Y'
        _dates = [datetime.strptime(x, date_format) for x in _dates]
        unique_years = list(set([x.year for x in _dates]))
        
        
        for cat in cat_tags:
            
            cat_index = self.tx_df[cat_col] == cat
            cat_vals = self.tx_df.loc[cat_index][value_col]
            cat_vals = cat_vals.str.replace(',','').astype(float)
            cat_val = cat_vals.sum()
            categories[cat] = cat_val
            
        date_series = pd.Series(_dates)
        year_series = date_series.map(lambda x: x.year)
            
        for year in unique_years:
            
            for cat in cat_tags:
                
                cat_index = (self.tx_df[cat_col] == cat) & (year_series == year)
                cat_vals = self.tx_df.loc[cat_index][value_col]
                cat_vals = cat_vals.str.replace(',','').astype(float)
                cat_val = cat_vals.sum()
                new_cat = cat + '_' + str(year)
                categories_year[new_cat] = cat_val

        if by_year:
            return categories_year
        else:
            return categories





#"""Testing"""
#income1 = 100
#best_in1 = 110
#worst_in1 = 90
#
#loss1 = -100
#best_loss1 = -90
#worst_loss1 = -110
#
#single_in1 = 100
#single_in_best1 = 110
#single_in_worst1 = 90
#
#single_loss1 = -100
#single_loss_best1 = -90
#single_loss_worst1 = -110
#
#period = 1/365
#time_start = date(2019, 6, 1)
#time_end = date(2019, 6, 5)
#
#my_income = Income(income1, period, time_start, time_end, 
#                   best_case=best_in1, worst_case=worst_in1)
#my_loss = Income(loss1, period, time_start, time_end, 
#                 best_case=best_loss1, worst_case=worst_loss1)
#my_single_in = Income(single_in1, period, time_start, time_end, 
#                      best_case=single_in_best1, worst_case=single_in_worst1, one_time=True)
#my_single_loss = Income(single_loss1, period, time_start, time_end, 
#                        best_case=single_loss_best1, worst_case=single_loss_worst1, one_time=True)
#
##Income
#print('best_case {}, {}'.format(best_in1,my_income.best_case))
#print('worst_case {}, {}'.format(worst_in1,my_income.worst_case))
#print('income {}, {}'.format(income1,my_income.income))
#income1_gradient = ((best_in1-worst_in1)*my_income.proba + worst_in1)/(period*365)
#print('Income Gradient {}, {}'.format(income1_gradient, my_income.calc_derivative()))
#
##Loss
#print('\n\n')
#print('best_case {}, {}'.format(best_loss1,my_loss.best_case))
#print('worst_case {}, {}'.format(worst_loss1,my_loss.worst_case))
#print('income {}, {}'.format(loss1,my_loss.income))
#loss1_gradient = ((best_loss1-worst_loss1)*my_loss.proba + worst_loss1)/(period*365)
#print('Income Gradient {}, {}'.format(loss1_gradient, my_loss.calc_derivative()))
#
##Single Income
#print('best_case {}, {}'.format(single_in_best1,my_single_in.best_case))
#print('worst_case {}, {}'.format(single_in_worst1,my_single_in.worst_case))
#print('income {}, {}'.format(single_in1,my_single_in.income))
#income1_gradient = ((single_loss_best1-single_loss_worst1)*my_single_in.proba + single_in_worst1)/(period*365)
#print('Income Gradient {}, {}'.format(income1_gradient, my_single_in.calc_derivative()))
#
##Single Loss
#print('\n\n')
#print('best_case {}, {}'.format(single_loss_best1,my_single_loss.best_case))
#print('worst_case {}, {}'.format(single_loss_worst1,my_single_loss.worst_case))
#print('income {}, {}'.format(single_loss1,my_single_loss.income))
#loss1_gradient = ((single_loss_best1-single_loss_worst1)*my_single_loss.proba + single_loss_worst1)/(period*365)
#print('Income Gradient {}, {}'.format(loss1_gradient, my_single_loss.calc_derivative()))
#
#
#delta = time_end - time_start
#dates = [(time_start + timedelta(days=i)) for i in range(delta.days + 1)]
































