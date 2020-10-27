# Finance-Calculator  
Visualization of incomes and outcomes in a (3) dimensional space 

![Example 3D](https://github.com/johnvorsten/Finance-Calculator/blob/master/images/3d.png)  
Project Description  
The goal of this project is to visualize income and expenses.  The output visualization should give the viewer an understanding of : 
1.	Their income over time (past and future if desired)
2.	How income is impacted by best-case and worst-case scenarios
This project was inspired by a discussion about finances.  How much money do I have to spend, not only now, but at future dates?

Free cash is complicated by : 
1.	Current and future levels of income  
2.  Current and future expenses
3.  Uncertainty of future expenses - Will my car break down?

What will future disposable income look like. This package has the following features : 
1.	Consider past expenses  
2.	Consider future projected expenses and incomes
a.	Graph future expenses with a level of uncertainty. This will be the third axis – probability 
3.	Look presentation-worthy  
4.	Be able to parse historical data – aka take in a spreadsheet of historical bank transactions to make historical graphing easy  


## Core functions and classes 

### Income(income, period, start_date, end_date, best_case=None, worst_case=None, probability=None, one_time=False)   
The income class models cash incomes and expenses. The class can handle transactions that occur over time (yearly incomes), or at one time (single purchases).  It optionally accepts a probability matrix for projecting worst/best case scenarios.

Parameters  
* income : (float/int) income you expect - define this per-period
* period : (float) the time over you receive income. Format days/(365 days)
* time_start : (datetime.date) start time for receiving this income
* time_end : (datetime.date) end time for receiving this income
* best_case : (float/int) best case income/expense scenario.
    income = best_case if best_case is defined.
* worst_case : (float/int) worst case income/expense scenario
* probability : (list | np.array | iterable) probability distribution
    between best and worst case.
    Type list or np.array of shape (11,). Structure [1,p(n-1)..0].
    Use this parameter to give the graph a different shape (risk distribution)
    If best_case is defined and proba=None,
    proba will be a liner gradient between [1 ... 0].
    Try np.linspace(0,1,num=11)
* one_time : (bool) if this is a one_time expense/income.
    In this case use time_start to signify the date of expense 

### calculate_gradients(incomes, start_date, end_date)
Calculate the gradients from incomes and integrate the gradients from incomes to net worth

Parameters
* incomes : (iterable) of income objects
* start_date : (datetime.date) beginning date to calculate gradients from
    income objects. If the date of an income is earlier than start_date,
    then the income object will not appear in the resulting net worth
* end_date : (datetime.date) ending date to calculate gradients from
    income objects. If the date of an income is later than end_date,
    then the income object will not appear in the resulting net worth

outputs
* gradients : (np.array) Derivative of cumulative worth distribution over 
    start_date to end_date
* values : (np.array) Cumulative worth distribution over start_date to end_date

### plot_integral(values, start_date, end_date, smooth=True, smooth_type='LSQ-bspline')
Plot cumulative net worth values over time

Parameters
* values : (np.array) of cumulative net worth over time. See calculate_gradients()
* start_date : (datetime.date) Start date of plotting
    This MUST be the same start_date used to calculate values
* end_date : (datetime.date) end date of plotting
    This MUST be the same end_date used to calculate values
* smooth : (bool) Apply smoothing to the 3D graph surface
* smooth_type : (str) one of ['bv-bspline','LSQ-bspline','Smooth-bspline',
    'wiener']. The smoothing method to apply to the 3-d surface

Outputs
* Graph (2D and 3D)

## Usage Example

1.	Create income objects.  
2.	(Optional) Import bank transaction .csv file  
3.	Calculate net worth gradient
4.  Graph net worth values 

```python
# Basic income - $60,000 to $40,000 per year
INC1 = Income(60000*0.7, 365/365, datetime.date(2019,6,5), datetime.date(2023,6,5), best_case=60000*0.7, worst_case=40000*0.7) #Income

# Single expense of $4,000, best $3,000 worst $5,000
INC2 = Income(-4000, 1/365,datetime.date(2019,10,1),datetime.date(2020,10,1), best_case=-3000, worst_case=-5000, one_time=True)

# Single expense of $7,500
INC3 = Income(-7500,1/365,datetime.date(2021,10,20),datetime.date(2021,10,20), one_time=True)

# Monthly expense of $2,000
INC4 = Income(-2000,31/365, datetime.date(2019,6,5), datetime.date(2025,6,5), best_case=-1800, worst_case=-2100, one_time=True)

# Other income bi-weekly
INC5 = Income(450, 14/365, datetime.date(2019, 6, 5), datetime.date(2025, 6, 5), best_case=500, worst_case=440)

# Single expense of $12,500 yearly over two years ($25,000 total expense)
# Range from $10,000 to $15,000 expense
INC6 = Income(-12500, 365/365, datetime.date(2023,6,5), datetime.date(2025,6,5), best_case=-10000,worst_case=-15000)

# Import bank transactions .csv for projection of daily expenses
path = r'../path_to/Transaction_Data.csv'
transaction_dataframe = pd.read_csv(path)
incomes_auto = create_transaction_objects(transaction_dataframe, value_col='Amount',date_col='Date')

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
```

The plotting function outputs a 2D image as well as a 3D image
![Example 3D](https://github.com/johnvorsten/Finance-Calculator/blob/master/images/2d%201.png)  

Transactions on a 1-day time period can make the graph appear “Choppy”.  This is especially true for large transactions relative to the current account balance.  
![Example 3D](https://github.com/johnvorsten/Finance-Calculator/blob/master/images/3d%202.png)  

This figure shows the “probability” axis of the 3D image.  Income can be projected in the worst case (0 probability) or best case (1 probability).  Different distributions can be input besides a linear distribution.  
![Example 3D](https://github.com/johnvorsten/Finance-Calculator/blob/master/images/3d%203.png)  
