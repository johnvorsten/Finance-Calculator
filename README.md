# Finance-Calculator
Visualization of incomes and outcomes in a (3) dimensional space
Project Description
The goal of this project is to visualize income and expenses.  The output visualization should give the viewer an understanding of : 
1.	Their income over time (past and future if desired)
2.	How income is impacted by best-case and worst-case scenarios
This project was inspired by an argument with my girlfriend  We were talking about getting engaged!  This was a great topic, except we were arguing about money : I didn’t think we had enough money; she thought we had plenty of money (a story as old as time).  The main point of argument was not about 1) how much we should spend on an engagement ring (although I wanted to spend less), or 2) how much money we have right now.  We argued about how much money we will have in the future.
Our future finances are complicated by a lot of things : 
1.	Engagement ring
2.	Wedding expenses
3.	School if I want to go back to get my masters
4.	Current and future levels of income
5.	If I need a new car, health insurance, health issues, feelings of security, putting money in retirement, leisure spending […] etc..
Because we were mostly worried about future spending, I wanted to make a graph to give me an idea of what my future disposable income would actually look like.  Therefore I wanted the following features : 
1.	Be able to graph past expenses
2.	Be able to graph future projected expenses
a.	Graph future expenses with a level of uncertainty.  I should be able to give best case and worst case scenarios AKA if a wedding costs $12,000 or $17,000 (or if school is $15,000  a year or $25,000 a year).  This will be the third axis – probability (the other axes will be time and money)
3.	Be intuitive to create new incomes/expenses
4.	Gracefully handle one-time incomes and continuous incomes
5.	Look presentation-worthy
6.	Be able to parse historical data – aka take in a spreadsheet of historical bank transactions to make historical graphing easy
Knowing this, lets get started.
Core Modules
My code is broken into three core modules : (2) classes and (1) plotting function.
Income Class
Income(income, period, time_start, time_end, best_case=None, worst_case=None, proba=None, one_time=False, tax=False, effective_tax_rate=0.2)
The income class models cash inflow and outflow.  I will call a cash inflow or outflow a transaction for simplicity.  The class can handle transactions that occur over time (yearly incomes), or at one time (single purchases).  It optionally accepts a probability matrix for projecting worst/best case scenarios and supports taxes for incomes. 
Parameters
•	income : Transaction value you expect - define this per-period
•	period : the time over the transaction happens. Format days/(365 days).  Example, for monthly rent input 30.5 / 365
•	time_start : start time for transaction (datetime.date object)
•	time_end : end time for transaction (datetime.date object)
•	best_case : best case transaction value scenario. income = best_case if best_case is defined.
•	worst_case : worst case transaction value scenario
•	proba : probability distribution between best and worst case. Type list or np.array of shape (10,). Structure [1, p(n-1) .. 0]. Intended to give the graph a different shape. If best_case is defined and proba=None, proba will be a linear gradient between [1 ... 0].  Try np.linspace(0,1,num=11)
•	one_time : if this is a one_time expense/income. In this case use time_start to signify the date of expense.  If one_time = True, then only time_start will be used in the run() method
•	tax : enable tax calculation for income source.  Uses effective_tax_rate to define a percentage.  If you want to inflate expenses by a certain percent, use a negative value (ex -0.08 for 8% sales tax).  The user should calculate other non-tax factors (401K, IRA, savings plans, other payments etc.) before using this class for typical earned income.
•	effective_tax_rate : your effective tax rate (default 20%)
Methods
Income.run(run_time)
•	returns a gradient across proba which defines the expected transaction value during a single day.  This method is called by the plotting function across a range of days (explained later).  The income class methods should not need to be called by the user except for troubleshooting or verification.  The primary job of this method is to return the income gradient if it is called within the class instances defined income period, or 0 if it is called outside of its time period.
Income. calc_derivative (run_time)
•	called by the run() method.  It uses a start date and end date to find the income gradient across a time.  The result is returned to run(), which is the expected transaction value during a single day.

Income_Import Class
Income(path)
Parameters
•	path: Directory to a csv file. Python will import the .csv file into a pandas dataframe and perform counting operations based on this data
The income_Import class is useful for importing transactions from bank statements or ledgers.  This class will also average transaction “types” or “categories” based on bank statements.  This is useful for projecting future expenses based on past actual transactions.
Methods
Income_Import. create_transaction_objects(value_col=’Amount’, date_col=’Date’)
•	Creates Income class instances based on the file passed to the class initializer.  The required rows of the bank records .csv file are <value_col> and <date_col>.  These columns should hold the transaction amounts, and transaction dates for each transaction.  The output is a list (iterable) of income classes. This is useful for passing to the plotting function.  The plotting function will iterate over these income classes, and plot their contributions to (+) or (-) cash flow.

Income_Import. categorize_transactions (cat_col='Category', date_col='Date',  by_year=True)
•	Outputs a dictionary of {category : value} based on the .csv file passed to the class initializer.  The required rows of the bank records .csv file are <cat_col>, <value_col>, and <date_col>.  <cat_col> is a column describing each transaction instances category (many banks export this information).  <value_col> again is the transaction amount. <date_col> is the transaction date.  The optional parameter by_year indicates if the user wants transactions categorized by year, or over the entire period in the .csv file.  This might be useful for more accurately projecting future expenses.

Usage Example

1.	Create custom income objects.
2.	Import bank transaction .csv file
3.	Create income objects from bank transactions
4.	Append (extend) custom income objects and imported income objects
5.	Pass the objects to the plotting function
