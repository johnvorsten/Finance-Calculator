# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:04:54 2019



@author: z003vrzk
"""
# Python imports
import datetime, re

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d, proj3d
from mpl_toolkits.mplot3d.axis3d import get_flip_min_max
from matplotlib.ticker import FuncFormatter
from scipy import interpolate, signal
import pandas as pd

# Local imports



#%%


class Income():

    def __init__(self, income, period, start_date, end_date, best_case=None,
                 worst_case=None, probability=None, one_time=False):
        """Inputs
        -------
        income : income you expect - define this per-period
        period : the time over you receive income. Format days/(365 days)
        time_start : start time for receiving this income (datetime.date object)
        time_end : end time for receiving this income (datetime.date object)
        best_case : best case income/expense scenario. income = best_case if
            best_case is defined.
        worst_case : worst case income/expense scenario
        probability : (list | np.array) probability distribution between best
            and worst case.
            Type list or np.array of shape (11,). Structure [1,p(n-1)..0].
            Use this parameter to give the graph a different shape (risk distribution)
            If best_case is defined and proba=None,
            proba will be a liner gradient between [1 ... 0].
            Try np.linspace(0,1,num=11)
        one_time : if this is a one_time expense/income. In this case use
            time_start to signify the date of expense
        tax : enable tax calculation for income source
        effective_tax_rate : your effective tax rate"""

        # Keep track of base income
        self.income = income

        # Best case and worse case for plotting
        if best_case is not None:
            assert worst_case is not None, 'Please enter worst_case'
            self.worst_case = worst_case
            self.best_case = best_case
        else:
            self.best_case = income
            self.worst_case = income


        days = period * 365
        error_msg_days = 'Enter a period greater than 1/365 and less than 365/365'
        error_msg_type = 'Income must be int or float type'
        assert (days >= 1 and days <= 365) is True, error_msg_days
        assert (type(income)==int or type(income)==float), error_msg_type

        # Handle dates
        if start_date == end_date:
            """The dates must be a gradient, and even through a transaction
            Occurs on a date it must be represented by a gradient between two
            Days"""
            self.start_date = start_date
            self.end_date = end_date + datetime.timedelta(days=1)
        else:
            self.start_date = start_date
            self.end_date = end_date

        # The period is how frequently the transaction occurs
        if one_time:
            self.period = 1/365
            # Enforce the end_date is one day after start_date
            self.end_date = start_date + datetime.timedelta(days=1)
        else:
            self.period = period

        if probability is None:
            self.probability = np.array([1,0.9,0.8,0.7,0.6,0.5,
                                         0.4,0.3,0.2,0.1,0])
        else:
            msg='Length of probability array must be 11, got {}'
            assert len(probability) == 11, msg.format(len(probability))
            self.probability=probability

    def run(self, run_time):
        """
        inputs
        -------
        run_time : (datetime.timedelta) period you want to know derivative.
        If not between self.time_start and self.time_end return 0

            Return the derivative for the time period specified.
        The period is by day by default.  Integrating is the job of
        a different class creating the graph
        run_time : """

        if (run_time > self.end_date or run_time < self.start_date):
            return 0
        else:
            return self.calc_derivative()

    def calc_derivative(self):

        days = self.period * 365
        income_proba = (self.best_case - self.worst_case)*(self.probability) + self.worst_case
        income_rate = income_proba / days

        return income_rate

    def __repr__(self):
        msg='<Income(income={}, best={}, worst={}, start={}, end={}, period={}/365>'
        return msg.format(self.income,self.best_case,self.worst_case,
                          self.start_date,self.end_date,int(self.period * 365))

    def __str__(self):
        msg='{income:{},best_case:{},worst_case:{},start_date:{},end_date:{},period={}/365}'
        return msg.format(self.income,self.best_case,self.worst_case,
                          self.start_date,self.end_date,int(self.period * 365))


def calculate_gradients(incomes, start_date, end_date):
    """Calculate the gradients from incomes
    integrate the gradients from incomes to net worth

    Consider a 3-dimensional space (date, risk, net value)
    On each date in the future a person has a net value with projected risk

    inputs
    -------
    incomes : (iterable) of income objects
    start_date : (datetime.date) beginning date to calculate gradients from
        income objects. If the date of an income is earlier than start_date,
        then the income object will not appear in the resulting net worth
    end_date : (datetime.date) ending date to calculate gradients from
        income objects. If the date of an income is later than end_date,
        then the income object will not appear in the resulting net worth"""

    # Check income objects
    lengths = []
    for income in incomes:
        lengths.append(len(income.probability))
    msg='All income obects must have the same shape probability array. Got {}'
    assert len(set(lengths)) == 1, msg.foramt(set(lengths))

    # Income gradients per day
    delta = end_date - start_date
    dates = [(start_date + datetime.timedelta(days=i)) for i in range(delta.days)]

    """Keep track of values and gradients
    values[i,1,k] is the net value at day i with risk probability k
    values[i,1,:] is the net value distribution over a risk probability distribution
    """
    n_days = len(dates)
    risk_points = 11
    values = np.zeros((n_days, 1, risk_points))
    gradients = np.zeros((n_days, 1, risk_points))

    """Iterate through income objects and summate the income gradients
    (Aka income/loss per day)"""
    for income in incomes:

        try:
            index_start = dates.index(income.start_date)
        except ValueError:
            # Date not in list
            if income.start_date <= start_date:
                # Income start date is before graphing region
                index_start = 0
            else:
                # Income start date is after graphing region
                continue

        try:
            index_end = dates.index(income.end_date)
        except ValueError:
            # Date not in list
            if income.end_date >= end_date:
                # Income end date is after graphing region
                index_end = len(dates) - 1
            else:
                # Income end date is before graphing region
                continue

        gradient = income.calc_derivative()
        gradients[index_start:index_end,0,:] = gradients[index_start:index_end,0,:] + gradient

    # values = integrate.cumtrapz(gradients, axis=0, initial=np.mean(gradients[0,0,:]))
    values = np.cumsum(gradients, axis=0)

    return gradients, values


class RotateTickLabel:
    def __init__(self, axis, axes):
        """
        Rotate a set of axis labels relative to the axis
        Hold state information (axis) to define which axis ticklabels should
        be redrawn
        inputs
        -------
        axis : (mpl_toolkits.mplot3d.axis3d.Axis) This is the Axis object
            you want to rotate tick labels to. It is NOT the AXES object -
            see the plot example for what to pass
        axes : (mpl_toolkits.mplot3d.axes3d.Axes3D) This is the Axes object
            that is probably a subplot of a figure. See the plot
            example below for arguments to pass"""
        self.axis = axis
        self.axes = axes
        self.cid = axes.figure.canvas.mpl_connect('draw_event', self)

    def __call__(self, event):
        self.set_axis_label_rotate(event)

    def set_axis_label_rotate(self, event):
        """Rotate a set of axis labels relative to the axis
        inputs
        -------
        event : (matplotlib.backend_bases.DrawEvent)
        https://matplotlib.org/3.2.2/api/backend_bases_api.html#matplotlib.backend_bases.DrawEvent"""

        # Setup
        renderer = event.renderer
        axes = self.axes # Axes - includes axes.xaxis and axes.yaxis
        axis = self.axis # Axis

        info = axis._axinfo
        mins, maxs, centers, deltas, tc, highs = axis._get_coord_info(renderer)

        # Determine grid lines
        minmax = np.where(highs, maxs, mins)

        # Draw main axis line
        juggled = info['juggled']
        edgep1 = minmax.copy()
        edgep1[juggled[0]] = get_flip_min_max(edgep1, juggled[0], mins, maxs)

        edgep2 = edgep1.copy()
        edgep2[juggled[1]] = get_flip_min_max(edgep2, juggled[1], mins, maxs)
        pep = proj3d.proj_trans_points([edgep1, edgep2], renderer.M)

        # Draw labels
        # The transAxes transform is used because the Text object
        # rotates the text relative to the display coordinate system.
        # Therefore, if we want the labels to remain parallel to the
        # axis regardless of the aspect ratio, we need to convert the
        # edge points of the plane to display coordinates and calculate
        # an angle from that.
        peparray = np.asanyarray(pep)
        dx, dy = (axes.axes.transAxes.transform([peparray[0:2, 1]]) -
                  axes.axes.transAxes.transform([peparray[0:2, 0]]))[0]

        # Rotate label
        for tick_label in axis.get_majorticklabels():
            angle = art3d._norm_text_angle(np.rad2deg(np.arctan2(dy, dx)))
            tick_label.set_rotation(angle)

        return None


def spline_smoothing_factor(size):
    high, low = (size - np.sqrt(2*size), size + np.sqrt(2*size))
    return high, low


def value_label_formatter(x, pos):
    """Inputs
    -------
    x : (float) value of tick label
    pos : () position of tick label
    outputs
    -------
    (str) formatted tick label"""

    if x > 100000000:
        # 100,000,000
        return '{:1.0e}'.format(x)
    if x > 1000000:
        # 1,000,000
        return '{:1.0f}M'.format(x / 1e7)
    elif x >= 1000:
        return '{:1.0f}k'.format(x / 1e3)
    else:
        return '{:1.0f}'.format(x)

    return str(x)


def plot_integral(values, start_date, end_date,
                  smooth=True, smooth_type='LSQ-bspline'):
    """
    inputs
    -------
    incomes : (list) of Income objects
    start_date : (datetime.date)
    end_date : (datetime.date)
    smooth : (bool)
    smooth_type : (str) one of ['bv-bspline','LSQ-bspline','Smooth-bspline',
        'wiener']. The smoothing method to apply to the 3-d surface
    outputs
    -------
    None
    """

    msg='values should be numpy array, got {}'
    assert isinstance(values, np.ndarray), msg.format(type(values))
    msg='date must be datetime.date, got {}'
    assert isinstance(start_date, datetime.date),msg.format(type(start_date))
    assert isinstance(end_date, datetime.date),msg.format(type(end_date))

    # Dates for labeling x axis
    delta = end_date - start_date
    dates = [(start_date + datetime.timedelta(days=i)) for i in range(delta.days)]

    #Apply b-spline fit to surface
    if smooth:

        if smooth_type == 'bv-bspline':
            x = np.arange(values.shape[0]) # Dates
            y = np.arange(values.shape[2]) # Probability
            xy, yx = np.meshgrid(x,y)
            xs = np.ravel(xy)
            ys = np.ravel(yx)
            # # Unpack in row-major order :)
            z = values[:,0,:]
            zs = np.ravel(values[:,0,:], order='F')

            """A sequence of length 5 returned by bisplrep containing the knot
            locations, the coefficients, and the degree of the spline:
                [tx, ty, c, kx, ky]."""
            s = sum(spline_smoothing_factor(xs.shape[0])) / 2
            tck = interpolate.bisplrep(xs, ys, zs, s=s*100, eps=1e-12)
            z3d = interpolate.bisplev(x, y, tck)

        elif smooth_type == 'LSQ-bspline':
            # Least squares bivariate spline approximation
            x = np.arange(values.shape[0]) # Dates
            y = np.arange(values.shape[2]) # Probability
            xy, yx = np.meshgrid(x,y)
            xs = np.ravel(xy)
            ys = np.ravel(yx)
            # # Unpack in row-major order :)
            z = values[:,0,:]
            zs = np.ravel(values[:,0,:], order='F')

            """Weighted least-squares bivariate spline approximation
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQBivariateSpline.html"""
            # Find knots
            # Knots are the intervals polynomials are broken into. More knots
            # Mean a more precise spline
            # knots = np.linspace(start, end, n_knots)
            tx = np.linspace(0, x.shape[0], int(np.sqrt(2 * x.shape[0])))
            ty = np.linspace(0, y.shape[0], int(np.sqrt(2 * x.shape[0])))
            # Evaluate spline
            LSQBSpline = interpolate.LSQBivariateSpline(xs,ys,zs,tx,ty)
            z3d = LSQBSpline(x, y, grid=True)

        elif smooth_type == 'Smooth-bspline':
            # Smooth bivariate spline approximation
            x = np.arange(values.shape[0]) # Dates
            y = np.arange(values.shape[2]) # Probability
            xy, yx = np.meshgrid(x,y)
            xs = np.ravel(xy)
            ys = np.ravel(yx)
            # # Unpack in row-major order :)
            z = values[:,0,:]
            zs = np.ravel(values[:,0,:], order='F')

            """Smooth bivariate spline approximation
            see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothBivariateSpline.html"""
            # s = sum(spline_smoothing_factor(xs.shape[0])) / 2
            SmoothBSpline = interpolate.SmoothBivariateSpline(xs, ys, zs)
            z3d = SmoothBSpline(x, y, grid=True)

        elif smooth_type == 'wiener':
            # Smooth bivariate spline approximation
            x = np.arange(values.shape[0]) # Dates
            y = np.arange(values.shape[2]) # Probability
            xy, yx = np.meshgrid(x,y)
            z = values[:,0,:]
            z3d = signal.wiener(z, mysize=(25,1))

    else:

        x = np.arange(values.shape[0])
        y = np.arange(values.shape[2])
        xy, yx = np.meshgrid(x,y)
        z3d = values[:,0,:]

    # X labels - dates
    num_ticks = 4
    tick_index = np.linspace(min(x), max(x), num=num_ticks, dtype=np.int16)
    xticks = x[tick_index]
    xlabel_index = tick_index
    xlabel_date = [dates[idx] for idx in xlabel_index]
    xlabels = [x.isoformat() for x in xlabel_date]
    # Y labels
    yticks = [int(y.shape[0]/2)]
    ylabels = ['Worst → Best Case']

    """2-dimensional plotting"""
    fig2d = plt.figure(1)
    ax2d = fig2d.add_subplot(111)
    z2d = values.mean(axis=2)
    line = ax2d.plot(x, z2d, label='Bank Value', linewidth=2)

    ax2d.set_title('Mean Value Non-smoothed')
    ax2d.set_xlabel('Date')
    ax2d.set_ylabel('Total Value [$]')
    ax2d.set_xticks(xticks)
    ax2d.set_xticklabels(xlabels)

    """3-dimensional plotting"""
    fig3d = plt.figure(2)
    ax3d = fig3d.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(xy, yx, z3d.transpose(), cmap='summer', linewidth=0,
                           antialiased=False)
    # Label date axis
    ax3d.set_xticks(xticks)
    ax3d.set_xticklabels(xlabels)
    # Label probability axis
    ax3d.set_yticks(yticks)
    ax3d.set_yticklabels(ylabels)
    # Connect event to rotate probability axsi label
    rotateTickLabel = RotateTickLabel(ax3d.yaxis, ax3d)
    fig3d.canvas.mpl_connect('draw_event', rotateTickLabel)
    # Label value axis
    formatter = FuncFormatter(value_label_formatter)
    ax3d.zaxis.set_major_formatter(formatter)
    # ax3d.tick_params(pad=10)

    # Label axis
    fig3d.colorbar(surf, shrink=0.5, aspect=10)
    ax3d.set_title('Value, Time, Risk')
    ax3d.set_xlabel('Date', labelpad=15)
    ax3d.set_zlabel('Total Value [$]', labelpad=10)

    # Elev is in z plane, azim is in x,y plane
    ax3d.view_init(elev=20, azim=125)

    for angle in range(100,125):
        ax3d.view_init(elev=20, azim=angle)
        plt.draw()
        plt.pause(0.003)

    plt.show()

    return None


def create_transaction_objects(transactions_dataframe,
                               value_col='Amount',
                               date_col='Date'):
    """
    inputs
    -------
    transactions_dataframe : (pandas.DataFrame) of transaction records. Each
        row should represent a single income/expense with information about the
        value of the transaction, and date of the transaction
        Example :
                {
                     Date      Amount  Simple Description
                0    6/3/2019  -52.29  H-E-B  #388 AUSTIN TX
                1    5/23/2019  -2.92  SQ *BAKERY LORRAINE Austin TX
                2    5/20/2019  37.3   Transfer
                }

    value_col : (str) denotating which column the value of the transaction is
        in. Above the value column is named 'Amount'
    date_col : (str) denotating which column the date of the transaction is in.
        Above the date column is named 'Date'
    outputs
    --------
    transactions : (list) of Income objects for each transaction (row) in
        transactions_dataframe
    """
    incomes = []
    date_format = r'%m/%d/%Y'
    period = 1/365
    reg = re.compile('[^0-9.-]')

    for idx, row in transactions_dataframe.iterrows():
        # TODO add more robust regex here
        # Replace commas and '$' strings
        income_str = row[value_col]
        income_str = income_str.replace(',', '')
        income = float(income_str)
        date_str = row[date_col]
        my_date = datetime.datetime.strptime(date_str, date_format) # datetime
        my_date = my_date.date()

        income_obj = Income(income, period, my_date, my_date,
                            one_time=True)
        incomes.append(income_obj)

    return incomes



def categorize_transactions(transaction_dataframe,
                            cat_col='Category',
                            date_col='Date',
                            value_col='Amount',
                            by_year=True):
    """Return a set of categories of expenses and average
    spending on these categories over the timeframe from the imported
    data.
    inputs
    -------
    transaction_dataframe : (pd.DataFrame)
    cat_col : (str)
    date_col : (str)
    value_col : (str)
    by_year : (bool)
    Returns
    -------
    categories : (dict) of category : expense/year
    categories_year : (dict) of category : expense/year.  This is not
        averaged across the whold data period = returns a category for each
        year
    """

    cat_tags = list(set(transaction_dataframe[cat_col]))
    categories = {}
    categories_year = {}

    dates = transaction_dataframe[date_col]
    date_format = r'%m/%d/%Y'
    dates = [datetime.strptime(x, date_format) for x in dates]
    unique_years = list(set([x.year for x in dates]))


    for cat in cat_tags:

        cat_index = transaction_dataframe[cat_col] == cat
        cat_vals = transaction_dataframe.loc[cat_index][value_col]
        cat_vals = cat_vals.str.replace(',','').astype(float)
        cat_val = cat_vals.sum()
        categories[cat] = cat_val

    date_series = pd.Series(dates)
    year_series = date_series.map(lambda x: x.year)

    for year in unique_years:

        for cat in cat_tags:

            cat_index = (transaction_dataframe[cat_col] == cat) & (year_series == year)
            cat_vals = transaction_dataframe.loc[cat_index][value_col]
            cat_vals = cat_vals.str.replace(',','').astype(float)
            cat_val = cat_vals.sum()
            new_cat = cat + '_' + str(year)
            categories_year[new_cat] = cat_val

    if by_year:
        return categories_year
    else:
        return categories




