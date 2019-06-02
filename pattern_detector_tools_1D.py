""" Module for basic correlation analysis of large datasets extracted via the
Quandl Python API.
General idea:

1. download/load all data
2. format data as independent time series
3. correlate against each other (all combinations of 2)
4. profit??? bit.ly/2KbRCME
5. wait no, format it as a table (for pretty printing) or as a pandas dataframe
   so you can actually do somethign with it.

All of the time series used by this author were very short so the time to run
was usually less than a minute, so not much care has been taken for
optimisation.

+------------------------------------------------------------------------------
Example:
+------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pattern_detector_tools_1D as pdt
import pandas as pd
import quandl

quandl.ApiConfig.api_key = "<YOUR_KEY_STRING>"
# quand_codes is a dict of the quandl code repped as 'name of qset: code'
dset = pdt.DataSet(<QUANDL_CODES>)

dset.get_qsets()
dset.get_time_series()
dset.pickle_time_series(filename='time_series')
dset.load_time_series(filename='time_series')

dset.build_table()
dset.print_table(filename='table.txt',
                 sortby="Correlation r value",
                 reversesort=True, title="<MY_TITLE>")

comparison_list = ["<TIME_SERIES_NAME_1",
                   "<TIME_SERIES_NAME_2"]
dset.plot_ts("time_series_comparison", comparison_list)
plt.show()

dset.build_dataframe()
dset.print_dataframe(filename='df.txt')

# find all correlations between avocado and real estate data
avoframe1 = dset.dataframe[
                    dset.dataframe["B's Quandl Set"].str.contains(
                        "avocado", case=False)]
avoframe2 = dset.dataframe[
                    dset.dataframe["A's Quandl Set"].str.contains(
                        "avocado", case=False)]
with open('avos.txt', "w") as text_file:
    text_file.write(avoframe1.to_string() + '\n' +
                    avoframe2.to_string())

meme1 = avoframe1[avoframe1["A's Quandl Set"].str.contains(
                        "real_estate", case=False)]

meme2 = avoframe2[avoframe2["B's Quandl Set"].str.contains(
                        "real_estate", case=False)]
meme = pd.concat([meme1, meme2], ignore_index=True)
with open('meme.txt', "w") as text_file:
    text_file.write(meme.to_string())
+------------------------------------------------------------------------------
+------------------------------------------------------------------------------
"""

__author__ = "Sam Scholten"

import quandl                         # API to access data
import numpy as np                    # numeric arrays
import pandas as pd                   # dataframes
from pathlib import Path              # access to path tools (i.e. cwd)
import pickle                         # save python objects
from prettytable import PrettyTable   # print a nice 2D table (ASCII)
import matplotlib.pyplot as plt       # plotting
import click                          # command line integration

import correlation_tools as corrt


###############################################################################


class DataSet(object):
    """ DataSet holding everything - the Quandl sets, TimeSeries, and all
        analysed Dataframe/PTable data. Also holds the methods to get all
        that juicy info """
    def_data_dir = str(Path.cwd()) + '/' + 'data'
    def_quandl_dir = str(Path.cwd()) + '/' + 'quandl'
    def_img_dir = str(Path.cwd()) + '/' + 'images'
    def_transform = "normalize"

    ###########################################################

    def __init__(self, codes_dict, data_dir=def_data_dir, img_dir=def_img_dir,
                 quandl_dir=def_quandl_dir, freq='annual',
                 transform=def_transform):
        self.qsets = self.tsets = None
        self.codes = codes_dict
        self.freq = freq
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.quandl_dir = quandl_dir
        self.transform = transform

    ###########################################################

    def get_qsets(self):
        """ download and organise the different Quandl data sets """
        self.qsets = []
        with click.progressbar(self.codes.keys(),
                               label='downloading Quandl datasets') as bar:
            for name in bar:
                qset = QuandlSet(name, self.codes[name], freq=self.freq,
                                 transform=self.transform)
                self.qsets.append(qset)

    ###########################################################

    def get_time_series(self):
        """ extract the time series from each qset """
        self.tsets = []
        with click.progressbar(self.qsets,
                               label='extracting time series') as bar:
            for qset in bar:
                qset.extract_cols()
                for ts in qset.time_series:
                    self.tsets.append(ts)

    ###########################################################

    def pickle_time_series(self, filename='time_series'):
        """ pickle the time series data """
        if self.tsets is None:
            raise RuntimeError("Couldn't find time series")
        path = self.data_dir + '/' + filename + '.pickle'
        pickle.dump(self.tsets, open(path, "wb"))

    ###########################################################

    def load_time_series(self, filename='time_series'):
        """ load the pickled time series data """
        path = self.data_dir + '/' + filename + ".pickle"
        self.tsets = pickle.load(open(path, "rb"))

    ###########################################################

    def print_table(self, filename='table.txt', sortby=None,
                    reversesort=False, title=None):
        """ print the table in a .txt file, sorted by the column name given by
            sortby (or no sorting) """
        with open(self.quandl_dir + '/' + filename, "w") as text_file:
            text_file.write(self.table.get_string(sortby=sortby,
                            reversesort=reversesort,
                            title=title))

    ###########################################################

    def print_dataframe(self, filename='df.txt'):
        """ save the pandas dataframe as a .txt file (for viewing, not
            loading back into python. Do that with pickle) """
        with open(self.quandl_dir + '/' + filename, "w") as text_file:
            text_file.write(self.dataframe.to_string(index_names=False))

    ###########################################################

    def build_table(self):
        """ build a PTable for the data. Pretty to view in notepad, not
            so useful for any further analysis. """
        self.tsA_lst = []
        self.tsA_qset_lst = []
        self.tsB_lst = []
        self.tsB_qset_lst = []
        self.r_vals = []
        self.offset_lst = []
        self.max_corrs = []

        with click.progressbar(length=len(self.tsets),
                               label="building table") as bar:
            for i, ts1 in enumerate(self.tsets):
                for j, ts2 in enumerate(self.tsets):
                    if j <= i:
                        continue
                    try:
                        ts1_locs, ts2_locs = self.same_times(ts1, ts2)
                    except NoOverlapError:
                        # r_val = np.nan
                        r_val = -np.inf
                    else:
                        try:
                            r_val = self.straight_corr(ts1, ts2, ts1_locs,
                                                       ts2_locs)
                        except corrt.HomogRegionError:
                            # r_val = np.nan
                            r_val = -np.inf

                    offset, max_corr = self.cross_corr(
                                    ts1, ts2, ts1_locs, ts2_locs)
                    self.tsA_lst.append(ts1.name)
                    self.tsA_qset_lst.append(ts1.qset)
                    self.tsB_lst.append(ts2.name)
                    self.tsB_qset_lst.append(ts2.qset)
                    self.r_vals.append(r_val)
                    self.offset_lst.append(offset)
                    self.max_corrs.append(max_corr)
                bar.update(1)

        self.table = PrettyTable()
        self.table.add_column("Time Series A", self.tsA_lst)
        self.table.add_column("A's Quandl Set", self.tsA_qset_lst)
        self.table.add_column("Time Series B", self.tsB_lst)
        self.table.add_column("B's Quandl Set", self.tsB_qset_lst)
        self.table.add_column("Correlation r value", self.r_vals)
        self.table.add_column("Max Cross-Corr Value", self.max_corrs)
        self.table.add_column("Offset (years)", self.offset_lst)

    ###########################################################

    def build_dataframe(self):
        """ build a dataframe out of the data (running correlations for all
            permutations) """
        self.tsA_lst = []
        self.tsA_qset_lst = []
        self.tsB_lst = []
        self.tsB_qset_lst = []
        self.r_vals = []
        self.offset_lst = []
        self.max_corrs = []

        with click.progressbar(length=len(self.tsets),
                               label="building dataframe") as bar:
            for i, ts1 in enumerate(self.tsets):
                for j, ts2 in enumerate(self.tsets):
                    if j <= i:
                        continue
                    try:
                        ts1_locs, ts2_locs = self.same_times(ts1, ts2)
                    except NoOverlapError:
                        # r_val = np.nan
                        r_val = -np.inf
                    else:
                        try:
                            r_val = self.straight_corr(ts1, ts2, ts1_locs,
                                                       ts2_locs)
                        except corrt.HomogRegionError:
                            # r_val = np.nan
                            r_val = -np.inf

                    offset, max_corr = self.cross_corr(
                                    ts1, ts2, ts1_locs, ts2_locs)
                    self.tsA_lst.append(ts1.name)
                    self.tsA_qset_lst.append(ts1.qset)
                    self.tsB_lst.append(ts2.name)
                    self.tsB_qset_lst.append(ts2.qset)
                    self.r_vals.append(r_val)
                    self.offset_lst.append(offset)
                    self.max_corrs.append(max_corr)
                bar.update(1)

        d = {"Time Series A": self.tsA_lst,
             "A's Quandl Set": self.tsA_qset_lst,
             "Time Series B": self.tsB_lst,
             "B's Quandl Set": self.tsB_qset_lst,
             "Correlation r value": self.r_vals,
             "Max Cross-Corr Value": self.max_corrs,
             "Offset (years)": self.offset_lst}
        self.dataframe = pd.DataFrame(data=d)
        self.dataframe = self.dataframe.sort_values(
                        "Correlation r value", ascending=False)

    ###########################################################

    def cross_corr(self, ts1, ts2, ts1_locs, ts2_locs):
        """ handles the cross correlation of two TimeSeries """
        corr_sys = corrt.CorrSystem(self.img_dir, self.data_dir)
        t = corrt.CorrObject()
        t.ar = ts1.data
        sr = corrt.CorrObject()
        sr.ar = ts2.data
        corr_sys.add_template(t)
        corr_sys.add_search_region(sr)
        try:
            corr_sys.cross_correlate()
        except corrt.SizeError:
            # swap search region and template (no difference)
            corr_sys.add_template(sr)
            corr_sys.add_search_region(t)
            corr_sys.cross_correlate()

        max_corr = np.amax(corr_sys.corr)
        max_corr_loc = np.argwhere(corr_sys.corr == max_corr)[0][0]
        if corr_sys.method == 'spectral':
            offset = abs(max_corr_loc)
        else:
            offset = abs(max_corr_loc - corr_sys.template.size)
        return offset, max_corr

    ###########################################################

    def straight_corr(self, ts1, ts2, ts1_locs, ts2_locs):
        """ handles the correlation (NOT cross correlation) of two
            TimeSeries """
        t_loc1, t_max_loc1 = ts1_locs
        t_loc2, t_max_loc2 = ts2_locs

        # if it's a tiny region return a nan
        if t_max_loc1 - t_loc1 < 5 or t_max_loc2 - t_loc2 < 5:
            # return np.nan
            return -np.inf

        # ok now we can cut out that time range and straight up correlate
        corr_sys = corrt.CorrSystem(self.img_dir, self.data_dir)
        t = ts1.data[t_loc1: t_max_loc1+1]
        A = ts2.data[t_loc2: t_max_loc2+1]
        r_val = corr_sys.norm_spatial_corr(t, A)
        return r_val

    ###########################################################

    def plot_ts(self, im_name, ts_name_lst):
        """ plot a set of TimeSeries (names of which are in ts_name_lst) on
            the same axis for comparison """
        fig, ax = plt.subplots()
        ax.set_xlabel("year")
        ax.set_ylabel("value (mean-shifted, normalised)")
        ax.set_title("Comparison of different time series")

        all_t = []
        # plot all together
        for ts in self.tsets:
            if ts.name in ts_name_lst:
                y_vals = ts.data
                x_vals = ts.time
                for x in x_vals:
                    if x not in all_t:
                        all_t.append(x)
                ax.plot(x_vals, y_vals, label=ts.name)

        ax.set_xticks(np.arange(min(all_t), max(all_t)+5, 5))

        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels)
        ax.legend(handles, labels, loc=2)

        fig.savefig(self.img_dir + '/' + im_name + '.png', dpi=600,
                    bbox_inches='tight')

    ###########################################################

    def same_times(self, ts1, ts2):
        """ determine what time range the TimeSeries have in common """
        if ts1.time[0] <= ts2.time[0]:
            t_loc2 = 0
            t_loc1 = None
            for i, t in enumerate(ts1.time):
                if t == ts2.time[0]:
                    t_loc1 = i
                    break
            if t_loc1 is None:
                raise NoOverlapError
        else:
            t_loc1 = 0
            t_loc2 = None
            for i, t in enumerate(ts2.time):
                if t == ts1.time[0]:
                    t_loc2 = i
                    break
            if t_loc2 is None:
                raise NoOverlapError
        # ok now iterate through to max t backwards
        if ts1.time[-1] >= ts2.time[-1]:
            t_max_loc2 = len(ts2.time)
            t_max_loc1 = None
            for i, t in reversed(list(enumerate(ts1.time))):
                if t == ts2.time[-1]:
                    t_max_loc1 = i
                    break
            if t_max_loc1 is None:
                raise NoOverlapError
        else:
            t_max_loc1 = len(ts1.time)
            t_max_loc2 = None
            for i, t in reversed(list(enumerate(ts2.time))):
                if t == ts1.time[-1]:
                    t_max_loc2 = i
                    break
            if t_max_loc2 is None:
                raise NoOverlapError
        return (t_loc1, t_max_loc1), (t_loc2, t_max_loc2)

###############################################################################


class TimeSeries(object):
    """ holds the data and other attributes of a quandl time series """
    def __init__(self, qset, name, data, time):
        """
        qset: name of the quandl dataset this timeseries is in
        name: name of this time series (i.e. column of qset)
        data: the numpy array of the data
        time: the time indexes for data
        """
        self.qset = qset
        self.name = name
        self.data = data
        self.time = time

###############################################################################


class QuandlSet(object):
    """ holds all of the time series and other attributes of a quandl data
        set """
    def __init__(self, name, code, freq, transform=None):
        self.name = name
        self.code = code
        self.freq = freq
        self.extracted = False
        self.time_series = None
        self.transform = transform
        self.extract()

    ###########################################################

    def extract(self):
        """
        so this extracts the data, columns and times
        """

        self.data = quandl.get(self.code,
                               collapse=self.freq,
                               transform=self.transform
                               )
        self.cols = list(self.data.columns.values)
        # get the years (the indices of the dataframe), as strings
        indices = np.array(self.data.index, dtype='S')
        # just take the year (4-num code)
        times_lst = []
        for t in indices:
            times_lst.append(t[:4])
        self.time = np.array(times_lst)

        # add Nones to missing years (will be extrapolated out later)
        int_time = [int(t) for t in self.time]
        data_lst = list(self.data.values)

        # make an array of None objects to insert for missing years
        lst = []
        for i in range(len(self.cols)):
            lst.append(None)
        row_of_Nones = np.array(None)

        i = 1
        while True:
            if i >= len(int_time):
                break
            expected_time = int_time[i-1] + 1
            if int_time[i] != expected_time:
                int_time.insert(i, expected_time)
                data_lst.insert(i, row_of_Nones)
            i += 1
        self.extracted_data = np.array(data_lst)

        self.time = np.array(int_time)
        self.extracted = True

    ###########################################################

    def extract_cols(self):
        """ extract the columns from each column in the Quandl data set,
            and creates a TimeSeries out of each. First has to handle nan or
            None data types. """
        if not self.extracted:
            self.extract()

        self.time_series = []
        cols_to_remove = []
        for i in range(len(self.cols)):
            # check for Nones in the dataset of this column

            ind_to_keep = np.ones(len(self.extracted_data[:, i]))
            for j, val in enumerate(self.extracted_data[:, i]):
                if val is None or np.isnan(val):
                    ind_to_keep[j] = 0      # don't keep this one

            # if they're all Nones, remove that column (we don't want it!)
            if not np.any(ind_to_keep) or \
                    len(ind_to_keep[ind_to_keep == 1]) <= 1:
                cols_to_remove.append(i)
            else:
                # otherwise just keep the indices we want from data and time
                good_data, good_times = handle_nulls(self.extracted_data[:, i],
                                                     self.time,
                                                     ind_to_keep)
                # ALSO mean shift, normalise data
                good_data = good_data - np.mean(good_data)
                good_data /= np.max(good_data)
                ts = TimeSeries(self.name,
                                self.cols[i],
                                good_data,
                                good_times)
                self.time_series.append(ts)
        self.cols = [el for i, el in enumerate(self.cols)
                     if i not in cols_to_remove]

###############################################################################
# helper fn


class NoOverlapError(Exception):
    pass


def handle_nulls(data, times, ind_to_keep):
    """ remove nan and Nones from start and end, interpolate over interstitials
    """
    dl = list(data)
    tl = list(times)
    linds = list(ind_to_keep)

    to_remove = []
    # first handle nones at the start
    if not linds[0]:
        for i_index, i in enumerate(linds):
            if i:
                break
            else:
                to_remove.append(i_index)
    # now at the end
    if not linds[-1]:
        for i_index, i in reversed(list(enumerate(linds))):
            if i:
                break
            else:
                to_remove.append(i_index)
    # remove those indices
    dl = [el for i, el in enumerate(dl) if i not in to_remove]
    tl = [el for i, el in enumerate(tl) if i not in to_remove]
    linds = [el for i, el in enumerate(linds) if i not in to_remove]

    # true if there are other Nones
    if any(el == 0 for el in linds):
        # now linearly interpolate for interstitial Nones
        last_good = linds[0]
        num_nan = 0
        for j, _ in enumerate(linds):
            # if it's a good value
            if linds[j]:
                # check if we need to interpolate something
                if num_nan:
                    # ok now go back and interpolate (neirest neigh linear)
                    m = (dl[j] - last_good)/(num_nan + 1)
                    c = last_good
                    # go back and plug in interpolated values
                    for s in range(1, num_nan+1):
                        f = linear(num_nan+1-s, m, c)
                        dl[j-s] = f

                # then reset our values
                last_good = dl[j]
                num_nan = 0
            else:
                num_nan += 1
    return np.array(dl), np.array(tl)

###############################################################################


def linear(x, m, c):
    """ linear function in x, y = m*x + c """
    return m*x + c
