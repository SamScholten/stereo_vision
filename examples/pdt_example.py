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
