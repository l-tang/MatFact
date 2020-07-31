"""
    Module for data visualization
    Date: 25/May/2020
    Author: Li Tang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.1.6'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'


class SuiDsVisualizationError(Exception):
    pass


def scatter(df, x, y, colour, title=None, x_label=None, y_label=None, shuffle=True, scaler=MinMaxScaler(),
            save_path=None):

    if scaler:
        df = pd.DataFrame(scaler.fit_transform(df[[x, y]].values), columns=[x, y])

    if shuffle:
        df = df.sample(frac=1)

    sc = plt.scatter(df[x], df[y], c=colour)
    plt.xlabel(x if x_label is None else x_label)
    plt.ylabel(y if y_label is None else y_label)
    plt.colorbar(sc)
    plt.subplots_adjust(top=2, bottom=1, right=2, left=1, hspace=0, wspace=0)
    if title is not None:
        plt.title(title, fontsize=14)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.show()
