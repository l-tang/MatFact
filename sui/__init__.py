from __future__ import absolute_import, division, print_function, unicode_literals
from .ml import FunkSVD, BiasSVD, SVDpp, BPR, ALS
from .dl.rank import PNN
from .dl.recall import GRU4Rec

__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.2.0'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'

my_dear = "Dear Miss Sui Lin, I love you!"

__api_info_dict = {
    "sui.FunkSVD": "FunkSVD",
    "sui.BiasSVD": "BiasSVD",
    "sui.SVDpp": "SVD++",
    "sui.BPR": "Bayesian Personalized Ranking",
    "sui.ALS": "Alternating Least Squares",
    "sui.GRU4Rec": "GRU4Rec",
    "sui.PNN": "Product-based Neural Networks",
    "sui.toolbox.top_k": "Return a list containing top k data for a specific dimension"
}


def api_info(api=None):
    if api is not None:
        if api in __api_info_dict:
            print('API: {}\nInfo: {}\n'.format(api, __api_info_dict[api]))
        else:
            print('{} is not a correct API.'.format(api))
    else:
        for api, info in __api_info_dict.items():
            print('API: {}\nInfo: {}\n'.format(api, info))
