"""sui.ds
Tool kits for data science
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__api_info_dict = {}


def api_info(api=None):
    if api is not None:
        if api in __api_info_dict:
            return 'API: {}\nInfo: {}\n'.format(api, __api_info_dict[api])
        else:
            return '{} is not an appreciable API under sui.ds'.format(api)
    else:
        for api, info in __api_info_dict.items():
            return 'API: {}\nInfo: {}\n'.format(api, info)
