import os
from random import shuffle

import numpy as np
from sklearn.model_selection import train_test_split
class Processor:
    DATA_DIRECTORY = "data"

    @classmethod
    def working_directory(cls):
        """
        get current directory
        """
        return os.path.join(os.getcwd(), cls.DATA_DIRECTORY)

    @classmethod
    def read_file_lines(cls, dataset, filename):
        """
        read all lines of file with file name, not full path
        """
        filepath = os.path.join(
            cls.working_directory(), dataset, filename)
        with open(filepath, 'r', encoding='utf-8') as content:
            return content.readlines()


    @classmethod
    def extract_features(cls, a_line):
        """
        extract features based on comma (,), return an np.array
        """
        return [x.strip() for x in a_line.split(',')]


    @classmethod
    def numericalize_feature(cls, feature, protocol_type, service, flag):

        protocol_type_count = len(protocol_type)
        service_count = len(service)
        flag_count = len(flag)

        second_index = int(protocol_type_count + 1)
        third_index = int(protocol_type_count + service_count + 1)
        forth_index = int(protocol_type_count + service_count + flag_count + 1)

        # index 1 is protocol_type
        feature[1:1] = protocol_type[feature[1]]
        feature.pop(second_index)

        # index 2 + protocol_type_count is service
        feature[second_index:second_index] = service[feature[second_index]]
        feature.pop(third_index)
        # # index 3 + protocol_type_count + service_count is flag
        feature[third_index:third_index] = flag[feature[third_index]]
        feature.pop(forth_index)

        # make all values np.float64
        feature = [np.float64(x) for x in feature]


        return np.array(feature)

    @classmethod
    def numericalize_result(cls, reslut, attack, attack_dict):
        second_index = int(1)
        # index 0 is attack
        reslut[0:0] = attack[attack_dict[reslut[0]]]
        reslut.pop(second_index)
        # make all values np.float64
        reslut = [np.float64(x) for x in reslut]

        return np.array(reslut)
    @classmethod
    def normalize_value(cls, value, min, max):
        value = np.float64(value)
        min = np.float64(min)
        max = np.float64(max)

        if min == np.float64(0) and max == np.float64(0):
            return np.float64(0)
        result = np.float64((value - min) / (max - min))
        return result
