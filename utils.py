import numpy as np
from math import ceil, floor

class TinyStatistician :
    def __add_fractional_to_base(self, array, p) :
        array = sorted(array)
        rank = (p * (len(array) - 1)) / 100
        integer_lower_rank = floor(rank)
        integer_higher_rank = ceil(rank)
        fractional_rank = rank - integer_lower_rank
        if integer_lower_rank == rank :
            return array[integer_lower_rank]
        return float(array[integer_lower_rank] + ((array[integer_higher_rank] - array[integer_lower_rank]) * fractional_rank))

    def mean(self, array) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        result = 0
        for _, value in enumerate(array) :
            result += value
        return float(result / len(array))

    def median(self, array) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        return float(self.__add_fractional_to_base(array, 50))

    def quartile(self, array) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        array.sort()
        return [float(self.__add_fractional_to_base(array, 25)), float(self.__add_fractional_to_base(array, 75))]

    def percentile(self, array, p) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        return float(self.__add_fractional_to_base(array, p))
 
    def var(self, array) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        mean = float(self.mean(array))
        total_sum = 0
        for value in array :
            total_sum += (float(value) - mean) ** 2
        return float(total_sum / len(array))
    
    def std(self, array) :
        if type(array).__module__ != np.__name__ :
                return None
        if array.size <= 0 :
            return None
        array = np.squeeze(array).astype(float) if array.shape != (1, 1) else array.flatten().astype(float)
        return float(self.var(array) ** (1 / 2))