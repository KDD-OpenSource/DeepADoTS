import numpy as np

from .synthetic_multivariate_dataset import SyntheticMultivariateDataset


class MultivariateAnomalyFunction:
    # ----- Functions generating the anomalous dimension --------- #
    # A MultivariateAnomalyFunction should return a tuple containing the following three values:
    # * The values of the second dimension (array of max `interval_length` numbers)
    # * Starting point for the anomaly
    # * End point for the anomaly section
    # The last two values are ignored for generation of not anomalous data

    # Get a dataset by passing the method name as string. All following parameters
    # are passed through. Throws AttributeError if attribute was not found.
    @staticmethod
    def get_multivariate_dataset(method, name=None, group_size=None, *args, **kwargs):
        name = name or f'Synthetic Multivariate {method} Curve Outliers'
        func = getattr(MultivariateAnomalyFunction, method)
        return SyntheticMultivariateDataset(anomaly_func=func,
                                            name=name,
                                            group_size=group_size,
                                            *args,
                                            **kwargs)

    @staticmethod
    def doubled(curve_values, anomalous, _):
        factor = 4 if anomalous else 2
        return curve_values * factor, 0, len(curve_values)

    @staticmethod
    def inversed(curve_values, anomalous, _):
        factor = -2 if anomalous else 2
        return curve_values * factor, 0, len(curve_values)

    @staticmethod
    def shrinked(curve_values, anomalous, _):
        if not anomalous:
            return curve_values, -1, -1
        else:
            new_curve = curve_values[::2]
            nonce = np.zeros(len(curve_values) - len(new_curve))
            values = np.concatenate([nonce, new_curve])
            return values, 0, len(values)

    @staticmethod
    def xor(curve_values, anomalous, interval_length):
        pause_length = interval_length - len(curve_values)
        if not anomalous:
            # No curve during the other curve in the 1st dimension
            nonce = np.zeros(len(curve_values))
            # Insert a curve with the same amplitude during the pause of the 1st dimension
            new_curve = MultivariateAnomalyFunction.shrink_curve(curve_values, pause_length)
            return np.concatenate([nonce, new_curve]), -1, -1
        else:
            # Anomaly: curves overlap (at the same time or at least half overlapping)
            max_pause = min(len(curve_values) // 2, pause_length)
            nonce = np.zeros(max_pause)
            return np.concatenate([nonce, curve_values]), len(nonce), len(curve_values)

    @staticmethod
    def delayed(curve_values, anomalous, interval_length):
        if not anomalous:
            return curve_values, -1, -1
        else:
            # The curve in the second dimension occurs a few timestamps later
            left_space = interval_length - len(curve_values)
            delay = min(len(curve_values) // 2, left_space)
            nonce = np.zeros(delay)
            values = np.concatenate([nonce, curve_values])
            return values, 0, len(values)

    @staticmethod
    def delayed_missing(curve_values, anomalous, interval_length):
        starting_point = len(curve_values) // 5
        # If the space is too small for the normal curve we're shrinking it (which is not anomalous)
        left_space = interval_length - starting_point
        new_curve_length = min(left_space, len(curve_values))
        if not anomalous:
            # The curve in the second dimension occurs a few timestamps later
            nonce = np.zeros(starting_point)
            new_curve = MultivariateAnomalyFunction.shrink_curve(curve_values, new_curve_length)
            values = np.concatenate([nonce, new_curve])
            return values, -1, -1
        else:
            end_point = starting_point + new_curve_length
            nonce = np.zeros(end_point)
            return nonce, starting_point, end_point

    """
        This is a helper function for shrinking an already generated curve.
    """

    @staticmethod
    def shrink_curve(curve_values, new_length):
        if new_length == len(curve_values):
            return curve_values
        orig_amplitude = max(abs(curve_values))
        orig_amplitude *= np.sign(curve_values.mean())
        return SyntheticMultivariateDataset.get_curve(new_length, orig_amplitude)
