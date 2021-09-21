import numpy as np
from sklearn.linear_model import LinearRegression
from abc import ABCMeta, abstractmethod
from collections import namedtuple

PREDICTION_TYPE = namedtuple(
    "Predictions",
    [
        "model_prediction",
        "true_label"
    ]
)


class ComparisonModel(metaclass=ABCMeta):
    def __init__(self):
        self._predictions = []
        self._true_labels = []
        self._trained = False

    @abstractmethod
    def train(self, labels_series):
        pass

    @abstractmethod
    def predict_label(self, index):
        pass


class NullModelPredictor(ComparisonModel):
    def __init__(self):
        super(NullModelPredictor, self).__init__()
        self._predictions = [None]
        self._true_labels = []

    def predict_label(self, index):
        assert (self._trained)
        assert (index > 0)
        return PREDICTION_TYPE(self._predictions[index], self._true_labels[index])


class NullModel(NullModelPredictor):
    def __init__(self):
        super(NullModel, self).__init__()

    def train(self, labels_series):
        assert (
            (len(self._predictions) == 1) and
            (self._predictions[0] is None) and
            (not self._true_labels)
        )
        self._true_labels.append(labels_series[0])
        for idx, l in enumerate(labels_series[:-1]):
            self._true_labels.append(labels_series[idx + 1])
            self._predictions.append(l)
        self._trained = True


class NullDiffModel(NullModelPredictor):
    def __init__(self):
        super(NullDiffModel, self).__init__()
        self._true_labels = [None]

    def train(self, labels_series):
        assert (
            (len(self._predictions) == 1) and
            (self._predictions[0] is None) and
            (len(self._true_labels) == 1) and
            (self._true_labels[0] is None)
        )
        for idx, l in enumerate(labels_series[:-1]):
            self._true_labels.append(labels_series[idx + 1]-l)
            self._predictions.append(np.zeros(len(l)))
        self._trained = True


class FirstOrderPredictor(ComparisonModel):
    def __init__(self):
        super(FirstOrderPredictor, self).__init__()
        self._lin_reg_models = []

    def predict_label(self, index):
        assert (self._trained)
        assert (index > 0)
        pred = [m.predict(np.array([[index]])) for m in self._lin_reg_models]
        return PREDICTION_TYPE(pred, self._true_labels[index])


class FirstOrderModel(FirstOrderPredictor):
    def __init__(self):
        super(FirstOrderModel, self).__init__()

    def train(self, labels_series):
        self._time_steps = np.arange(len(labels_series)-1).reshape(-1, 1)
        for idx, n in enumerate(labels_series.T):
            self._lin_reg_models.append(
                LinearRegression().fit(self._time_steps, n[:-1])
            )
        for idx, l in enumerate(labels_series):
            self._true_labels.append(l)
        self._trained = True


class FirstOrderDiffModel(FirstOrderPredictor):
    def __init__(self):
        super(FirstOrderModel, self).__init__()
        self._true_labels = [None]

    def train(self, labels_series):
        self._time_steps = np.arange(len(labels_series)-1).reshape(-1, 1)
        for idx, n in enumerate(labels_series.T):
            diffs = []
            for idx, m in enumerate(n[:-1]):
                diffs.append(n[idx+1]-m)
            self._lin_reg_models.append(
                LinearRegression().fit(self._time_steps, diffs)
            )
        for idx, l in enumerate(labels_series[:-1]):
            self._true_labels.append(labels_series[idx + 1]-l)
        self._trained = True


class AveragePredictor(ComparisonModel):
    def __init__(self, average_time):
        super(AveragePredictor, self).__init__()
        self._average_time = average_time
        self._averages = [None] * self._average_time
        self._weights = None
        self._total_weight = 0

    def train(self, labels_series):
        for t in range(self._average_time, labels_series.shape[0]):
            self._averages.append(
                np.sum(
                    np.multiply(
                        self._weights,
                        labels_series[t-self._average_time:t].T
                    ),
                    axis=1
                ) / self._total_weight
            )
        for idx, l in enumerate(labels_series):
            self._true_labels.append(l)
        self._trained = True

    def predict_label(self, index):
        assert (self._trained)
        assert (self._average_time <= index)
        return PREDICTION_TYPE(self._averages[index], self._true_labels[index])


# As defined in https://www.cl.cam.ac.uk/~rja14/Papers/computernets12.pdf under 5.1 Uniform Average Centrality
class UniformAverageModel(AveragePredictor):
    def __init__(self, average_time):
        super(UniformAverageModel, self).__init__(average_time)
        self._weights = [1/self._average_time] * self._average_time
        self._total_weight = np.sum(self._weights)


# As defined in https://www.cl.cam.ac.uk/~rja14/Papers/computernets12.pdf under 5.1 Weighted Average Centrality
# in our case l = 1, m = 1, and w is normalized to be 1
class LinearWeightedAverageModel(AveragePredictor):
    def __init__(self, average_time):
        super(LinearWeightedAverageModel, self).__init__(average_time)
        self._weights = [1/i for i in range(self._average_time, 0, -1)]
        self._total_weight = np.sum(self._weights)


# As defined in https://www.cl.cam.ac.uk/~rja14/Papers/computernets12.pdf under 5.1 Weighted Average Centrality
# in our case l = 1, m = 1, and w is normalized to be 1
class SquareRootWeightedAverageModel(AveragePredictor):
    def __init__(self, average_time):
        super(SquareRootWeightedAverageModel, self).__init__(average_time)
        self._weights = [
            1/np.sqrt(i) for i in range(self._average_time, 0, -1)
        ]
        self._total_weight = np.sum(self._weights)


# As defined in https://www.cl.cam.ac.uk/~rja14/Papers/computernets12.pdf under 5.1 Polynomial Regression
# in our case l = 1, m = 1, and w is normalized to be 1, though we support degree > m, since the degree used in the paper is 3.
class PolynomialRegressionModel(ComparisonModel):
    def __init__(self, average_time, degree, epsilon):
        super(PolynomialRegressionModel, self).__init__()
        self._average_time = average_time
        self._degree = degree
        self._epsilon = epsilon
        self._poly_coefficients = []
        self._polynomials = [None] * self._average_time

    def train(self, labels_series):
        self._time_steps = np.arange(len(labels_series)-1)
        for t in range(self._average_time, labels_series.shape[0]):
            self._poly_coefficients.append(
                np.polyfit(
                    self._time_steps[:self._average_time],
                    labels_series[t-self._average_time:t, :],
                    self._degree
                )
            )
            self._polynomials.append(
                [np.poly1d(p)for p in self._poly_coefficients[-1].T]
            )
        for idx, l in enumerate(labels_series):
            self._true_labels.append(l)
        self._trained = True

    def predict_label(self, index):
        assert(self._trained)
        assert(self._average_time <= index)
        return PREDICTION_TYPE(
            np.array(
                [
                    p(self._time_steps[-1] + 1 - self._epsilon)
                    for p in self._polynomials[index]
                ]
            ),
            self._true_labels[index]
        )


# As defined in https://www.cl.cam.ac.uk/~rja14/Papers/computernets12.pdf under 5.1 Periodic Intervals
# in our case l = 1, m = 1, and w is normalized to be 1
class PeriodicAveragePredictor(AveragePredictor):
    def __init__(self, average_time, period, epsilon):
        super(PeriodicAveragePredictor, self).__init__(average_time)
        self._period = period
        self._epsilon = epsilon
        self._d_period = [
            i % self._period for i in range(1, self._average_time + 1)
        ]


# As defined in https://www.cl.cam.ac.uk/~rja14/Papers/computernets12.pdf under 5.1 Periodic Intervals
# in our case l = 1, m = 1, and w is normalized to be 1
class UniformPeriodicAverageModel(PeriodicAveragePredictor):
    def __init__(self, average_time, period, epsilon):
        super(UniformPeriodicAverageModel, self).__init__(
            average_time,
            period,
            epsilon
        )
        self._weights = [
            1 / (
                np.abs(
                    np.arcsin(
                        np.cos(
                            (np.pi * d_period + self._epsilon) /
                            self._period
                        )
                    )
                )
            )
            for d_period in self._d_period
        ]
        self._total_weight = np.sum(self._weights)


# As defined in https://www.cl.cam.ac.uk/~rja14/Papers/computernets12.pdf under 5.1 Periodic Intervals
# in our case l = 1, m = 1, and w is normalized to be 1
class WeightedPeriodicAverageModel(PeriodicAveragePredictor):
    def __init__(self, average_time, period, epsilon):
        super(WeightedPeriodicAverageModel, self).__init__(
            average_time,
            period,
            epsilon
        )
        self._weights = [
            1 / (
                np.abs(
                    np.arcsin(
                        np.cos(
                            (np.pi * d_period + self._epsilon) /
                            self._period
                        )
                    )
                ) * d
            )
            for d_period, d in zip(self._d_period, range(1, self._average_time + 1))
        ]
        self._total_weight = np.sum(self._weights)
