import warnings
from abc import ABC, abstractmethod
from typing import Tuple, Iterable, Callable, Any

import numpy as np
from sklearn.utils import column_or_1d, assert_all_finite, check_consistent_length
from sklearn.metrics import auc, roc_curve, precision_recall_curve

# ----------------------------------------- parent Classes 


# To transform predictions to one value per timepoint 
class ADMetric(ABC):
    """Base class for metric implementations that score anomaly scorings against ground truth binary labels. Every
    subclass must implement :func:`~timeeval.metrics.Metric.name`, :func:`~timeeval.metrics.Metric.score`, and
    :func:`~timeeval.metrics.Metric.supports_continuous_scorings`.
    """

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:  # type: ignore[no-untyped-def]
        y_true, y_score = self._validate_scores(y_true, y_score, **kwargs)
        if np.unique(y_score).shape[0] == 1:
            warnings.warn("Cannot compute metric for a constant value in y_score, returning 0.0!")
            return 0.
        return self.score(y_true, y_score)

    def _validate_scores(self, y_true: np.ndarray, y_score: np.ndarray,
                         inf_is_1: bool = True,
                         neginf_is_0: bool = True,
                         nan_is_0: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        y_true = np.array(y_true).copy()
        y_score = np.array(y_score).copy()
        # check labels
        if self.supports_continuous_scorings() and y_true.dtype == np.float_ and y_score.dtype == np.int_:
            warnings.warn("Assuming that y_true and y_score where permuted, because their dtypes indicate so. "
                          "y_true should be an integer array and y_score a float array!")
            return self._validate_scores(y_score, y_true)

        y_true: np.ndarray = column_or_1d(y_true)  # type: ignore
        assert_all_finite(y_true)

        # check scores
        y_score: np.ndarray = column_or_1d(y_score)  # type: ignore

        check_consistent_length([y_true, y_score])
        if not self.supports_continuous_scorings():
            if y_score.dtype not in [np.int_, np.bool_]:
                raise ValueError("When using Metrics other than AUC-metric that need discrete (0 or 1) scores (like "
                                 "Precision, Recall or F1-Score), the scores must be integers and should only contain "
                                 "the values {0, 1}. Please consider applying a threshold to the scores!")
        else:
            if y_score.dtype != np.float_:
                raise ValueError("When using continuous scoring metrics, the scores must be floats!")

        # substitute NaNs and Infs
        nan_mask = np.isnan(y_score)
        inf_mask = np.isinf(y_score)
        neginf_mask = np.isneginf(y_score)
        penalize_mask = np.full_like(y_score, dtype=bool, fill_value=False)
        if inf_is_1:
            y_score[inf_mask] = 1
        else:
            penalize_mask = penalize_mask | inf_mask
        if neginf_is_0:
            y_score[neginf_mask] = 0
        else:
            penalize_mask = penalize_mask | neginf_mask
        if nan_is_0:
            y_score[nan_mask] = 0.
        else:
            penalize_mask = penalize_mask | nan_mask
        y_score[penalize_mask] = (~np.array(y_true[penalize_mask], dtype=bool)).astype(np.int_)

        assert_all_finite(y_score)
        return y_true, y_score

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of this metric."""
        ...

    @abstractmethod
    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Implementation of the metric's scoring function.

        Please use :func:`~timeeval.metrics.Metric.__call__` instead of calling this function directly!

        Examples
        --------

        Instantiate a metric and call it using the ``__call__`` method:

        >>> import numpy as np
        >>> from timeeval.metrics import RocAUC
        >>> metric = RocAUC(plot=False)
        >>> metric(np.array([0, 1, 1, 0]), np.array([0.1, 0.4, 0.35, 0.8]))
        0.5

        """
        ...

    @abstractmethod
    def supports_continuous_scorings(self) -> bool:
        """Whether this metric accepts continuous anomaly scorings as input (``True``) or binary classification
        labels (``False``)."""
        ...
        

class AucMetric(ADMetric, ABC):
    """Base class for area-under-curve-based metrics.

    All AUC-Metrics support continuous scorings, calculate the area under a curve function, and allow plotting this
    curve function. See the subclasses' documentation for a detailed explanation of the corresponding curve and metric.
    """
    def __init__(self, plot: bool = False, plot_store: bool = False) -> None:
        self._plot = plot
        self._plot_store = plot_store

    def _auc(self,
             y_true: np.ndarray,
             y_score: Iterable[float],
             curve_function: Callable[[np.ndarray, np.ndarray], Any]) -> float:
        x, y, thresholds = curve_function(y_true, np.array(y_score))
        if "precision_recall" in curve_function.__name__:
            # swap x and y
            x, y = y, x
        area: float = auc(x, y)
        if self._plot:
            import matplotlib.pyplot as plt

            name = curve_function.__name__
            plt.plot(x, y, label=name, drawstyle="steps-post")
            # plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
            plt.title(f"{name} | area = {area:.4f}")
            if self._plot_store:
                plt.savefig(f"fig-{name}.pdf")
            plt.show()
        return area

    def supports_continuous_scorings(self) -> bool:
        return True


# Area Under the curve!
class RocAUC(AucMetric):
    """Computes the area under the receiver operating characteristic curve.

    Parameters
    ----------
    plot : bool
        Set this parameter to ``True`` to plot the curve.
    plot_store : bool
        If this parameter is ``True`` the curve plot will be saved in the current working directory under the name
        template "fig-{metric-name}.pdf".

    """
    def __init__(self, plot: bool = False, plot_store: bool = False) -> None:
        super().__init__(plot, plot_store)

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self._auc(y_true, y_score, roc_curve)

    @property
    def name(self) -> str:
        return "ROC_AUC"


class PrAUC(AucMetric):
    """Computes the area under the precision recall curve.

    Parameters
    ----------
    plot : bool
        Set this parameter to ``True`` to plot the curve.
    plot_store : bool
        If this parameter is ``True`` the curve plot will be saved in the current working directory under the name
        template "fig-{metric-name}.pdf".
    """
    def __init__(self, plot: bool = False, plot_store: bool = False) -> None:
        super().__init__(plot, plot_store)

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self._auc(y_true, y_score, precision_recall_curve)

    @property
    def name(self) -> str:
        return "PR_AUC"
