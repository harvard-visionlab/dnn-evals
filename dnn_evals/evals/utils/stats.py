import numpy as np
from functools import partial
from fastprogress import progress_bar 

class AccumMetric:
    def __init__(self, scoring_func, ci_level=0.95):
        self.scores = []
        self.scoring_func = scoring_func
        self.ci_level = ci_level    

    def reset(self):
        self.scores = []

    def stats(self, ci_level=None, axis=None):
        if not self.scores:
            return None
        ci_level = self.ci_level if ci_level is None else ci_level
        axis = 0 if axis is None else axis
        
        # Calculate the mean
        mean_score = np.mean(self.scores, axis=axis)

        # Calculate the lower and upper percentiles for the confidence interval
        lower_percentile = 100 * (1 - self.ci_level) / 2
        upper_percentile = 100 * (1 + self.ci_level) / 2

        lower_ci = np.percentile(self.scores, lower_percentile, axis=axis)
        upper_ci = np.percentile(self.scores, upper_percentile, axis=axis)

        return {
            "mean": mean_score,
            f"{int(self.ci_level * 100)}% CI": (lower_ci, upper_ci),
        }
    
    def __call__(self, data):
        # Calculate the score using the provided scoring function
        score = self.scoring_func(data)
        self.scores.append(score)
        
def estimate_thresh_crossing(xs, ys, threshold):
    crossings = []

    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        y1, y2 = ys[i], ys[i + 1]

        if (y1 < threshold and y2 >= threshold) or (y1 >= threshold and y2 < threshold):
            # Linear interpolation to estimate the crossing point
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            crossing_x = (threshold - intercept) / slope
            crossings.append(crossing_x)

    if crossings:
        # Calculate the average crossing point if multiple crossings occurred
        estimated_epsilon = np.mean(crossings)
    else:
        estimated_epsilon = None

    return estimated_epsilon, crossings        