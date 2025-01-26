# Reimport necessary libraries and reinitialize everything after environment reset
from calendar import c
from datetime import datetime
import dis
from operator import ne, neg
import os
import random
import re
import stat
import time
from turtle import st
from typing import final
from matplotlib.pylab import f
import numpy as np
import matplotlib.pyplot as plt
import shutil
import scipy as sp
from scipy import stats
from tqdm import tqdm


# cwd
cwd = os.getcwd()
# Set up output directory
output_dir = os.path.join(cwd, "resource_graphs")
os.makedirs(output_dir, exist_ok=True)

# Define graph generation data
combinations = [
    (0, 0, 0, 0),
    (3, 5, 0, 1),
    (1, 2, 0, 0),
    (15, 20, 5, 10),
    (50, 40, 35, 70),
    (4, 7, 0, 1),
    (18, 25, 5, 11),
    (53, 45, 35, 71),
    (16, 22, 5, 10),
    (51, 42, 35, 70),
    (65, 60, 40, 80),
    (19, 27, 5, 11),
    (54, 47, 35, 71),
    (68, 65, 40, 81),
    (66, 62, 40, 80),
    (69, 67, 40, 81),
]


def calculate_alternating_points(data):
    import numpy as np

    # Calculate mean
    mean = np.mean(data)

    # Count how many points properly alternate
    alternating_count = 0
    for i in range(1, len(data) - 1):
        # Check if current point is on opposite side of mean compared to neighbors
        curr_above_mean = data[i] > mean
        prev_above_mean = data[i - 1] > mean
        next_above_mean = data[i + 1] > mean

        if curr_above_mean != prev_above_mean and curr_above_mean != next_above_mean:
            alternating_count += 1

    return alternating_count


def check_alternating_pattern(data):
    import numpy as np

    mean = np.mean(data)
    threshold = np.std(data) * 0.25  # Reduced threshold to be more sensitive

    # Check each point's distance from mean and relationship to neighbors
    for i in range(1, len(data) - 1):
        curr_dist = abs(data[i] - mean)
        if curr_dist < threshold:
            return False  # Point too close to mean

        if (data[i] > mean and data[i - 1] > mean and data[i + 1] > mean) or (
            data[i] < mean and data[i - 1] < mean and data[i + 1] < mean
        ):
            return False  # Three consecutive points on same side

    return True


class SpikinessFixer:
    def __init__(self, final_value, resource):
        self.final_value = final_value
        self.resource = resource
        if self.resource == "Disk":
            # Disk usage typically maintains 75%+ baseline with occasional dips to 25%
            # Gradual upward trend with occasional spikes
            self.thresholds = {
                "std": 0.15 * final_value,  # Less variation
                "range": 0.25 * final_value,  # Smaller range
                "avg_diff": 0.1 * final_value,  # Smoother changes
                "min_diff": 0.05 * final_value,  # Small minimum changes
                "min_slope": 58,  # 30° normalized
                "max_non_alternating": 5,  # Allow longer trends
                "min_value": 0.50 * final_value,  # Minimum baseline
                "baseline": 0.75 * final_value,  # Typical minimum
            }
        elif self.resource == "CPU":
            # High variance and very spiky, with a lot of noise, but also a lot of alternating patterns, and high slope and high range and high avg_diff but low min_diff, and very high std deviation
            self.thresholds = {
                "std": 0.35 * final_value,  # Higher variation for spikes
                "range": 0.95 * final_value,  # Allow near-full range
                "avg_diff": 0.4 * final_value,  # Sharp changes
                "min_diff": 0.1 * final_value,  # Maintain minimum movement
                "min_slope": 373,  # 75° angle for sharp spikes
                "max_non_alternating": 2,  # Quick alternations
                "idle_baseline": 0.05 * final_value,  # Typical idle CPU level
            }
        elif self.resource == "GPU":
            # GPU usage has sharp transitions like CPU but with longer sustained peaks
            # and fewer rapid alternations. Maintains clear idle baseline.
            self.thresholds = {
                "std": 0.30 * final_value,  # High variation but less than CPU
                "range": 0.95 * final_value,  # Allow full utilization range
                "avg_diff": 0.35 * final_value,  # Sharp but fewer changes than CPU
                "min_diff": 0.15 * final_value,  # Larger minimum movements
                "min_slope": 373,  # 75° angle for sharp transitions
                "max_non_alternating": 4,  # Allow longer sustained levels
                "idle_baseline": 0.05 * final_value,  # Typical idle GPU level
            }
        elif self.resource == "Memory":
            # Memory usage shows gradual changes with high baseline
            # Usually 40-80% utilized with small fluctuations
            self.thresholds = {
                "std": 0.15 * final_value,  # Moderate variation
                "range": 0.40 * final_value,  # Typically 40-80% range
                "avg_diff": 0.10 * final_value,  # Gradual changes
                "min_diff": 0.05 * final_value,  # Small minimum changes
                "min_slope": 58,  # 30° angle for gradual changes
                "max_non_alternating": 8,  # Allow long sustained levels
                "baseline": 0.40 * final_value,  # Minimum OS usage
                "typical_range": (0.4, 0.8),  # Normal operating range
            }
        else:
            print("Unknown resource type")
            self.thresholds = {
                "std": (
                    0.1 * final_value
                    if final_value >= 10
                    else max(0.05 * final_value, 2)
                ),
                "range": (
                    0.3 * final_value
                    if final_value >= 20
                    else max(0.2 * final_value, 2)
                ),
                "avg_diff": 0.20 * (0.3 * final_value),  # 20% of expected range
                "min_diff": (
                    0.03 * final_value
                    if final_value >= 10
                    else max(0.01 * final_value, 2)
                ),
                "min_slope": 58,  # 30 degree angle when normalized
                "max_non_alternating": 3,
            }

    def normalize_slope(self, slope, max_height=100):
        """Convert slope to normalized form assuming max height of 100"""
        return slope * (max_height / 1)

    def check_slope(self, points):
        # Get slopes between consecutive points
        slopes = np.diff(points[:14]) / 1

        # Separate positive and negative slopes
        pos_slopes = slopes[slopes > 0]
        neg_slopes = slopes[slopes < 0]

        # Find min positive and max negative (most negative)
        min_pos = np.min(pos_slopes) if len(pos_slopes) > 0 else 0
        max_neg = np.max(neg_slopes) if len(neg_slopes) > 0 else 0

        # Normalize both
        norm_pos = self.normalize_slope(min_pos)
        norm_neg = self.normalize_slope(abs(max_neg))

        # Check both against threshold
        pos_ok = norm_pos > self.thresholds["min_slope"]
        neg_ok = norm_neg > self.thresholds["min_slope"]

        # Return combined result and minimum of normalized slopes
        return (pos_ok and neg_ok), min(norm_pos, norm_neg)

    def fix_slope(self, points):
        max_height = 100  # Match normalization from check_slope
        target_slope = (
            self.thresholds["min_slope"] / max_height
        )  # De-normalize threshold

        for i in range(1, 14):
            # Calculate normalized slope
            current_slope = (points[i] - points[i - 1]) / 1
            normalized_slope = abs(self.normalize_slope(current_slope))

            if normalized_slope < self.thresholds["min_slope"]:
                direction = np.sign(points[i] - points[i - 1])
                # Apply de-normalized target slope
                points[i] = points[i - 1] + (direction * target_slope)

        return points

    def fix_std_deviation(self, points, current_std):
        target_std = self.thresholds["std"]
        if current_std < target_std:
            scale_factor = target_std / current_std
            mean = np.mean(points[:14])
            points[:14] = mean + (points[:14] - mean) * scale_factor
        return points

    def fix_range(self, points, current_range):
        target_range = self.thresholds["range"]
        if current_range < target_range:
            scale = target_range / current_range
            mean = np.mean(points[:14])
            points[:14] = mean + (points[:14] - mean) * scale
        return points

    def fix_avg_diff(self, points, current_avg_diff):
        target_diff = self.thresholds["avg_diff"]
        if current_avg_diff < target_diff:
            for i in range(1, 14):
                if abs(points[i] - points[i - 1]) < target_diff:
                    points[i] += np.sign(points[i] - points[i - 1]) * (
                        target_diff - current_avg_diff
                    )
        return points

    def fix_alternating(self, points):
        # mean = np.mean(points[:14])
        # consecutive_same_side = 0

        # for i in range(1, 14):
        #     if (points[i] > mean and points[i - 1] > mean) or (
        #         points[i] < mean and points[i - 1] < mean
        #     ):
        #         consecutive_same_side += 1
        #         if consecutive_same_side >= self.thresholds["max_non_alternating"]:
        #             points[i] = mean + (points[i] - mean) * 0.5
        #             consecutive_same_side = 0
        # return points

        # Replace the smoothing section:
        mean = np.mean(points[:14])
        consecutive_same_side = 0

        for i in range(1, 14):
            if (points[i] > mean and points[i - 1] > mean) or (
                points[i] < mean and points[i - 1] < mean
            ):
                consecutive_same_side += 1
                if consecutive_same_side >= self.thresholds["max_non_alternating"]:
                    # 30% chance to flip to opposite side
                    if random.random() < 0.3:
                        distance_from_mean = points[i] - mean
                        points[i] = mean - distance_from_mean  # Flip to opposite side
                    else:
                        points[i] = (
                            mean + (points[i] - mean) * 0.5
                        )  # Original smoothing
                    consecutive_same_side = 0
        return points

    def fix_min_diff(self, points, current_min_diff):
        target_min = self.thresholds["min_diff"]
        if current_min_diff < target_min:
            for i in range(1, 14):
                if abs(points[i] - points[i - 1]) < target_min:
                    points[i] += np.sign(points[i] - points[i - 1]) * target_min
        return points

    def apply_corrections(self, points, spike_results, stats, max_allowed):
        std_ok, range_ok, avg_diff_ok, min_diff_ok, alternating_ok, slope_ok = (
            spike_results
        )
        std_dev, min_max_range, avg_diff, min_diff, alternating, min_slope = stats

        if not std_ok:
            points = self.fix_std_deviation(points, std_dev)
        if not range_ok:
            points = self.fix_range(points, min_max_range)
        if not avg_diff_ok:
            points = self.fix_avg_diff(points, avg_diff)
        if not min_diff_ok:
            points = self.fix_min_diff(points, min_diff)
        if not alternating_ok:
            points = self.fix_alternating(points)

        # Clip to max allowed values
        for i in range(14):
            points[i] = min(max_allowed[i], points[i])

        return points
        # Check spikiness function

    def check_spikiness(self, points, final_value, resource) -> tuple:
        # Original statistical metrics
        std_dev = np.std(points[:14])
        min_max_range = max(points) - min(points)
        abs_diffs = np.abs(np.diff(points[:14]))
        avg_diff = np.mean(abs_diffs)
        min_diff = np.min(abs_diffs)

        # Enhanced alternating pattern check
        mean = np.mean(points[:14])
        alternating_ok = True
        consecutive_same_side = 0

        # Original threshold logic
        # if final_value >= 10:
        std_ok = std_dev > self.thresholds["std"]
        range_ok = min_max_range > self.thresholds["range"]
        avg_diff_ok = avg_diff > self.thresholds["avg_diff"]
        min_diff_ok = min_diff > self.thresholds["min_diff"]
        if final_value >= 10:
            slope_ok, min_slope = self.check_slope(points)
        else:
            slope_ok, min_slope = True, 0

        # else:
        # std_ok = std_dev > 2
        # range_ok = min_max_range > 2
        # avg_diff_ok = avg_diff > 0.06 * final_value
        # min_diff_ok = min_diff > 0.02 * final_value
        # slope_ok, min_slope = self.check_slope(points)
        store_cons = 0
        for i in range(1, len(points[:14])):
            if (points[i] > mean and points[i - 1] > mean) or (
                points[i] < mean and points[i - 1] < mean
            ):
                consecutive_same_side += 1

                slope_between = abs(points[i] - points[i - 1]) / 1  # time step = 1
                if (
                    consecutive_same_side >= 3
                    and slope_between < self.thresholds["min_slope"]
                ) or (
                    consecutive_same_side >= 5
                    and slope_between < 0.5 * self.thresholds["min_slope"]
                ):
                    store_cons = consecutive_same_side
                    alternating_ok = False
                    break
            else:
                consecutive_same_side = 0

        return (
            [std_ok, range_ok, avg_diff_ok, min_diff_ok, alternating_ok, slope_ok],
            [
                std_dev,
                min_max_range,
                avg_diff,
                min_diff,
                store_cons,
                min_slope,
            ],
        )


def exp_smooth(data, alpha=0.3):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


base_values = {
    "CPU": np.minimum(
        np.maximum(np.random.normal(2, 1, 16), 2), 3
    ),  # Set maximum to 10
    "Memory": np.random.uniform(7, 10, 16),
    "GPU": np.random.uniform(1, 3, 16),
    "Disk": np.random.uniform(3, 8, 16),
}
colors = {"CPU": "teal", "Memory": "blue", "GPU": "mediumpurple", "Disk": "lightgreen"}
# Calculate total iterations
total_combinations = len(combinations)
total_resources = 4  # CPU, Memory, GPU, Disk
total_iterations = total_combinations * total_resources

# Create progress bars
pbar_main = tqdm(total=total_iterations, desc="Generating graphs")


def compare_arrays(arr1, arr2=None, name="Arrays"):
    """Compare statistical properties of two arrays.

    Args:
        arr1: First numpy array
        arr2: Second numpy array
        name: Description of comparison (default: "Arrays")
    """
    if arr2 is None:
        stats = f"""
        {name} Statistics:
        --------------------------------
        Max: {np.max(arr1)}
        Min: {np.min(arr1)}
        Mean: {np.mean(arr1)}
        Std: {np.std(arr1)}
        Median: {np.median(arr1)}
        Sum: {np.sum(arr1)}
        Variance: {np.var(arr1)}
        """
        return (
            stats,
            {
                "max": np.max(arr1),
                "min": np.min(arr1),
                "mean": np.mean(arr1),
                "std": np.std(arr1),
                "median": np.median(arr1),
                "sum": np.sum(arr1),
                "variance": np.var(arr1),
            },
        )
    else:
        diff = arr1 - arr2
        stats = f"""
        {name} Comparison:
        --------------------------------
        Difference: {diff}
        Array1 stats: max={np.max(arr1)}, min={np.min(arr1)}
        Array2 stats: max={np.max(arr2)}, min={np.min(arr2)}
        Difference stats:
        Max: {np.max(diff)}
        Min: {np.min(diff)} 
        Mean: {np.mean(diff)}
        Std: {np.std(diff)}
        Median: {np.median(diff)}
        Sum: {np.sum(diff)}
        Variance: {np.var(diff)}

        Highest variance: Array1: {np.var(arr1)} | Array2: {np.var(arr2)} Winner: {'Array1' if np.var(arr1) > np.var(arr2) else 'Array2'}
        Highest std: Array1: {np.std(arr1)} | Array2: {np.std(arr2)} Winner: {'Array1' if np.std(arr1) > np.std(arr2) else 'Array2'}
        Highest min-max range: Array1: {np.max(arr1) - np.min(arr1)} | Array2: {np.max(arr2) - np.min(arr2)} Winner: {'Array1' if (np.max(arr1) - np.min(arr1)) > (np.max(arr2) - np.min(arr2)) else 'Array2'}
        Spikiest: {'Array1' if np.max(arr1) - np.min(arr1) > np.max(arr2) - np.min(arr2) else 'Array2'} | {'Array1' if np.std(arr1) > np.std(arr2) else 'Array2'} | {'Array1' if np.var(arr1) > np.var(arr2) else 'Array2'} Best: {'Array1' if np.max(arr1) - np.min(arr1) > np.max(arr2) - np.min(arr2) else 'Array2'} | {'Array1' if np.std(arr1) > np.std(arr2) else 'Array2'} | {'Array1' if np.var(arr1) > np.var(arr2) else 'Array2'}
        """
    return (
        stats,
        {
            "max_diff": np.max(diff),
            "min_diff": np.min(diff),
            "mean_diff": np.mean(diff),
            "std_diff": np.std(diff),
            "median_diff": np.median(diff),
            "sum_diff": np.sum(diff),
            "variance_diff": np.var(diff),
            "highest_var:": "Array1" if np.var(arr1) > np.var(arr2) else "Array2",
            "highest_std:": "Array1" if np.std(arr1) > np.std(arr2) else "Array2",
            "highest_minmax_range:": (
                "Array1"
                if (np.max(arr1) - np.min(arr1)) > (np.max(arr2) - np.min(arr2))
                else "Array2"
            ),
            "spikiest:": (
                "Array1"
                if np.max(arr1) - np.min(arr1) > np.max(arr2) - np.min(arr2)
                else "Array2"
            ),
        },
    )


def enforce_upward_trend(points, resource):
    """Ensure each point that is less than the previous point does not go lower than the lowest of the last three decreasing points

    Args:
        points: numpy array of points

    Returns:
        points: numpy array of points with enforced upward trend
    """
    # Steps:
    # 1. Iterate through the points
    # 2. Check if the current point is less than the previous point
    # 3. If it is, collect the last three decreasing points
    # 4. Ensure the current point is not lower than the lowest of the last three decreasing points
    if resource == "CPU":
        return points
    if resource == "Memory":
        # each successive point must be the same or higher than the previous point
        for i in range(1, len(points)):
            if points[i] < points[i - 1]:
                points[i] = points[i - 1]
    for i in range(1, len(points)):
        if points[i] < points[i - 1]:
            # Collect the last three points that were lower than their previous point
            last_three = []
            for j in range(i, -1, -1):
                if points[j] < points[j - 1]:
                    last_three.append(points[j])
                    if len(last_three) == 3:
                        break
            # Ensure the current point is not lower than the lowest of the last three decreasing points
            if len(last_three) == 3:
                lowest = min(last_three)
                if points[i] < lowest:
                    points[i] = lowest

    return points


# Function to generate graph points with updated logic for alternation, reflection, and spikiness
def generate_graph_points_with_alternation(
    final_value, base_values, resource, graph_num
):
    # Check if final_value is zero; if so, use base_values[15] + base_values[15]
    if final_value == 0:
        final_value = base_values[15]
    spike_checker = SpikinessFixer(final_value, resource)

    points = np.zeros(16)  # 16 points per graph

    # 1–12: Generate random interpolations with increased variance
    random_points = np.zeros(12)
    if resource == "Memory":
        # Memory will be mostly flat, always increasing, and with less variance
        random_points = np.linspace(0.8 * final_value, 0.7 * final_value, 12)
        # Add small random variations while maintaining increasing trend
        variations = np.random.uniform(0, 0.05 * final_value, 12)
        random_points += np.cumsum(variations)
        # Ensure points stay within bounds and maintain upward trend
        random_points = np.clip(random_points, 0.8 * final_value, final_value)
        for i in range(1, len(random_points)):
            if random_points[i] < random_points[i - 1]:
                random_points[i] = random_points[i - 1]
    elif resource == "Disk":
        # Disk will be highly variable, with some points potentially being negative, but the mean stays close to the final_value
        random_points = np.random.normal(
            0.55 * final_value, 0.15 * final_value, 12
        )  # Higher variance for disk
        # Add upward trend bias
        trend = np.linspace(0, 0.2 * final_value, 12)  # 20% increase over time
        random_points += trend
        # Add occasional spikes (10% chance per point)
        spikes = np.random.choice([0, 1], 12, p=[0.9, 0.1])
        random_points += spikes * (0.2 * final_value)
        # Clip values between 25% and 110% of final value
        random_points = np.clip(
            random_points,
            0.50 * final_value,  # Minimum 25%
            max(0.9 * final_value, 100),  # Allow some overflow
        )
    elif resource == "CPU":
        # Start with low baseline (idle) pattern
        random_points = np.random.normal(
            0.05 * final_value,  # Idle baseline
            0.20 * final_value,  # Initial variation
            12,
        )

        # Add sharp spikes (30% chance per point)
        spikes = np.random.choice([0, 0.3, 0.6, 0.8], 12, p=[0.5, 0.25, 0.2, 0.05])
        random_points += spikes * final_value

        # Add alternating pattern
        alternating = np.sin(np.linspace(0, 4 * np.pi, 12)) * (0.2 * final_value)
        random_points += alternating

        # Clip values between idle and max
        random_points = np.clip(
            random_points,
            (
                0.25 * final_value if final_value > 10 else 0.05 * final_value
            ),  # Minimum idle level
            0.75 * final_value,  # Maximum utilization
        )
    elif resource == "GPU":
        # Start with idle baseline
        random_points = np.random.normal(
            0.05 * final_value,  # Idle baseline
            0.10 * final_value,  # Initial variation
            12,
        )

        # Add fewer but longer spikes (20% chance, higher values)
        spikes = np.zeros(12)
        for i in range(0, 12, 3):  # Check every 3 points for sustained peaks
            if np.random.random() < 0.2:  # 20% chance of spike
                spikes[i : i + 3] = np.random.choice([0.7, 0.9])  # Sustained high usage
        random_points += spikes * final_value

        # Add some random sustained levels
        sustained = np.random.choice([0, 0.3, 0.6], 12, p=[0.6, 0.3, 0.1])
        random_points += sustained * final_value

        # Clip values between idle and max
        random_points = np.clip(
            random_points,
            0.15 * final_value,  # Minimum idle level
            0.95 * final_value,  # Maximum utilization
        )
    else:
        random_points = np.random.normal(
            0.5 * final_value, 0.25 * final_value, 12
        )  # Increased variance
        random_points = np.clip(random_points, 0, 0.8 * final_value + base_values[15])
    points[:12] = random_points

    # Updated interpolation for points 12, 13, and 14
    points[12] = np.interp(0.6, [0, 1], [points[11], final_value])
    points[13] = np.interp(0.3, [0, 1], [points[12], final_value])
    points[14] = np.interp(0.8, [0, 1], [points[13], final_value])

    # 16: Final value
    points[15] = min(
        min(final_value + base_values[15], 100), final_value
    )  # Cap at 100% utilization
    # Add base values
    points += base_values

    # Enforce upward trend
    points = enforce_upward_trend(points, resource)
    try:
        # attempt to display the graph if final_value is one of the following: 69, 67, 40, 81, 54, 47, 35, 71, 3, 5, 0, 1, 15, 20, 5, 10
        x = list(range(1, 17))  # 16 time points

        # Create figure and plot
        fig = plt.figure(figsize=(4, 4))
        plt.fill_between(x, points, color=color, alpha=0.4)
        # plt.plot(x, points, marker="o", color=color, linewidth=2, markersize=6)

        # Style the graph
        # Style improvements
        # Style the graph
        plt.title(f"{resource} Usage")
        plt.ylim(0, 100)
        plt.grid(alpha=0.3)

        # Remove X-axis ticks and labels
        plt.xticks([])

        # Remove Y-axis label
        plt.gca().set_ylabel(None)

        # Adjust layout to remove outer borders
        plt.tight_layout(pad=0)
        filename = os.path.join(output_dir, f"{graph_num}_{resource}_PRECHECKS.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")  # X resolution halved
        plt.close(fig)
    except Exception as e:
        print(
            f"Failed to display graph for {resource} with final value {final_value}: {e}, trying again"
        )
        try:
            if final_value in [
                69,
                67,
                40,
                81,
                54,
                47,
                35,
                71,
                3,
                5,
                0,
                1,
                15,
                20,
                5,
                10,
            ]:
                # try a different method to display the graph
                # start by creating a new figure
                fig = plt.figure()
                # create a new subplot
                ax = fig.add_subplot(111)
                plt.fill_between(x, points, color=color, alpha=0.4)

                # plot the points
                ax.plot(points, marker="o", color=colors[resource])
                # set the title
                ax.set_title(f"{resource} Usage")
                # set the x-axis label
                ax.set_xlabel("Time")
                # set the y-axis label
                ax.set_ylabel("Usage")
                # show the grid
                ax.grid(True)
                print(f"Displaying graph for {resource} with final value {final_value}")
                # show the graph
                plt.show()
        except Exception as ee:
            print(
                f"Failed to display graph for {resource} with final value {final_value} with errors {e} and {ee}"
            )
            pass

    # Calculate mean values for scaling
    mean_value = np.mean(points[:12])

    def compute_max_allowed(
        points, final_value, baseline_percentage, growth_type="linear", curve_factor=1.0
    ):
        """
        Computes an array of 'max_allowed' values that starts from the higher of
        (baseline) or (points[0]) and progresses (increasingly) up to but never
        beyond min(final_value, 100).

        Parameters:
        - points: list of numeric values
        - final_value: the target upper value we're working toward
        - baseline_percentage: fraction of final_value to use as a baseline
        - growth_type: 'linear', 'quadratic', 'sqrt', or 'custom' (controls the progression shape)
        - curve_factor: float, exponent used for 'custom' growth_type.
                        For 'quadratic', use 2.
                        For 'sqrt', use 0.5.
                        Default is 1.0 (linear).

        Returns:
        A list of 'max_allowed' values with the same length as 'points'.
        """

        # 1. Calculate numeric baseline from the baseline_percentage
        baseline = baseline_percentage * final_value

        # 2. Determine the starting value
        start_value = max(baseline, max(points[0] + 1, np.max(base_values)))

        # 3. Determine the upper limit
        limit = min(final_value, 100)

        # 4. Prepare a list to hold the results
        max_allowed = []

        # 5. Determine how many 'points' we have
        num_points = len(points)

        # Edge case: If there's only one point, just clamp to limit
        if num_points <= 1:
            return [min(start_value, limit)]

        # 6. Calculate each successive max_allowed
        for i in range(num_points):
            # fraction will move from 0 at i=0 to 1 at i=(num_points - 1)
            fraction = i / (num_points - 1)

            # Determine the growth factor based on growth_type
            if growth_type == "linear":
                # Linear interpolation
                fraction_growth = fraction
            elif growth_type == "quadratic":
                # Quadratic growth: fraction^2
                fraction_growth = fraction**2
            elif growth_type == "sqrt":
                # Square root growth: sqrt(fraction)
                fraction_growth = fraction**0.5
            elif growth_type == "custom":
                # Custom growth based on curve_factor
                fraction_growth = fraction**curve_factor
            else:
                # Default to linear if an unknown type is provided
                fraction_growth = fraction

            # Interpolate between start_value and limit
            val = start_value + (limit - start_value) * fraction_growth

            # Round to nearest integer
            val = round(val)

            # Ensure non-decreasing
            if i > 0 and val < max_allowed[-1]:
                val = max_allowed[-1]

            if val > limit:
                val = limit

            max_allowed.append(val)

        return max_allowed

    if resource == "Memory":
        max_allowed = compute_max_allowed(
            points[:14], final_value, 0.6, growth_type="sqrt"
        )
    elif resource == "Disk":
        max_allowed = compute_max_allowed(
            points[:14], final_value, 0.25, growth_type="linear"
        )
    elif resource == "CPU":
        max_allowed = compute_max_allowed(
            points[:14], final_value, 0.10, growth_type="quadratic"
        )
    elif resource == "GPU":
        max_allowed = compute_max_allowed(
            points[:14], final_value, 0.05, growth_type="linear"
        )
    assert (
        np.max(max_allowed) <= 100
    ), f"Max allowed exceeds 100, max_allowed: {np.max(max_allowed)}"
    assert np.min(max_allowed) >= 0, "Max allowed is below 0"
    assert (
        np.max(max_allowed) <= final_value
    ), f"Max allowed exceeds final value, final_value: {final_value}, max_allowed: {np.max(max_allowed)}, \n max_allowed: {max_allowed}"
    # max_allowed = compute_max_allowed(points, final_value, 0.75, growth_type="linear")
    print(f"max_allowed: {max_allowed} for {resource} with final value {final_value}")
    # Apply scaled sinusoidal effect
    scaling_factor = np.random.uniform(0.1, 0.3)  # Random scaling factor for variation
    min_baseval, max_baseval = np.min(base_values), np.max(base_values)
    range_basevals = max_baseval - min_baseval
    mean_value = np.mean(points)
    spikiness = spike_checker.check_spikiness(points, final_value, resource)[0]
    if resource not in ["Memory", "Disk"]:

        print("Applying sinusoidal effect.")
        sinusoidal_effect = (
            (mean_value * scaling_factor) + (range_basevals * 0.2)
        ) * np.sin(np.linspace(0, 2 * np.pi, 14))
        # Replace existing prints with:
        print(
            compare_arrays(
                max_allowed, points[:14], "max_allowed vs points[:14] Pre-sinusoidal"
            )[0]
        )
        print(
            compare_arrays(
                max_allowed,
                points[:14] + sinusoidal_effect,
                "max_allowed vs points[:14] Post-sinusoidal",
            )[0]
        )
        # print the sinusoidal effect stats
        print(compare_arrays(sinusoidal_effect, name="Sinusoidal Effect")[0])
        # print the points[:14] after sinusoidal effect
        print(
            compare_arrays(
                points[:14] + sinusoidal_effect, name="points[:14] + sinusoidal_effect"
            )[0]
        )
        # print the difference between points[:14] and points[:14] + sinusoidal effect
        print(
            compare_arrays(
                points[:14],
                points[:14] + sinusoidal_effect,
                "points[:14] vs points[:14] + sinusoidal_effect",
            )[0]
        )

        # Apply sinusoidal effect and ensure within max allowed

        points[:14] += sinusoidal_effect
        spikiness, spikestats = spike_checker.check_spikiness(
            points, final_value, resource
        )
        # counts falses in spikiness
        count_or = sum(not x for x in spikiness)
        if not spikiness[-1]:
            # Slope check failed. Apply corrections
            print(
                f"Slope check failed. Applying corrections. slope: {spikestats[-1]} needed: {spike_checker.thresholds['min_slope']} resource: {resource} graph_num: {graph_num}"
            )
            points_before = points.copy()
            points = spike_checker.fix_slope(points)
            spikiness = spike_checker.check_spikiness(points, final_value, resource)[0]
            count_new = sum(not x for x in spikiness)
            if count_new > count_or:
                points = points_before.copy()
            else:
                count_or = count_new
        points_before = points.copy()
        avgdiff_before = spikiness[-4]
        alternating_before = spikiness[-2]
        slope_before = spikiness[-1]
        min_diff_before = spikiness[-3]
        if not spikiness[-3]:
            print(
                f"Min diff check failed. Applying corrections. min_diff: {spikestats[-3]} needed: {spike_checker.thresholds['min_diff']} resource: {resource} graph_num: {graph_num}"
            )

            points = spike_checker.fix_min_diff(
                points, np.min(np.abs(np.diff(points[:14])))
            )
            spikiness = spike_checker.check_spikiness(points, final_value, resource)[0]
            count_new = sum(not x for x in spikiness)
            if count_new > count_or:
                points = points_before.copy()
            else:
                if count_new < count_or:
                    count_or = count_new
                elif count_new == count_or:
                    if spikiness[-3]:
                        apply_fix = True
                        # make sure previously successful checked conditions are not reversed
                        for x in zip(
                            [spikiness[-4], spikiness[-3], spikiness[-2]],
                            [avgdiff_before, min_diff_before, alternating_before],
                        ):
                            if not x[0] and x[1]:
                                # do not apply the fix
                                apply_fix = False
                                break
                        if not apply_fix:
                            print("Rejected min diff fix. Reverted to previous points.")
                            points = points_before.copy()
                        else:
                            print("Accepted min diff fix.")
                            count_or = count_new
        points_before = points.copy()

        if not spikiness[-4]:
            print(
                f"Avg diff check failed. Applying corrections. avg_diff: {spikestats[-4]} needed: {spike_checker.thresholds['avg_diff']} resource: {resource} graph_num: {graph_num}"
            )
            points = spike_checker.fix_avg_diff(
                points, np.mean(np.abs(np.diff(points[:14])))
            )
            spikiness = spike_checker.check_spikiness(points, final_value, resource)[0]
            count_new = sum(not x for x in spikiness)
            if count_new > count_or:
                points = points_before.copy()
            else:
                if count_new < count_or:
                    count_or = count_new
                elif count_new == count_or:
                    if spikiness[-4]:
                        apply_fix = True
                        # make sure previously successful checked conditions are not reversed
                        for x in zip(
                            [spikiness[-4], spikiness[-3], spikiness[-2]],
                            [avgdiff_before, min_diff_before, alternating_before],
                        ):
                            if not x[0] and x[1]:
                                # do not apply the fix
                                apply_fix = False
                                break
                        if not apply_fix:
                            print("Rejected avg diff fix. Reverted to previous points.")
                            points = points_before.copy()
                        else:
                            print("Accepted avg diff fix.")
                            count_or = count_new
        points_before = points.copy()

        if not spikiness[-2]:
            print(
                f"Alternating pattern check failed. Applying corrections. num_alternating: {spikestats[-2]} needed less than: {spike_checker.thresholds['max_non_alternating']} resource: {resource} graph_num: {graph_num}"
            )
            points = spike_checker.fix_alternating(points)
            spikiness = spike_checker.check_spikiness(points, final_value, resource)[0]
            count_new = sum(not x for x in spikiness)
            if count_new > count_or:
                points = points_before.copy()
            else:
                if count_new < count_or:
                    count_or = count_new
                elif count_new == count_or:
                    if spikiness[-2]:
                        apply_fix = True
                        # make sure previously successful checked conditions are not reversed
                        for x in zip(
                            [spikiness[-4], spikiness[-3], spikiness[-2]],
                            [avgdiff_before, min_diff_before, alternating_before],
                        ):
                            if not x[0] and x[1]:
                                # do not apply the fix
                                apply_fix = False
                                break
                        if not apply_fix:
                            print(
                                "Rejected alternating pattern fix. Reverted to previous points."
                            )
                            points = points_before.copy()
                        else:
                            print("Accepted alternating pattern fix.")
                            count_or = count_new
        points_before = points.copy()
        count_a = 0
        spike_results = spike_checker.check_spikiness(points, final_value, resource)
        while not all([spike_results[0]]) and resource != "Memory" and count_a < 5:
            points = spike_checker.apply_corrections(
                points, spike_results[0], spike_results[1], max_allowed
            )
            spike_results = spike_checker.check_spikiness(
                points, final_value, resource
            )[0]
            count_a += 1
            count_new = sum(not x for x in spikiness)
            if count_new > count_or:
                points = points_before.copy()
            else:
                if count_new < count_or:
                    count_or = count_new
                elif count_new == count_or:
                    apply_fix = True
                    # make sure previously successful checked conditions are not reversed
                    for x in zip(
                        [spike_results[-4], spike_results[-3], spike_results[-2]],
                        [spikiness[-4], spikiness[-3], spikiness[-2]],
                    ):
                        if not x[0] and x[1]:
                            # do not apply the fix
                            apply_fix = False
                            break
                    if not apply_fix:
                        print("Rejected spikiness fix. Reverted to previous points.")
                        points = points_before.copy()
                    else:
                        print("Accepted spikiness fix.")
                        count_or = count_new

    # points[:14] = np.clip(points[:14], 0, max_allowed)
    # points[:14] = np.where(np.isclose(points[:14], 0), -points[:14], points[:14])
    # points[:14] = np.where(
    #     np.isclose(points[:14], max_allowed),
    #     max_allowed - (points[:14] - max_allowed),
    #     points[:14],
    # )
    points[:14] = np.clip(points[:14], base_values[:14], max_allowed)
    try:
        # attempt to display the graph if final_value is one of the following: 69, 67, 40, 81, 54, 47, 35, 71, 3, 5, 0, 1, 15, 20, 5, 10
        x = list(range(1, 17))  # 16 time points

        # Create figure and plot
        fig = plt.figure(figsize=(4, 4))
        plt.fill_between(x, points, color=color, alpha=0.4)
        # plt.plot(x, points, marker="o", color=color, linewidth=2, markersize=6)

        # Style the graph
        # Style improvements
        # Style the graph
        plt.title(f"{resource} Usage")
        plt.ylim(0, 100)
        plt.grid(alpha=0.3)

        # Remove X-axis ticks and labels
        plt.xticks([])

        # Remove Y-axis label
        plt.gca().set_ylabel(None)

        # Adjust layout to remove outer borders
        plt.tight_layout(pad=0)
        filename = os.path.join(
            output_dir, f"{graph_num}_{resource}_FIRST_SPIKECHECK_.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches="tight")  # X resolution halved
        plt.close(fig)
    except Exception as e:
        print(
            f"Failed to save graph for {resource} with final value {final_value}: {e}"
        )
    for i in range(13, -1, -1):
        points[i] = min(max_allowed[i], points[i])
    # Guarantee alternation if sinusoidal adjustments are suppressed
    flip_chance = 0.999  # 90% chance to do the reflection
    flip_strength = np.random.uniform(0.3, 0.7)  # Vary reflection strength

    points_copy = points.copy()
    mean_value = np.mean(points[:14])
    if resource != "Memory":
        for i in range(1, 14):
            flip_strength = np.random.uniform(0.3, 0.7)  # Vary reflection strength
            mean_value = np.mean(points[:14])
            if mean_value > max_allowed[i]:
                mean_value = np.mean(points[: i + 1])
            if mean_value > max_allowed[i]:
                mean_value = np.mean(points[i - 1 : i + 1])

            if (
                (points[i - 1] > mean_value and points[i] > mean_value)
                or (points[i - 1] < mean_value and points[i] < mean_value)
                and points[i - 1] < max(max_allowed[i - 1], final_value)
                and points[i - 1] > min(base_values[i - 1], 6)
            ):
                if (
                    np.random.rand() < flip_chance
                    and points[i] >= base_values[i]
                    and points[i] <= max_allowed[i]
                ):
                    try:
                        # Instead of mean_value - offset, do a partial reflection
                        offset = points[i] - mean_value
                        print(f"mean_value: {mean_value}, offset: {offset}")
                        print(
                            f"Flipping point {i} with offset {offset} and strength {flip_strength}, \n new value: {mean_value - flip_strength * offset}, \n old value: {points[i]} previous value: {points[i - 1]}, \n max allowed: {max_allowed[i]}, \n base value: {base_values[i]}"
                        )
                        points[i] = mean_value - flip_strength * offset
                        # assert (
                        #     round(points[i]) >= base_values[i]
                        # ), f"Assertion failed: points at index {i} is {points[i]} which is less than in base_values which is {base_values[i]} "
                        # assert (
                        #     round(points[i]) <= max_allowed[i]
                        # ), f"Assertion failed: points at index {i} is {points[i]} which is more than in max_allowed which is {max_allowed[i]} "

                        if points[i] < base_values[i]:
                            print(
                                f"Assertion failed: points at index {i} is {points[i]} which is less than in base_values which is {base_values[i]} "
                            )
                            points[i] = base_values[i]
                        if points[i] > max_allowed[i]:
                            print(
                                f"Assertion failed: points at index {i} is {points[i]} which is more than in max_allowed which is {max_allowed[i]} "
                            )
                            points[i] = max_allowed[i]

                        # if points[i - 1] > mean_value:
                        #     assert (
                        #         points[i - 1] > mean_value and points[i] < mean_value
                        #     ), f"Assertion failed: {points[i - 1]} > {mean_value} and {points[i]} < {mean_value}"
                        # elif points[i - 1] < mean_value:
                        #     assert (
                        #         points[i - 1] < mean_value and points[i] > mean_value
                        #     ), f"Assertion failed: {points[i - 1]} < {mean_value} and {points[i]} > {mean_value}"

                        # Revert to doing the mean_value + offset instead of the partial reflection
                        if points[i] > mean_value and points[i - 1] > mean_value:
                            #
                            # points[i] = mean_value + flip_strength * offset # should
                            if i < 13 and points[i + 1] < mean_value:
                                # this means we both the current and previous point are above the mean_value, but the next point is below the mean_value, so if we flip this point, the graph will be flipped, so lets look at point[i + 2] to see if we can do the flip. If point[i + 2] is also above the mean_value, then if we do the flip, the graph will be flipped, so we need to do the opposite of the flip. If point[i + 2] is below the mean_value, then we can do the flip because the
                                # so we should also look at point[i + 2] to see if we can do the flip. If point[i + 2] is below the mean_value, as well, then we can do the flip. If point[i + 2] is above the mean_value, then we need to do the opposite of the flip.
                                if i < 12 and points[i + 2] < mean_value:
                                    # this means we can do the flip because the next point is on the same side of the mean_value as this and the previous point, so when this is flipped, the graph will still be flipped
                                    points[i] = mean_value - flip_strength * offset
                                    if points[i] > mean_value:
                                        # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                        points[i] -= 2 * (mean_value - points[i])
                                elif i < 12 and points[i + 2] > mean_value:
                                    # this indicates that perhaps we should either do the opposite of the flip or do nothing, so we will do nothing
                                    if i > 1 and points[i - 2] > mean_value:
                                        # if 2 points before this point is also above the mean_value, then we can should flip point[i - 1] instead of point[i]. However if points_copy[i - 1] or points_copy[i - 2] is below the mean_value, that means we are flipping the graph, so we should backtrack
                                        if points_copy[i - 2] > mean_value:
                                            if points_copy[i - 1] > mean_value:
                                                # this indicates that neither points[i - 1] or points[i - 2] was flipped
                                                points[i - 1] = (
                                                    mean_value
                                                    - flip_strength
                                                    * (points[i - 1] - mean_value)
                                                )
                                                if points[i - 1] > mean_value:
                                                    # this means we need to do either do another flip or do something else. We can just do point[i] - = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                    points[i - 1] -= 2 * (
                                                        mean_value - points[i - 1]
                                                    )
                                                    if points[i - 1] > mean_value:
                                                        # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                        points[i - 1] -= 2 * (
                                                            mean_value - points[i - 1]
                                                        )
                                                        if points[i - 1] > mean_value:
                                                            points[i - 1] = (
                                                                points_copy[i - 1]
                                                                - (
                                                                    points[i - 1]
                                                                    - mean_value
                                                                )
                                                                + 1
                                                            )
                                            elif points_copy[i - 1] < mean_value:
                                                points[i - 1] = points_copy[i - 1]
                                        elif (
                                            points_copy[i - 2] < mean_value
                                            and points_copy[i - 1] < mean_value
                                        ):
                                            points[i - 2] = points_copy[i - 2]
                                        elif (
                                            points_copy[i - 2] < mean_value
                                            and points_copy[i - 1] > mean_value
                                        ):
                                            points[i - 1] = (
                                                mean_value
                                                - flip_strength
                                                * (points[i - 1] - mean_value)
                                            )
                                            if points[i - 1] > mean_value:
                                                # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                points[i - 1] -= 2 * (
                                                    mean_value - points[i - 1]
                                                )
                                                if points[i - 1] > mean_value:
                                                    # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                    points[i - 1] -= 2 * (
                                                        mean_value - points[i - 1]
                                                    )

                                    elif i > 1 and points[i - 2] < mean_value:

                                        if points_copy[i - 2] < mean_value:
                                            if points_copy[i - 1] > mean_value:
                                                # this indicates that neither points[i - 1] or points[i - 2] was flipped
                                                points[i - 1] = (
                                                    mean_value
                                                    + flip_strength
                                                    * (points[i - 1] - mean_value)
                                                )
                                                if points[i - 1] < mean_value:
                                                    # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                    points[i - 1] += 2 * (
                                                        mean_value - points[i - 1]
                                                    )
                                                    if points[i - 1] < mean_value:
                                                        # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                        points[i - 1] += 2 * (
                                                            mean_value - points[i - 1]
                                                        )
                                                        if points[i - 1] < mean_value:
                                                            points[i - 1] = (
                                                                points_copy[i - 1]
                                                                + (
                                                                    mean_value
                                                                    - points[i - 1]
                                                                )
                                                                + 1
                                                            )
                                            elif points_copy[i - 1] < mean_value:
                                                points[i - 2] = (
                                                    mean_value
                                                    + flip_strength
                                                    * (points[i - 2] - mean_value)
                                                )
                                                if points[i - 2] < mean_value:
                                                    # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                    points[i - 2] += 2 * (
                                                        mean_value - points[i - 2]
                                                    )
                                                    if points[i - 2] < mean_value:
                                                        # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                        points[i - 2] += 2 * (
                                                            mean_value - points[i - 2]
                                                        )
                                                        if points[i - 2] < mean_value:
                                                            points[i - 2] = (
                                                                points_copy[i - 2]
                                                                + (
                                                                    mean_value
                                                                    - points[i - 2]
                                                                )
                                                                + 1
                                                            )
                                        elif (
                                            points_copy[i - 2] > mean_value
                                            and points_copy[i - 1] < mean_value
                                        ):
                                            points[i - 2] = points_copy[i - 2]
                                            points[i - 1] = points_copy[i - 1]
                                        elif (
                                            points_copy[i - 2] > mean_value
                                            and points_copy[i - 1] > mean_value
                                        ):
                                            points[i - 1] = (
                                                mean_value
                                                + flip_strength
                                                * (points[i - 1] - mean_value)
                                            )
                                            if points[i - 1] < mean_value:
                                                # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                points[i - 1] += 2 * (
                                                    mean_value - points[i - 1]
                                                )
                                                if points[i - 1] < mean_value:
                                                    # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                    points[i - 1] += 2 * (
                                                        mean_value - points[i - 1]
                                                    )

                                if points[i] < mean_value:
                                    # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                    points[i] += 2 * (mean_value - points[i])
                            elif i < 13 and points[i + 1] > mean_value:
                                if i < 12 and points[i + 2] > mean_value:
                                    # this means we can do the flip because the next point is on the same side of the mean_value as this and the previous point, so when this is flipped, the graph will still be flipped

                                    points[i] = mean_value - flip_strength * offset
                                    if points[i] > mean_value:
                                        # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                        points[i] += 2 * (mean_value - points[i])
                                if points[i] > mean_value:
                                    # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                    points[i] += 2 * (mean_value - points[i])
                                else:
                                    # this means we are good
                                    print("Good flip")

                        elif points[i] < mean_value and points[i - 1] < mean_value:
                            # first, also check next point to ensure it is on the same side of the mean_value as this and the previous point so that we don't flip the graph. If it is, then do the flip. If it isn't, then after the flip, the graph will be flipped, so we need to do the opposite of the flip.
                            if i < 13 and points[i + 1] > mean_value:
                                if i < 12 and points[i + 2] > mean_value:

                                    if i > 1 and points[i - 2] > mean_value:

                                        # this means we need to do the opposite of the flip because the next point is on the opposite side of the mean_value as this and the previous point

                                        points[i] = mean_value - flip_strength * offset
                                        if points[i] < mean_value:
                                            # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                            points[i] += 2 * (mean_value - points[i])
                                            if points[i] < mean_value:
                                                # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                points[i] += 2 * (
                                                    mean_value - points[i]
                                                )
                                                if points[i] < mean_value:
                                                    points[i] = (
                                                        points_copy[i]
                                                        + (mean_value - points[i])
                                                        + 1
                                                    )
                                    elif i > 1 and points[i - 2] < mean_value:
                                        if points_copy[i - 1] > mean_value:
                                            points[i - 1] = points_copy[i - 1]
                                        elif points_copy[i - 1] < mean_value:
                                            points[i - 1] = (
                                                mean_value
                                                + flip_strength
                                                * (points[i - 1] - mean_value)
                                            )
                                            if points[i - 1] < mean_value:
                                                # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                points[i - 1] += 2 * (
                                                    mean_value - points[i - 1]
                                                )
                                                if points[i - 1] < mean_value:
                                                    # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                    points[i - 1] += 2 * (
                                                        mean_value - points[i - 1]
                                                    )
                                                    if points[i - 1] < mean_value:
                                                        points[i - 1] = (
                                                            points_copy[i - 1]
                                                            + (
                                                                mean_value
                                                                - points[i - 1]
                                                            )
                                                            + 1
                                                        )
                                elif i < 12 and points[i + 2] < mean_value:
                                    if i > 1 and points[i - 2] < mean_value:

                                        # if 2 points before this point is also above the mean_value, then we can should flip point[i - 1] instead of point[i]. However if points_copy[i - 1] or points_copy[i - 2] is below the mean_value, that means we are flipping the graph, so we should backtrack
                                        if points_copy[i - 2] < mean_value:
                                            if points_copy[i - 1] < mean_value:
                                                # this indicates that neither points[i - 1] or points[i - 2] was flipped
                                                points[i - 1] = (
                                                    mean_value
                                                    + flip_strength
                                                    * (points[i - 1] - mean_value)
                                                )
                                                if points[i - 1] < mean_value:
                                                    # this means we need to do either do another flip or do something else. We can just do point[i] - = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                    points[i - 1] += 2 * (
                                                        mean_value - points[i - 1]
                                                    )
                                                    if points[i - 1] < mean_value:
                                                        # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                        points[i - 1] += 2 * (
                                                            mean_value - points[i - 1]
                                                        )
                                                        if points[i - 1] < mean_value:
                                                            points[i - 1] = (
                                                                points_copy[i - 1]
                                                                + (
                                                                    points[i - 1]
                                                                    - mean_value
                                                                )
                                                                + 1
                                                            )
                                            elif points_copy[i - 1] < mean_value:
                                                points[i - 1] = points_copy[i - 1]
                                        elif (
                                            points_copy[i - 2] < mean_value
                                            and points_copy[i - 1] < mean_value
                                        ):
                                            points[i - 2] = points_copy[i - 2]
                                            points[i - 1] = (
                                                mean_value
                                                + flip_strength
                                                * (points[i - 1] - mean_value)
                                            )
                                            if points[i - 1] < mean_value:
                                                # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                points[i - 1] += 2 * (
                                                    mean_value - points[i - 1]
                                                )
                                                if points[i - 1] < mean_value:
                                                    # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                    points[i - 1] += 2 * (
                                                        mean_value - points[i - 1]
                                                    )
                                                    if points[i - 1] < mean_value:
                                                        points[i - 1] = (
                                                            points_copy[i - 1]
                                                            + (
                                                                mean_value
                                                                - points[i - 1]
                                                            )
                                                            + 1
                                                        )
                                        elif (
                                            points_copy[i - 2] < mean_value
                                            and points_copy[i - 1] > mean_value
                                        ):
                                            points[i - 2] = points_copy[i - 2]
                                            points[i - 1] = points_copy[i - 1]
                                    elif i > 1 and points[i - 2] > mean_value:

                                        if points_copy[i - 2] > mean_value:
                                            if points_copy[i - 1] < mean_value:
                                                # this indicates that neither points[i - 1] or points[i - 2] was flipped
                                                points[i - 1] = (
                                                    mean_value
                                                    + flip_strength
                                                    * (points[i - 1] - mean_value)
                                                )
                                                if points[i - 1] < mean_value:
                                                    # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                    points[i - 1] += 2 * (
                                                        mean_value - points[i - 1]
                                                    )
                                                    if points[i - 1] < mean_value:
                                                        # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                        points[i - 1] += 2 * (
                                                            mean_value - points[i - 1]
                                                        )
                                                        if points[i - 1] < mean_value:
                                                            points[i - 1] = (
                                                                points_copy[i - 1]
                                                                + (
                                                                    mean_value
                                                                    - points[i - 1]
                                                                )
                                                                + 1
                                                            )
                                                points[i - 2] = (
                                                    mean_value
                                                    - flip_strength
                                                    * (points[i - 2] - mean_value)
                                                )
                                                if points[i - 2] > mean_value:
                                                    # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                    points[i - 2] -= 2 * (
                                                        mean_value - points[i - 2]
                                                    )
                                                    if points[i - 2] > mean_value:
                                                        # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                        points[i - 2] -= 2 * (
                                                            mean_value - points[i - 2]
                                                        )
                                                        if points[i - 2] > mean_value:
                                                            points[i - 2] = (
                                                                points_copy[i - 2]
                                                                - (
                                                                    mean_value
                                                                    - points[i - 2]
                                                                )
                                                                + 1
                                                            )
                                            elif points_copy[i - 1] > mean_value:
                                                points[i - 1] = points_copy[i - 1]
                                                points[i - 2] = (
                                                    mean_value
                                                    - flip_strength
                                                    * (points[i - 2] - mean_value)
                                                )
                                                if points[i - 2] > mean_value:
                                                    # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                    points[i - 2] -= 2 * (
                                                        mean_value - points[i - 2]
                                                    )
                                                    if points[i - 2] > mean_value:
                                                        # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure
                                                        points[i - 2] -= 2 * (
                                                            mean_value - points[i - 2]
                                                        )
                                                        if points[i - 2] > mean_value:
                                                            points[i - 2] = (
                                                                points_copy[i - 2]
                                                                - (
                                                                    mean_value
                                                                    - points[i - 2]
                                                                )
                                                                + 1
                                                            )
                                        elif points_copy[i - 2] < mean_value:
                                            points[i - 2] = points_copy[i - 2]
                                            if points_copy[i - 1] > mean_value:
                                                points[i - 1] = points_copy[i - 1]
                                            elif points_copy[i - 1] < mean_value:
                                                points[i - 1] = (
                                                    mean_value
                                                    + flip_strength
                                                    * (points[i - 1] - mean_value)
                                                )
                                                if points[i - 1] < mean_value:
                                                    # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                    points[i - 1] += 2 * (
                                                        mean_value - points[i - 1]
                                                    )
                                                    if points[i - 1] < mean_value:
                                                        # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                        points[i - 1] += 2 * (
                                                            mean_value - points[i - 1]
                                                        )
                                                        if points[i - 1] < mean_value:
                                                            points[i - 1] = (
                                                                points_copy[i - 1]
                                                                + (
                                                                    mean_value
                                                                    - points[i - 1]
                                                                )
                                                                + 1
                                                            )
                            elif i < 13 and points[i + 1] < mean_value:
                                if i < 12 and points[i + 2] < mean_value:
                                    if i > 1 and points[i - 2] < mean_value:
                                        # this means we can do the flip because the next point is on the same side of the mean_value as this and the previous point, so when this is flipped, the graph will still be flipped
                                        points[i - 1] = (
                                            mean_value + flip_strength * offset
                                        )
                                        if points[i - 1] < mean_value:
                                            # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                            points[i - 1] += 2 * (
                                                mean_value - points[i - 1]
                                            )
                                            if points[i - 1] < mean_value:
                                                # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                points[i - 1] += 2 * (
                                                    mean_value - points[i - 1]
                                                )
                                                if points[i - 1] < mean_value:
                                                    points[i - 1] = (
                                                        points_copy[i - 1]
                                                        + (mean_value - points[i - 1])
                                                        + 1
                                                    )
                                    elif i > 1 and points[i - 2] > mean_value:
                                        # this means we need to do the opposite of the flip because the next point is on the opposite side of the mean_value as this and the previous point
                                        points[i] = mean_value - flip_strength * offset
                                        if points[i] > mean_value:
                                            # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                            points[i] -= 2 * (mean_value - points[i])
                                elif i < 12 and points[i + 2] > mean_value:
                                    if i > 1 and points[i - 2] > mean_value:
                                        # this means we can do the flip because the next point is on the same side of the mean_value as this and the previous point, so when this is flipped, the graph will still be flipped
                                        points[i] = mean_value + flip_strength * offset
                                        if points[i] < mean_value:
                                            # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                            points[i] += 2 * (mean_value - points[i])
                                            if points[i] < mean_value:
                                                # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                points[i] += 2 * (
                                                    mean_value - points[i]
                                                )
                                                if points[i] < mean_value:
                                                    points[i] = (
                                                        points_copy[i]
                                                        + (mean_value - points[i])
                                                        + 1
                                                    )
                                    elif i > 1 and points[i - 2] < mean_value:
                                        if points_copy[i - 2] > mean_value:
                                            points[i - 2] = points_copy[i - 2]
                                        elif points_copy[i - 2] < mean_value:
                                            points[i - 2] = (
                                                mean_value
                                                + flip_strength
                                                * (points[i - 2] - mean_value)
                                            )
                                            if points[i - 2] < mean_value:
                                                # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                points[i - 2] += 2 * (
                                                    mean_value - points[i - 2]
                                                )
                                                if points[i - 2] < mean_value:
                                                    # this means we need to do either do another flip or do something else. We can just do point[i] + = 2 * (mean_value - points[i]) to ensure it is on the correct side of the mean_value
                                                    points[i - 2] += 2 * (
                                                        mean_value - points[i - 2]
                                                    )
                                                    if points[i - 2] < mean_value:
                                                        points[i - 2] = (
                                                            points_copy[i - 2]
                                                            + (
                                                                mean_value
                                                                - points[i - 2]
                                                            )
                                                            + 1
                                                        )
                                            points[i] = mean_value - flip_strength * (
                                                points[i] - mean_value
                                            )
                                            if points[i] < mean_value:
                                                points[i] += 2 * (
                                                    mean_value - points[i]
                                                )
                                                if points[i] < mean_value:
                                                    points[i] += 2 * (
                                                        mean_value - points[i]
                                                    )
                                                    if points[i] < mean_value:
                                                        points[i] = (
                                                            points_copy[i]
                                                            + (mean_value - points[i])
                                                            + 1
                                                        )

                        if (points[i] > mean_value and points[i - 1] < mean_value) or (
                            points[i] < mean_value and points[i - 1] > mean_value
                        ):
                            # this means we are at a valley or peak, so we need to do nothing
                            print("Valley or peak, skipping flip")
                            pass
                        else:
                            print(f"Failed to flip point {i} AGAIN")
                            print(f"Current point: {points[i]} AGAIN")
                            print(f"Previous point: {points[i - 1]} AGAIN")
                            print(f"Mean value: {mean_value} AGAIN")
                            print(f"Max allowed: {max_allowed[i]} AGAIN")
                            print(f"Base value: {base_values[i]} AGAIN")
                            raise Exception(f"Failed to flip point {i} AGAIN")

                        if points[i] < base_values[i]:
                            points[i] = base_values[i]
                        elif points[i] > max_allowed[i]:
                            points[i] = max_allowed[i]
                    except Exception as e:
                        print(f"Error during flipping: {e}")
                        print(f"Failed to flip point {i}")
                        print(f"Current point: {points[i]}")
                        print(f"Previous point: {points[i - 1]}")
                        print(f"Mean value: {mean_value}")
                        print(f"Max allowed: {max_allowed[i]}")
                        print(f"Base value: {base_values[i]}")
                        print(f"Error: {e}")
                points[i] = min(max_allowed[i], points[i])
                points[i] = max(base_values[i] + 1, points[i])
        # compare points[i] with points_copy[i] and decide which one to keep
        stats = compare_arrays(points, points_copy)
        # print(stats[0])

        highest_std_dev = ""
        highest_variance = ""
        highest_min_max_range = ""
        spikiest = ""
        for key, value in stats[1].items():
            if isinstance(value, str):
                if key == "highest_std":
                    highest_std_dev = value
                elif key == "highest_var":
                    highest_variance = value
                elif key == "highest_minmax_range":
                    highest_min_max_range = value
                elif key == "spikiest":
                    spikiest = value

        # first, which one has a negative dependency, as in a point that is lower than the previous point, and the next point is higher than the previous point. This indicates a spike, so we want the spikiest one
        # Replace existing negative dependency check with:
        negative_dependencies_points = calculate_alternating_points(np.array(points))
        negative_dependencies_points_copy = calculate_alternating_points(
            np.array(points_copy)
        )
        winner = "points"

        # Choose based on number of valleys
        if (
            negative_dependencies_points
            and not negative_dependencies_points_copy
            and spikiest.lower().strip() == "array1"
        ):
            winner = "points"
        elif (
            negative_dependencies_points
            and not negative_dependencies_points_copy
            and spikiest.lower().strip() == "array2"
        ):
            winner = "points_copy"
        else:
            if negative_dependencies_points and not negative_dependencies_points_copy:
                winner = "points"
            elif negative_dependencies_points_copy and not negative_dependencies_points:
                winner = "points_copy"
            elif negative_dependencies_points and negative_dependencies_points_copy:
                # If equal valleys, use weighted scoring as fallback
                points_score = 0
                points_copy_score = 0
                if negative_dependencies_points > negative_dependencies_points_copy:
                    points_score += 2
                elif negative_dependencies_points < negative_dependencies_points_copy:
                    points_copy_score += 2
                winner = ""
                if highest_std_dev.lower().strip() == "array1":
                    points_score += 2
                elif highest_std_dev.lower().strip() == "array2":
                    points_copy_score += 2
                if highest_variance.lower().strip() == "array1":
                    points_score += 1
                elif highest_variance.lower().strip() == "array2":
                    points_copy_score += 1
                if highest_min_max_range.lower().strip() == "array1":
                    points_score += 1
                elif highest_min_max_range.lower().strip() == "array2":
                    points_copy_score += 1
                if spikiest.lower().strip() == "array1":
                    points_score += 1
                elif spikiest.lower().strip() == "array2":
                    points_copy_score += 1

                if points_score > points_copy_score:
                    winner = "points"
                elif points_copy_score > points_score:
                    winner = "points_copy"
                else:
                    points_score = 0
                    points_copy_score = 0
                    # spikiest is worth 4 points, highest_std is worth 3 points, highest_var is worth 2 points, highest_minmax_range is worth 1 point
                    if spikiest.lower().strip() == "array1":
                        points_score += 4
                    elif spikiest.lower().strip() == "array2":
                        points_copy_score += 4
                    if highest_std_dev.lower().strip() == "array1":
                        points_score += 3
                    elif highest_std_dev.lower().strip() == "array2":
                        points_copy_score += 3
                    if highest_variance.lower().strip() == "array1":
                        points_score += 2
                    elif highest_variance.lower().strip() == "array2":
                        points_copy_score += 2
                    if highest_min_max_range.lower().strip() == "array1":
                        points_score += 1
                    elif highest_min_max_range.lower().strip() == "array2":
                        points_copy_score += 1
                    if points_score > points_copy_score:
                        winner = "points"
                    elif points_copy_score > points_score:
                        winner = "points_copy"
                    else:
                        winner = "points"
                        # this means we have a tie, so we can just return the original points
        if winner == "points":
            old_points = points_copy
            points = points
        elif winner == "points_copy":
            old_points = points
            points = points_copy

        print(f"\n\nWinner: {winner}\n\n")

        for i in range(13, -1, -1):
            points[i] = min(max_allowed[i], points[i])
        # Ensure point 16 is correctly restored
        points[15] = final_value + base_values[15]
        for name, p in {
            "POSTDEPENDENCY_CHECKS_LOSER": old_points,
            "POSTDEPENDENCY_CHECKS_WINNER": points,
        }.items():

            try:
                # attempt to display the graph if final_value is one of the following: 69, 67, 40, 81, 54, 47, 35, 71, 3, 5, 0, 1, 15, 20, 5, 10
                x = list(range(1, 17))  # 16 time points
                # Create figure and plot
                fig = plt.figure(figsize=(4, 4))
                plt.fill_between(x, p, color=color, alpha=0.4)
                # plt.plot(x, p, marker="o", color=color, linewidth=2, markersize=6)

                # Style the graph
                # Style improvements
                # Style the graph
                plt.title(f"{resource} Usage")
                plt.ylim(0, 100)
                plt.grid(alpha=0.3)

                # Remove X-axis ticks and labels
                plt.xticks([])

                # Remove Y-axis label
                plt.gca().set_ylabel(None)

                # Adjust layout to remove outer borders
                plt.tight_layout(pad=0)
                filename = os.path.join(
                    output_dir,
                    f"{graph_num}_{resource}_{name}.png",
                )
                plt.savefig(
                    filename, dpi=150, bbox_inches="tight"
                )  # X resolution halved
                plt.close(fig)
            except Exception as e:
                print(
                    f"Failed to save graph for {resource}, graph_num {graph_num}, final_value {final_value},error: {e}"
                )
    elif resource == "Memory":
        points_original = points.copy()
        points = exp_smooth(points, alpha=0.3)
        points[12:] = points_original[12:]
        points[15] = final_value + base_values[15]
        # Control slopes with min/max bounds
        max_slope = 0.3  # Maximum allowed slope between points
        min_slope = 0.1  # Minimum allowed slope to prevent flat lines

        for i in range(len(points) - 2, 0, -1):
            if i < 13:
                current_slope = abs(points[i + 1] - points[i])
                if current_slope > max_slope or current_slope < min_slope:
                    direction = np.sign(points[i + 1] - points[i])
                    # Use max_slope if too steep, min_slope if too flat
                    target_slope = max_slope if current_slope > max_slope else min_slope
                    points[i] = points[i + 1] - (direction * target_slope)

        # Preserve endpoints
        points[0:5] = np.clip(
            points_original[0:5], points_original[6] / 2, points_original[6] / 2
        )
        points[-1] = points_original[-1]
        # add a sharp jump between points 4 and 6 by interpolating between them
        points[5] = np.interp(5, [4, 6], [points[4], points[6]])

        # Final clip
        for i in range(13, -1, -1):
            points[i] = min(max_allowed[i], points[i])

        try:
            # attempt to display the graph if final_value is one of the following: 69, 67, 40, 81, 54, 47, 35, 71, 3, 5, 0, 1, 15, 20, 5, 10
            x = list(range(1, 17))  # 16 time points

            # Create figure and plot
            fig = plt.figure(figsize=(4, 4))
            plt.fill_between(x, points, color=color, alpha=0.4)
            # plt.plot(x, points, marker="o", color=color, linewidth=2, markersize=6)

            # Style the graph
            # Style improvements
            # Style the graph
            plt.title(f"{resource} Usage")
            plt.ylim(0, 100)
            plt.grid(alpha=0.3)

            # Remove X-axis ticks and labels
            plt.xticks([])

            # Remove Y-axis label
            plt.gca().set_ylabel(None)

            # Adjust layout to remove outer borders
            plt.tight_layout(pad=0)
            filename = os.path.join(
                output_dir,
                f"{graph_num}_{resource}_POSTDEPENDENCY_CHECKS_.png",
            )
            plt.savefig(filename, dpi=150, bbox_inches="tight")  # X resolution halved
            plt.close(fig)
        except Exception as e:
            print(
                f"Failed to save graph for {resource}, graph_num {graph_num}, final_value {final_value},error: {e}"
            )
    # Final clip to ensure values stay in bounds
    for i in range(13, -1, -1):
        points[i] = min(max_allowed[i], points[i])
        points[i] = max(base_values[i] + 1, points[i])

    count = 0
    spike_results = spike_checker.check_spikiness(points, final_value, resource)

    while not all(spike_results[0]) and resource != "Memory" and count < 5:
        # Replace existing regeneration code:
        assert len(spike_results) == 2
        assert len(spike_results[0]) == 6
        points = spike_checker.apply_corrections(
            points, spike_results[0], spike_results[1], max_allowed
        )
        spike_results = spike_checker.check_spikiness(points, final_value, resource)
        count += 1
    points = enforce_upward_trend(points, resource)

    for i in range(13, -1, -1):
        points[i] = min(max_allowed[i], points[i])
        points[i] = max(base_values[i] + 1, points[i])
    # Ensure point 16 is correctly restored
    points[15] = final_value + base_values[15]
    return points


# Generate several versions for each graph, evaluate with check_spikiness, and pick the best
def generate_best_graph_points(final_value, base_values, resource):
    best_points = None
    best_score = -1
    for _ in range(3):  # Try 3 variations
        candidate_points = generate_graph_points_with_alternation(
            final_value, base_values, resource
        )
        std_dev = np.std(candidate_points[:15])  # Prioritize STD for "spikiness"
        if std_dev > best_score:
            best_points = candidate_points
            best_score = std_dev
    return best_points


# Generate and save graphs with the updated logic
# Store sample figures for later display
sample_figs = []


def interpolate_points(points):
    new_points = []
    for i in range(len(points) - 1):
        new_points.append(points[i])  # Add original point
        # Add interpolated point (average of current and next WITH VARIATION)
        if (points[i] + points[i + 1]) / 2 + np.random.uniform(
            points[i] * 0.1, points[i + 1] * 0.1
        ) > max(points[i], points[i + 1]):
            max_ = max(points[i], points[i + 1])
            min_ = min(points[i], points[i + 1])
            new_points.append(max_ - np.random.uniform(min_, max_ * 0.1))
        else:

            new_points.append(
                (points[i] + points[i + 1]) / 2
                + np.random.uniform(points[i] * 0.1, points[i + 1] * 0.1)
            )  # Add interpolated point
    new_points.append(points[-1])  # Add final point
    return new_points


for i, (cpu, memory, gpu, disk) in enumerate(combinations, start=1):
    for resource, final_value, color in zip(
        ["CPU", "Memory", "GPU", "Disk"],
        [cpu, memory, gpu, disk],
        [colors["CPU"], colors["Memory"], colors["GPU"], colors["Disk"]],
    ):
        # Generate the best points for the graph
        graph_points = generate_graph_points_with_alternation(
            final_value, base_values[resource], resource, i
        )

        x = list(range(1, 17))  # 16 time points

        # Create figure and plot
        fig = plt.figure(figsize=(4, 4))
        plt.fill_between(x, graph_points, color=color, alpha=0.4)

        # Style the graph
        # Style the graph
        plt.title(f"{resource} Usage")
        plt.ylim(0, 100)
        plt.grid(alpha=0.3)

        # Remove X-axis ticks and labels
        plt.xticks([])

        # Remove Y-axis label
        plt.gca().set_ylabel(None)

        # Adjust layout to remove outer borders
        plt.tight_layout(pad=0)
        # Save the graph
        filename = os.path.join(output_dir, f"{i}_{resource}_FINAL.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")  # X resolution halved
        plt.close()

        # Replace in the main code:
        x = list(range(1, 32))  # 32 time points instead of 16
        graph_points = interpolate_points(graph_points)
        if final_value < 3:
            final_value *= 2
        if final_value < 20 and resource in ["CPU", "GPU"]:
            # # add another sinusoidal wave to the graph
            # graph_points = [
            #     x + 5 * np.sin(2 * np.pi * (i / 16)) for i, x in enumerate(graph_points)
            # ]
            # Add a few spikes to the graph, jumping to final_value plus or minus between 1 and 5
            spikes = []
            for _ in range(random.randint(1, 6)):
                spike_idx = random.randint(0, 28)
                while (
                    spike_idx in spikes
                ):  # Ensure dip is not at the same point as a spike
                    spike_idx = random.randint(0, 28)
                spikes.append(spike_idx)
                spike_val = random.uniform(-1, 3)
                if graph_points[spike_idx] + spike_val >= 0:
                    graph_points[spike_idx] += spike_val + final_value
                    # bring the points on either side of the spike down to a dip
                    dip_idx = spike_idx - 1 if spike_idx > 1 else 0
                    if dip_idx in spikes:
                        pass
                    else:
                        graph_points[dip_idx] = random.uniform(1, 5)
                    dip_idx = spike_idx + 1 if spike_idx < 28 else 28
                    if dip_idx in spikes:
                        pass
                    else:
                        graph_points[dip_idx] = random.uniform(1, 5)
                else:
                    graph_points[spike_idx] = final_value + random.uniform(1, 5)

            for _ in range(random.randint(1, 6)):
                dip_idx = random.randint(0, 24)
                while (
                    dip_idx in spikes
                ):  # Ensure dip is not at the same point as a spike
                    dip_idx = random.randint(0, 24)
                dip_val = random.uniform(1, 3)
                if dip_val >= 0:
                    graph_points[dip_idx] = dip_val
                else:
                    dip_val = 1
                    graph_points[dip_idx] = dip_val
                # bring the points on either side of the dip up to a peak
                peak_idx = dip_idx - 1 if dip_idx > 1 else 0
                if peak_idx in spikes:
                    pass
                else:
                    graph_points[peak_idx] = dip_val + random.uniform(1, final_value)
                peak_idx = dip_idx + 1 if dip_idx < 24 else 24
                if peak_idx in spikes:
                    pass
                else:
                    graph_points[peak_idx] = dip_val + random.uniform(1, final_value)
            # spike_checker = SpikinessFixer()
            # graph_points = spike_checker.apply_corrections(
            #     graph_points,
            #     *spike_checker.check_spikiness(graph_points, final_value, resource),
            # )
        # Create figure and plot
        fig = plt.figure(figsize=(4, 4))
        plt.fill_between(x, graph_points, color=color, alpha=0.4)

        # Style the graph
        plt.title(f"{resource} Usage")
        plt.ylim(0, 100)
        plt.grid(alpha=0.3)

        # Remove X-axis ticks and labels
        plt.xticks([])

        # Remove Y-axis label
        plt.gca().set_ylabel(None)

        # Adjust layout to remove outer borders
        plt.tight_layout(pad=0)

        # Save the graph
        filename = os.path.join(output_dir, f"{i}_{resource}_FINAL_INTERPOLATED.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")  # X resolution halved
        plt.close()

        # Double the points with interpolation
        # Update progress
        pbar_main.update(1)
        pbar_main.set_postfix(
            {"Combo": f"{i}/{total_combinations}", "Resource": resource}
        )

pbar_main.close()


def display_graphs(sample_figs, resources, output_dir):
    plt.close("all")  # Clear existing plots

    # Create figure list

    # Generate one figure per combination
    for idx, fig in enumerate(sample_figs):
        # fig.clf()
        plt.close(fig)
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()
        for ax, resource in zip(axs, resources):
            ax.fill_between(
                range(1, 17), graph_points, color=colors[resource], alpha=0.4
            )
            ax.set_title(f"{resource} Usage")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{resource} Usage (%)")
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.3)
        fig.suptitle(f"Combination {idx+1}")

        plt.tight_layout()
    plt.savefig(
        filename, dpi=150, bbox_inches="tight"
    )  # X resolution halved    # Display plots
    plt.ion()
    for fig in sample_figs:
        fig.show()
        plt.pause(0.5)

    plt.ioff()
    input("Press Enter to continue...")
    return sample_figs


# Replace existing display code with:
resources = ["CPU", "Memory", "GPU", "Disk"]
# display_graphs(sample_figs, resources, output_dir)
# Zip the folder containing all updated graphs
try:
    if os.path.exists(
        os.path.join(output_dir, "resource_graphs_final_alternation.zip")
    ):
        dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(
            os.path.join(output_dir, "resource_graphs_final_alternation.zip"),
            os.path.join(
                output_dir, f"resource_graphs_final_alternation_old_{dtime}.zip"
            ),
        )
    shutil.make_archive("resource_graphs_final_alternation", "zip", output_dir)
except Exception as e:
    print(f"Error zipping the folder: {e}")

print("Graphs generated and saved successfully.")
