"""
Multi-Armed Bandit Implementation for A/B Testing.

This module implements Epsilon-Greedy and Thompson Sampling algorithms for
solving the multi-armed bandit problem in advertisement selection. It also
provides utilities to visualize cumulative rewards and regrets and to export
experiment results as CSV files.

The reward model here uses a Gaussian observation model where each arm `i`
returns a reward distributed as N(p_i, 1/precision). For Epsilon-Greedy,
`precision` is implicitly 1 (unit variance); for Thompson Sampling, it is
configurable per bandit instance.

Notes
-----
- The abstract base `Bandit` defines the required interface for bandit arms.
- `Visualization` gathers plotting helper methods (saved as PNG files).
- CSV outputs:
    * `epsilon_greedy_results.csv`
    * `thompson_sampling_results.csv`
    * `combined_results.csv`
- Figures:
    * `Epsilon-Greedy_performance.png`
    * `Thompson-Sampling_performance.png`
    * `algorithm_comparison.png`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple
from pathlib import Path  # <-- added

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# --- New: centralized output folders ---
IMG_DIR = Path("img")
REPORT_DIR = Path("report")
def _ensure_output_dirs() -> None:
    """Create output directories if missing."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


class Bandit(ABC):
    """Abstract interface for a single-armed bandit.

    Parameters
    ----------
    p : float
        Latent mean reward of the arm (e.g., a click-through uplift or true mean).

    Notes
    -----
    Implementations must (minimally) track:
    - `p` (true mean used by the simulator),
    - an internal estimate of the mean (algorithm-specific),
    - `N` number of pulls,
    - `reward_history` (list of observed rewards).
    """

    @abstractmethod
    def __init__(self, p: float) -> None:
        """Initialize the bandit with its latent mean `p`."""
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """String representation for debugging/logging."""
        raise NotImplementedError

    @abstractmethod
    def pull(self) -> float:
        """Simulate pulling the arm and return an observed reward.

        Returns
        -------
        float
            The sampled reward from the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, x: float) -> None:
        """Update internal posterior/estimate using observed reward.

        Parameters
        ----------
        x : float
            Observed reward.
        """
        raise NotImplementedError

    @abstractmethod
    def experiment(
        self, bandits: Sequence["Bandit"], num_trials: int
    ) -> Tuple[List[float], List[int]]:
        """Run the selection policy across `num_trials`.

        Parameters
        ----------
        bandits : Sequence[Bandit]
            Collection of candidate arms among which the policy selects.
        num_trials : int
            Number of iterations (pulls) to simulate.

        Returns
        -------
        rewards : list of float
            Reward observed at each trial.
        selected_bandits : list of int
            Index of the chosen arm at each trial.
        """
        raise NotImplementedError

    @abstractmethod
    def report(
        self,
        rewards: Sequence[float],
        selected_bandits: Sequence[int],
        bandits: Sequence["Bandit"],
        bandit_rewards: Sequence[float],
    ) -> None:
        """Print a summary and persist per-trial results.

        Parameters
        ----------
        rewards : Sequence[float]
            Per-trial observed rewards for this run.
        selected_bandits : Sequence[int]
            Per-trial chosen arm indices.
        bandits : Sequence[Bandit]
            The arms used in the experiment (for counts and display).
        bandit_rewards : Sequence[float]
            Ground-truth means for each arm (used to compute regret).
        """
        raise NotImplementedError


class Visualization:
    """Collection of plotting utilities for bandit experiments."""

    def plot1(self, bandits: Sequence[Bandit], algorithm_name: str) -> None:
        """Plot per-arm cumulative average rewards (linear and log-x scales).

        For each arm, the cumulative average reward over trials is shown. Two
        subplots are saved as a single image: one with a linear x-axis and one
        with a logarithmic x-axis.

        Parameters
        ----------
        bandits : Sequence[Bandit]
            Arms that were run (each must have `reward_history` populated).
        algorithm_name : str
            Name used in titles and filename (e.g., "Epsilon-Greedy").
        """
        _ensure_output_dirs()  # ensure folders exist
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        for i, bandit in enumerate(bandits):
            rewards = np.asarray(bandit.reward_history, dtype=float)
            if rewards.size == 0:
                continue
            cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
            ax1.plot(cumulative_avg, label=f"Bandit {i+1} (p={bandit.p})")

        ax1.set_xlabel("Trial Number")
        ax1.set_ylabel("Average Reward")
        ax1.set_title(f"{algorithm_name} - Average Reward (Linear Scale)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for i, bandit in enumerate(bandits):
            rewards = np.asarray(bandit.reward_history, dtype=float)
            if rewards.size == 0:
                continue
            cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
            ax2.plot(cumulative_avg, label=f"Bandit {i+1} (p={bandit.p})")

        ax2.set_xlabel("Trial Number")
        ax2.set_ylabel("Average Reward")
        ax2.set_title(f"{algorithm_name} - Average Reward (Log Scale)")
        ax2.set_xscale("log")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out = IMG_DIR / f"{algorithm_name}_performance.png"  # changed path
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Performance plot saved to {out}")

    def plot2(
        self,
        eg_rewards: Sequence[float],
        ts_rewards: Sequence[float],
        eg_regrets: Sequence[float],
        ts_regrets: Sequence[float],
    ) -> None:
        """Compare algorithms via cumulative rewards and regrets.

        Parameters
        ----------
        eg_rewards : Sequence[float]
            Cumulative rewards over trials for Epsilon-Greedy.
        ts_rewards : Sequence[float]
            Cumulative rewards over trials for Thompson Sampling.
        eg_regrets : Sequence[float]
            Cumulative regrets over trials for Epsilon-Greedy.
        ts_regrets : Sequence[float]
            Cumulative regrets over trials for Thompson Sampling.
        """
        _ensure_output_dirs()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(eg_rewards, label="Epsilon-Greedy", alpha=0.8)
        ax1.plot(ts_rewards, label="Thompson Sampling", alpha=0.8)
        ax1.set_xlabel("Trial Number")
        ax1.set_ylabel("Cumulative Reward")
        ax1.set_title("Cumulative Rewards Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(eg_regrets, label="Epsilon-Greedy", alpha=0.8)
        ax2.plot(ts_regrets, label="Thompson Sampling", alpha=0.8)
        ax2.set_xlabel("Trial Number")
        ax2.set_ylabel("Cumulative Regret")
        ax2.set_title("Cumulative Regrets Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out = IMG_DIR / "algorithm_comparison.png"  # changed path
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Comparison plot saved to {out}")


class EpsilonGreedy(Bandit):
    """Epsilon-Greedy bandit (decaying epsilon = `initial_epsilon` / t).

    Parameters
    ----------
    p : float
        True mean reward of this arm.
    initial_epsilon : float, default=1.0
        Initial exploration rate; epsilon at trial `t` is `initial_epsilon / t`.

    Attributes
    ----------
    p_estimate : float
        Running estimate of the arm's mean reward (sample average).
    N : int
        Number of times this arm was selected.
    reward_history : list of float
        Observed rewards for this arm (appended when the arm is chosen).
    """

    def __init__(self, p: float, initial_epsilon: float = 1.0) -> None:
        self.p = float(p)
        self.p_estimate = 0.0
        self.N = 0
        self.initial_epsilon = float(initial_epsilon)
        self.reward_history: List[float] = []
        logger.debug(f"EpsilonGreedy bandit initialized with p={p}")

    def __repr__(self) -> str:
        return f"EpsilonGreedy(p={self.p:.3f}, estimate={self.p_estimate:.3f}, N={self.N})"

    def pull(self) -> float:
        """Sample reward from N(p, 1).

        Returns
        -------
        float
            Observed reward.
        """
        reward = float(np.random.randn() + self.p)
        logger.debug(f"Pulled arm with p={self.p}, got reward={reward:.3f}")
        return reward

    def update(self, x: float) -> None:
        """Update sample-mean estimate with the new observation.

        Parameters
        ----------
        x : float
            Observed reward.
        """
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
        logger.debug(f"Updated estimate to {self.p_estimate:.3f} after {self.N} pulls")

    def experiment(
        self, bandits: Sequence[Bandit], num_trials: int
    ) -> Tuple[List[float], List[int]]:
        """Run Epsilon-Greedy selection across all arms.

        The policy explores with probability `epsilon_t = initial_epsilon / t`,
        otherwise exploits the arm with the highest current estimate.

        Parameters
        ----------
        bandits : Sequence[Bandit]
            Arms among which we choose each step.
        num_trials : int
            Number of pulls.

        Returns
        -------
        rewards : list of float
            Per-trial observed rewards (from the chosen arms).
        selected_bandits : list of int
            Indices of chosen arms per trial.
        """
        logger.info(f"Starting Epsilon-Greedy experiment with {num_trials} trials")
        rewards: List[float] = []
        selected_bandits: List[int] = []

        for t in range(1, num_trials + 1):
            epsilon = self.initial_epsilon / t
            if np.random.random() < epsilon:
                chosen_idx = int(np.random.choice(len(bandits)))
                chosen_bandit = bandits[chosen_idx]
                logger.debug(f"Trial {t}: Exploring (epsilon={epsilon:.4f}) -> arm {chosen_idx}")
            else:
                chosen_idx = int(np.argmax([b.p_estimate for b in bandits]))
                chosen_bandit = bandits[chosen_idx]
                logger.debug(f"Trial {t}: Exploiting -> arm {chosen_idx}")

            reward = chosen_bandit.pull()
            chosen_bandit.update(reward)
            chosen_bandit.reward_history.append(reward)

            rewards.append(reward)
            selected_bandits.append(chosen_idx)

            if t % 5000 == 0:
                logger.info(f"Completed {t}/{num_trials} trials")

        logger.info("Epsilon-Greedy experiment completed")
        return rewards, selected_bandits

    def report(
        self,
        rewards: Sequence[float],
        selected_bandits: Sequence[int],
        bandits: Sequence[Bandit],
        bandit_rewards: Sequence[float],
    ) -> None:
        """Print summary stats and save per-trial CSV for Epsilon-Greedy.

        Parameters
        ----------
        rewards : Sequence[float]
            Per-trial observed rewards.
        selected_bandits : Sequence[int]
            Indices of chosen arms.
        bandits : Sequence[Bandit]
            Arms used in the experiment.
        bandit_rewards : Sequence[float]
            Ground-truth means for each arm (used to compute regret).
        """
        _ensure_output_dirs()
        logger.info("Generating Epsilon-Greedy report")

        cumulative_reward = float(np.sum(rewards))
        avg_reward = float(np.mean(rewards))

        optimal_reward = float(np.max(bandit_rewards))
        regret = [optimal_reward - float(bandit_rewards[i]) for i in selected_bandits]
        cumulative_regret = float(np.sum(regret))
        avg_regret = float(np.mean(regret))

        df = pd.DataFrame(
            {
                "Bandit": [f"Bandit_{i+1}" for i in selected_bandits],
                "Reward": rewards,
                "Algorithm": "EpsilonGreedy",
            }
        )
        # --- changed path: CSV to ./report
        csv_path = REPORT_DIR / "epsilon_greedy_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

        # --- NEW: also write the printed summary to a text file in ./report
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("EPSILON-GREEDY ALGORITHM RESULTS")
        lines.append("=" * 60)
        lines.append(f"Total Trials: {len(rewards)}")
        lines.append(f"Cumulative Reward: {cumulative_reward:.2f}")
        lines.append(f"Average Reward: {avg_reward:.4f}")
        lines.append(f"Cumulative Regret: {cumulative_regret:.2f}")
        lines.append(f"Average Regret: {avg_regret:.4f}")
        lines.append("\nBandit Selection Counts:")
        for i, bandit in enumerate(bandits):
            count = selected_bandits.count(i)
            percentage = (count / max(1, len(selected_bandits))) * 100
            lines.append(f"  Bandit {i+1} (p={bandit.p}): {count} times ({percentage:.2f}%)")
        lines.append("=" * 60 + "\n")

        # Print to console (unchanged behavior)
        print("\n".join(lines))

        # Save to txt
        txt_path = REPORT_DIR / "epsilon_greedy_report.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Text report saved to {txt_path}")

        logger.info(f"Average Reward: {avg_reward:.4f}, Average Regret: {avg_regret:.4f}")


class ThompsonSampling(Bandit):
    """Thompson Sampling bandit with Beta-Bernoulli pseudo-update on sign.

    Parameters
    ----------
    p : float
        True mean reward of this arm.
    precision : float, default=1.0
        Precision (inverse variance) of the Gaussian observation model.

    Notes
    -----
    This implementation samples an arm via Beta(alpha, beta) per step
    using a simple sign-based surrogate to increment `alpha` (if reward > 0)
    or `beta` (if reward <= 0). This keeps the algorithm illustrative while
    using Gaussian rewards. For a fully consistent Gaussian-TS, one would
    maintain Normal-Gamma (or Normal-Inverse-Gamma) posteriors.
    """

    def __init__(self, p: float, precision: float = 1.0) -> None:
        self.p = float(p)
        self.precision = float(precision)
        self.alpha = 1  # Prior successes
        self.beta_param = 1  # Prior failures
        self.N = 0
        self.reward_history: List[float] = []
        self.mean_estimate = 0.0
        logger.debug(f"ThompsonSampling bandit initialized with p={p}, precision={precision}")

    def __repr__(self) -> str:
        return (
            f"ThompsonSampling(p={self.p:.3f}, alpha={self.alpha}, "
            f"beta={self.beta_param}, N={self.N})"
        )

    def pull(self) -> float:
        """Sample reward from N(p, 1/precision).

        Returns
        -------
        float
            Observed reward.
        """
        reward = float(np.random.randn() / np.sqrt(self.precision) + self.p)
        logger.debug(f"Pulled arm with p={self.p}, got reward={reward:.3f}")
        return reward

    def sample(self) -> float:
        """Draw a Thompson sample from the Beta posterior.

        Returns
        -------
        float
            A single Beta(alpha, beta) sample used for arm selection.
        """
        return float(np.random.beta(self.alpha, self.beta_param))

    def update(self, x: float) -> None:
        """Update internal state given an observation.

        Parameters
        ----------
        x : float
            Observed reward.
        """
        self.N += 1
        self.mean_estimate = ((self.N - 1) * self.mean_estimate + x) / self.N

        if x > 0:
            self.alpha += 1
        else:
            self.beta_param += 1

        logger.debug(f"Updated parameters: alpha={self.alpha}, beta={self.beta_param}")

    def experiment(
        self, bandits: Sequence[Bandit], num_trials: int
    ) -> Tuple[List[float], List[int]]:
        """Run Thompson Sampling across all arms.

        At each step, sample one value from each arm's Beta posterior and pick
        the arm with the highest sample.

        Parameters
        ----------
        bandits : Sequence[Bandit]
            Arms among which we choose each step.
        num_trials : int
            Number of pulls.

        Returns
        -------
        rewards : list of float
            Per-trial observed rewards.
        selected_bandits : list of int
            Indices of chosen arms per trial.
        """
        logger.info(f"Starting Thompson Sampling experiment with {num_trials} trials")
        rewards: List[float] = []
        selected_bandits: List[int] = []

        for t in range(1, num_trials + 1):
            samples = [b.sample() for b in bandits]
            chosen_idx = int(np.argmax(samples))
            chosen_bandit = bandits[chosen_idx]

            logger.debug(f"Trial {t}: Samples={samples}, chose arm {chosen_idx}")

            reward = chosen_bandit.pull()
            chosen_bandit.update(reward)
            chosen_bandit.reward_history.append(reward)

            rewards.append(reward)
            selected_bandits.append(chosen_idx)

            if t % 5000 == 0:
                logger.info(f"Completed {t}/{num_trials} trials")

        logger.info("Thompson Sampling experiment completed")
        return rewards, selected_bandits

    def report(
        self,
        rewards: Sequence[float],
        selected_bandits: Sequence[int],
        bandits: Sequence[Bandit],
        bandit_rewards: Sequence[float],
    ) -> None:
        """Print summary stats and save per-trial CSV for Thompson Sampling.

        Parameters
        ----------
        rewards : Sequence[float]
            Per-trial observed rewards.
        selected_bandits : Sequence[int]
            Indices of chosen arms.
        bandits : Sequence[Bandit]
            Arms used in the experiment.
        bandit_rewards : Sequence[float]
            Ground-truth means for each arm (used to compute regret).
        """
        _ensure_output_dirs()
        logger.info("Generating Thompson Sampling report")

        cumulative_reward = float(np.sum(rewards))
        avg_reward = float(np.mean(rewards))

        optimal_reward = float(np.max(bandit_rewards))
        regret = [optimal_reward - float(bandit_rewards[i]) for i in selected_bandits]
        cumulative_regret = float(np.sum(regret))
        avg_regret = float(np.mean(regret))

        df = pd.DataFrame(
            {
                "Bandit": [f"Bandit_{i+1}" for i in selected_bandits],
                "Reward": rewards,
                "Algorithm": "ThompsonSampling",
            }
        )
        # --- changed path: CSV to ./report
        csv_path = REPORT_DIR / "thompson_sampling_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

        # --- NEW: also write the printed summary to a text file in ./report
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("THOMPSON SAMPLING ALGORITHM RESULTS")
        lines.append("=" * 60)
        lines.append(f"Total Trials: {len(rewards)}")
        lines.append(f"Cumulative Reward: {cumulative_reward:.2f}")
        lines.append(f"Average Reward: {avg_reward:.4f}")
        lines.append(f"Cumulative Regret: {cumulative_regret:.2f}")
        lines.append(f"Average Regret: {avg_regret:.4f}")
        lines.append("\nBandit Selection Counts:")
        for i, bandit in enumerate(bandits):
            count = selected_bandits.count(i)
            percentage = (count / max(1, len(selected_bandits))) * 100
            lines.append(f"  Bandit {i+1} (p={bandit.p}): {count} times ({percentage:.2f}%)")
        lines.append("=" * 60 + "\n")

        # Print to console (unchanged behavior)
        print("\n".join(lines))

        # Save to txt
        txt_path = REPORT_DIR / "thompson_sampling_report.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Text report saved to {txt_path}")

        logger.info(f"Average Reward: {avg_reward:.4f}, Average Regret: {avg_regret:.4f}")


def comparison(
    eg_rewards: Sequence[float],
    ts_rewards: Sequence[float],
    eg_selected: Sequence[int],
    ts_selected: Sequence[int],
    bandit_rewards: Sequence[float],
) -> None:
    """Create comparison plot and combined CSV for both algorithms.

    Parameters
    ----------
    eg_rewards : Sequence[float]
        Per-trial rewards from Epsilon-Greedy.
    ts_rewards : Sequence[float]
        Per-trial rewards from Thompson Sampling.
    eg_selected : Sequence[int]
        Indices of arms chosen by Epsilon-Greedy per trial.
    ts_selected : Sequence[int]
        Indices of arms chosen by Thompson Sampling per trial.
    bandit_rewards : Sequence[float]
        Ground-truth means for each arm (used to compute the regret baselines).

    Notes
    -----
    Saves:
    - `algorithm_comparison.png` (cumulative reward & regret curves)
    - `combined_results.csv` (row-wise concat of both per-trial CSVs)
    """
    _ensure_output_dirs()
    logger.info("Comparing algorithm performances")

    eg_cumulative = np.cumsum(eg_rewards)
    ts_cumulative = np.cumsum(ts_rewards)

    optimal_reward = float(np.max(bandit_rewards))
    eg_regret = np.cumsum([optimal_reward - float(bandit_rewards[i]) for i in eg_selected])
    ts_regret = np.cumsum([optimal_reward - float(bandit_rewards[i]) for i in ts_selected])

    viz = Visualization()
    viz.plot2(eg_cumulative, ts_cumulative, eg_regret, ts_regret)

    # --- changed path: combined CSV to ./report
    df_combined = pd.concat(
        [
            pd.read_csv(REPORT_DIR / "epsilon_greedy_results.csv"),
            pd.read_csv(REPORT_DIR / "thompson_sampling_results.csv"),
        ],
        ignore_index=True,
    )
    comb_path = REPORT_DIR / "combined_results.csv"
    df_combined.to_csv(comb_path, index=False)
    logger.info(f"Combined results saved to {comb_path}")


if __name__ == "__main__":
    _ensure_output_dirs()  # make sure folders exist before running
    logger.info("Starting A/B Testing Experiment")

    BANDIT_REWARDS = [1, 2, 3, 4]
    NUM_TRIALS = 20000
    EPSILON = 1.0
    PRECISION = 1.0

    logger.info(f"Configuration: Rewards={BANDIT_REWARDS}, Trials={NUM_TRIALS}")

    logger.info("=" * 60)
    logger.info("STARTING EPSILON-GREEDY EXPERIMENT")
    logger.info("=" * 60)

    eg_bandits = [EpsilonGreedy(p, initial_epsilon=EPSILON) for p in BANDIT_REWARDS]
    eg_rewards, eg_selected = eg_bandits[0].experiment(eg_bandits, NUM_TRIALS)
    eg_bandits[0].report(eg_rewards, eg_selected, eg_bandits, BANDIT_REWARDS)

    viz = Visualization()
    viz.plot1(eg_bandits, "Epsilon-Greedy")

    logger.info("=" * 60)
    logger.info("STARTING THOMPSON SAMPLING EXPERIMENT")
    logger.info("=" * 60)

    ts_bandits = [ThompsonSampling(p, precision=PRECISION) for p in BANDIT_REWARDS]
    ts_rewards, ts_selected = ts_bandits[0].experiment(ts_bandits, NUM_TRIALS)
    ts_bandits[0].report(ts_rewards, ts_selected, ts_bandits, BANDIT_REWARDS)

    viz.plot1(ts_bandits, "Thompson-Sampling")

    logger.info("=" * 60)
    logger.info("GENERATING COMPARISON")
    logger.info("=" * 60)

    comparison(eg_rewards, ts_rewards, eg_selected, ts_selected, BANDIT_REWARDS)

    logger.info("Experiment completed successfully!")
    logger.success("All results saved and visualized")
