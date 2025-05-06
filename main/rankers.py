
from enum import Enum, IntEnum
import abc
import typing
import trueskill
import logging
import scipy
import numpy as np
# Choice constants
class UserChoice(IntEnum):
    NO_CHOICE = 0
    LHS = 1
    RHS = 2
    TIE = 3

class RankingAlgorithms(Enum):
    WIN_RATE = "win_rate"
    ELO = "elo"
    TRUE_SKILL = "true_skill"
    EIGEN = "eigenvector_centrality"
    BRADLEY_TERRY = "bradley_terry"


class RankersFactory:

    def get(self, algo):
        match algo:
            case RankingAlgorithms.WIN_RATE:
                return WinRateRanker()
            case RankingAlgorithms.ELO:
                return EloRanker()
            case RankingAlgorithms.TRUE_SKILL:
                return TrueSkillRanker()
            case RankingAlgorithms.EIGEN:
                return EigenvectorCentralityRanker()
            case RankingAlgorithms.BRADLEY_TERRY:
                return BradleyTerryRanker()


# Base ranker
class BaseRanker(abc.ABC):
    """Base class for rankers."""

    def __init__(self) -> None:
        """Initialize the ranker."""
        super().__init__()

        self.principles: list = []
        self.principles_params: dict = {}

        self.rating: dict = {}
        self.ranking: dict = {}

    @abc.abstractmethod
    def _create_principle(self) -> typing.Any:
        """Create a default principle."""

    def add_principle(self, principle: str) -> None:
        """Add a principle to the ranker.

        Args:
            principle (str): principle to add.
        """

        if principle in self.principles:
            raise ValueError(f"Principle '{principle}' already exists.")

        self.principles.append(principle)
        self.principles_params[principle] = self._create_principle()

    def add_principles(self, principles: list[str]) -> None:
        """Add principles to the ranker.

        Args:
            principles (list[str]): principles to add.
        """

        for principle in principles:
            self.add_principle(principle)

    @abc.abstractmethod
    def _update_principles(self, upvoted: str, downvoted: str, is_tie: bool) -> None:
        """Update the principles after a comparison.

        Args:
            upvoted (str): principle upvoted.
            downvoted (str): principle downvoted.
            is_tie (bool): whether the comparison is a tie.
        """

    def add_comparison(self, upvoted: str, downvoted: str, is_tie: bool = False, update_ranking: bool = True) -> None:
        """Add a comparison to the ranker.

        Args:
            upvoted (str): principle upvoted.
            downvoted (str): principle downvoted.
            is_tie (bool, optional): whether the comparison is a tie. Defaults to False.
            update_ranking (bool, optional): whether to update the ranking. Defaults to True.
        """

        if upvoted not in self.principles:
            raise ValueError(f"Principle '{upvoted}' does not exist.")
        if downvoted not in self.principles:
            raise ValueError(f"Principle '{downvoted}' does not exist.")

        self._update_principles(upvoted=upvoted, downvoted=downvoted, is_tie=is_tie)

        if update_ranking:
            self._update_rating()
            self._update_ranking()

    def add_comparisons(self, comparisons: list[tuple[str, str, bool]], update_ranking: bool = True) -> None:
        """Add comparisons to the ranker.

        Args:
            comparisons (list[tuple[str, str, bool]]): comparisons to add.
            update_ranking (bool, optional): whether to update the ranking. Defaults to True.
        """

        for upvoted, downvoted, is_tie in comparisons:
            self.add_comparison(upvoted, downvoted, is_tie, update_ranking=False)

        if update_ranking:
            self._update_rating()
            self._update_ranking()

    @abc.abstractmethod
    def _update_rating(self) -> None:
        """Update the rating."""

    def _update_ranking(self) -> None:
        """Update the ranking."""

        self.ranking = {
            principle: len(self.principles) - rank + 1
            for principle, rank in zip(
                list(self.rating.keys()),
                scipy.stats.rankdata(list(self.rating.values()), method="max"),
            )
        }

    @abc.abstractmethod
    def get_result(self) -> dict:
        """Get the result of the algorithm.

        Returns:
            dict: result of the algorithm.
        """

    def compute(self, votes):
        logging.warning("Computing %s on %d votes", self.ranker_name, len(votes))
        principle_ids = set([str(x["lhs_id"]) for x in votes])
        principle_ids.update([str(x["rhs_id"]) for x in votes])
        self.add_principles(principle_ids)
        self.add_comparisons([(str(x["rhs_id"]) if x["choice"] == UserChoice.RHS.value else str(x["lhs_id"]),
                               str(x["lhs_id"]) if x["choice"] == UserChoice.RHS.value else str(x["rhs_id"]),
                               x["choice"] == UserChoice.TIE.value) for x in votes
                              if x["choice"] != UserChoice.NO_CHOICE.value])
        ids = sorted(self.principles, key=lambda principle: self.ranking[principle])
        return [int(x) for x in ids]




class TrueSkillRanker(BaseRanker):
    """Trueskill ranker."""
    ranker_name = "true_skill"

    def __init__(self, mu0: float = 25, sigma0: float = 25 / 3) -> None:
        """Initialize the ranker.

        Args:
            mu0 (float, optional): initial mu. Defaults to 25.
            sigma0 (float, optional): initial sigma. Defaults to 25/3.
        """
        super().__init__()

        self.mu0 = mu0
        self.sigma0 = sigma0

    def _create_principle(self) -> trueskill.Rating:
        """Create a default principle."""

        return trueskill.Rating(mu=self.mu0, sigma=self.sigma0)

    def _update_principles(self, upvoted: str, downvoted: str, is_tie: bool) -> None:
        """Update the principles after a comparison.

        Args:
            upvoted (str): principle upvoted.
            downvoted (str): principle downvoted.
            is_tie (bool): whether the vote is a tie.
        """

        self.principles_params[upvoted], self.principles_params[downvoted] = trueskill.rate_1vs1(
            self.principles_params[upvoted],
            self.principles_params[downvoted],
            drawn=is_tie,
        )

    def _update_rating(self) -> None:
        """Update the rating."""

        self.rating = {
            principle: trueskill.expose(principle_params)
            for principle, principle_params in self.principles_params.items()
        }

    def get_result(self) -> dict:
        """Get the result of the algorithm.

        Returns:
            dict: result of the algorithm.
                mu (float): mu of the principle.
                sigma (float): sigma of the principle.
                rate (float): rate of the principle.
                rank (int): rank of the principle.
        """

        return {
            principle: {
                "mu": self.principles_params[principle].mu,
                "sigma": self.principles_params[principle].sigma,
                "rate": self.rating[principle],
                "rank": self.ranking[principle],
            }
            for principle in sorted(self.principles, key=lambda principle: self.ranking[principle])
        }



# Elo ranker
class EloRanker(BaseRanker):
    """Elo ranker."""
    ranker_name = "elo"

    def __init__(self, r0: float = 1_500, k: float = 32) -> None:
        """Initialize the ranker.

        Args:
            r0 (float, optional): initial rating. Defaults to 1_500.
            k (float, optional): k-factor. Defaults to 32.
        """
        super().__init__()

        self.r0 = r0
        self.k = k

    def _create_principle(self) -> float:
        """Create a default principle."""

        return self.r0

    def _update_principles(self, upvoted: str, downvoted: str, is_tie: bool) -> None:
        """Update the principles after a comparison.

        Args:
            upvoted (str): principle upvoted.
            downvoted (str): principle downvoted.
            is_tie (bool): whether the vote is a tie.
        """

        def compute_expectation(rate_a: float, rate_b: float) -> float:
            """Compute the expectation of a comparison.

            Args:
                rate_a (float): rating of the first principle.
                rate_b (float): rating of the second principle.

            Returns:
                float: expectation of the comparison.
            """
            return 1 / (1 + 10**((rate_b - rate_a) / 400))

        rate_up, rate_down = self.principles_params[upvoted], self.principles_params[downvoted]
        expectation_up, expectation_down = compute_expectation(rate_up,
                                                               rate_down), compute_expectation(rate_down, rate_up)

        if not is_tie:
            self.principles_params[upvoted] += self.k * (1 - expectation_up)
            self.principles_params[downvoted] += self.k * (0 - expectation_down)
        else:
            self.principles_params[upvoted] += self.k * (0.5 - expectation_up)
            self.principles_params[downvoted] += self.k * (0.5 - expectation_down)

    def _update_rating(self) -> None:
        """Update the rating."""

        self.rating = self.principles_params

    def get_result(self) -> dict:
        """Get the result of the algorithm.

        Returns:
            dict: result of the algorithm.
                rate (float): rate of the principle.
                rank (int): rank of the principle.
        """

        return {
            principle: {
                "rate": self.rating[principle],
                "rank": self.ranking[principle],
            }
            for principle in sorted(self.principles, key=lambda principle: self.ranking[principle])
        }


# WinRate ranker
class WinRateRanker(BaseRanker):
    """Win rate ranker."""
    ranker_name = "win_rate"

    def _create_principle(self) -> dict:
        """Create a default principle."""

        return {"upvotes": 0, "tievotes": 0, "downvotes": 0}

    def _update_principles(self, upvoted: str, downvoted: str, is_tie: bool) -> None:
        """Update the principles after a comparison.

        Args:
            upvoted (str): principle upvoted.
            downvoted (str): principle downvoted.
            is_tie (bool): whether the vote is a tie.
        """

        if not is_tie:
            self.principles_params[upvoted]["upvotes"] += 1
            self.principles_params[downvoted]["downvotes"] += 1
        else:
            self.principles_params[upvoted]["tievotes"] += 1
            self.principles_params[downvoted]["tievotes"] += 1

    def _update_rating(self) -> None:
        """Update the rating."""

        def compute_rate(upvotes: int | float, tievotes: int | float, downvotes: int | float) -> float:
            """Compute the rate of a principle.

            Args:
                upvotes (int|float): number of upvotes.
                tievotes (int|float): number of tievotes.
                downvotes (int|float): number of downvotes.

            Returns:
                float: rate of the principle.
            """

            #HACK to avoid division by zero
            upvotes += 1e-5
            downvotes += 1e-5

            numerator = upvotes + 0.5 * tievotes
            denominator = upvotes + tievotes + downvotes

            return numerator / denominator

        self.rating = {
            principle:
                compute_rate(
                    principle_params["upvotes"],
                    principle_params["tievotes"],
                    principle_params["downvotes"],
                )
            for principle, principle_params in self.principles_params.items()
        }

    def get_result(self) -> dict:
        """Get the result of the algorithm.

        Returns:
            dict: result of the algorithm.
                upvotes (int): number of upvotes of the principle.
                tievotes (int): number of tievotes of the principle.
                downvotes (int): number of downvotes of the principle.
                votes (int): total number of votes.
                rate (float): rate of the principle.
                rank (int): rank of the principle.
        """

        return {
            principle: {
                **self.principles_params[principle],
                "votes": (self.principles_params[principle]["upvotes"] + self.principles_params[principle]["tievotes"] +
                          self.principles_params[principle]["downvotes"]),
                "rate": self.rating[principle],
                "rank": self.ranking[principle],
            }
            for principle in sorted(self.principles, key=lambda principle: self.ranking[principle])
        }



# EigenvectorCentrality ranker
class EigenvectorCentralityRanker(BaseRanker):
    """Eigenvector centrality ranker."""
    ranker_name = "eigenvector_centrality"

    def __init__(self, max_iter: int = 1_000, patience: int = 25) -> None:
        """Initialize the ranker.

        Args:
            max_iter (int, optional): maximum number of iterations. Defaults to 1_000.
            patience (int, optional): number of iterations without improvement before stopping. Defaults to 25.
        """
        super().__init__()

        self.max_iter = max_iter
        self.patience = patience

    def _create_principle(self) -> dict:
        """Create a default principle."""

        return {}

    def _update_principles(self, upvoted: str, downvoted: str, is_tie: bool) -> None:
        """Update the principles after a comparison.

        Args:
            upvoted (str): principle upvoted.
            downvoted (str): principle downvoted.
            is_tie (bool): whether the vote is a tie.
        """

        if not is_tie:
            self.principles_params[upvoted][downvoted] = self.principles_params[upvoted].get(downvoted, 0) + 1
        else:
            self.principles_params[upvoted][downvoted] = self.principles_params[upvoted].get(downvoted, 0) + 0.5
            self.principles_params[downvoted][upvoted] = self.principles_params[downvoted].get(upvoted, 0) + 0.5

    def _update_rating(self) -> None:
        """Update the rating."""

        # create the loss count matrix (m[i, j] = number of losses of i against j)
        m = np.zeros((len(self.principles), len(self.principles)))
        for i, principle1 in enumerate(self.principles):
            for j, principle2 in enumerate(self.principles):
                m[j, i] = self.principles_params.get(principle1, {}).get(principle2, 0)

        # compute the loss ratio matrix (m[i, j] = ratio of losses of i against j)
        m = np.divide(m, m + m.T, out=np.zeros_like(m), where=(m + m.T) != 0)

        # scale the loss ratio matrix
        d_max = np.max(np.sum(m, axis=1))
        m = m / d_max
        # add self-loops
        np.fill_diagonal(m, 1 - np.sum(m, axis=1))

        # perform estimation through random-walk
        p0 = np.ones(len(self.principles)) / len(self.principles)
        patience = self.patience
        for _ in range(self.max_iter):
            pk = p0 @ m
            if (p0.argsort() == pk.argsort()).all():
                patience -= 1
                if patience == 0:
                    break
            else:
                patience = self.patience
            p0 = pk.copy()

        # update the rating
        self.rating = dict(zip(self.principles, p0))

    def get_result(self) -> dict:
        """Get the result of the algorithm.

        Returns:
            dict: result of the algorithm.
                rate (float): rate of the principle.
                rank (int): rank of the principle.
        """

        return {
            principle: {
                "rate": self.rating[principle],
                "rank": self.ranking[principle],
            }
            for principle in sorted(self.principles, key=lambda principle: self.ranking[principle])
        }


# BradleyTerry ranker
class BradleyTerryRanker(BaseRanker):
    """Bradley-Terry ranker."""
    ranker_name = "bradley_terry"

    def __init__(self, max_iter: int = 1_000, patience: int = 5) -> None:
        """Initialize the ranker.

        Args:
            max_iter (int, optional): maximum number of iterations. Defaults to 1_000.
            patience (int, optional): number of iterations without improvement before stopping. Defaults to 5.
        """
        super().__init__()

        self.max_iter = max_iter
        self.patience = patience

    def _create_principle(self) -> dict:
        """Create a default principle."""

        return {}

    def _update_principles(self, upvoted: str, downvoted: str, is_tie: bool) -> None:
        """Update the principles after a comparison.

        Args:
            upvoted (str): principle upvoted.
            downvoted (str): principle downvoted.
            is_tie (bool): whether the vote is a tie.
        """

        if not is_tie:
            self.principles_params[upvoted][downvoted] = self.principles_params[upvoted].get(downvoted, 0) + 1
        else:
            self.principles_params[upvoted][downvoted] = self.principles_params[upvoted].get(downvoted, 0) + 0.5
            self.principles_params[downvoted][upvoted] = self.principles_params[downvoted].get(upvoted, 0) + 0.5

    def _update_rating(self) -> None:
        """Update the rating."""

        # create the win ratio matrix (m[i, j] = ratio of wins of i against j)
        m = np.zeros((len(self.principles), len(self.principles)))
        for i, principle1 in enumerate(self.principles):
            for j, principle2 in enumerate(self.principles):
                m[i, j] = self.principles_params.get(principle1, {}).get(principle2, 0)
                #HACK to avoid division by zero
                if i != j:
                    m[i, j] += 1e-5

        # perform estimation through maximum-likelihood-estimation
        p0 = np.ones(len(self.principles))
        patience = self.patience
        for _ in range(self.max_iter):
            pk = p0.copy()
            for i in np.random.choice(len(self.principles), len(self.principles), replace=False):
                numerators = m[i, :] * pk / (pk[i] + pk)
                denominators = m[:, i] / (pk[i] + pk)
                pk[i] = np.sum(numerators) / np.sum(denominators)
            pk = pk / np.exp(np.mean(np.log(pk)))
            if (p0.argsort() == pk.argsort()).all():
                patience -= 1
                if patience == 0:
                    break
            else:
                patience = self.patience
            p0 = pk.copy()

        # update the rating
        self.rating = dict(zip(self.principles, p0))

    def get_result(self) -> dict:
        """Get the result of the algorithm.

        Returns:
            dict: result of the algorithm.
                rate (float): rate of the principle.
                rank (int): rank of the principle.
        """

        return {
            principle: {
                "rate": self.rating[principle],
                "rank": self.ranking[principle],
            }
            for principle in sorted(self.principles, key=lambda principle: self.ranking[principle])
        }