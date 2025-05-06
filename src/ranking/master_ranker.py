from ._base import BaseRanker
from .win_rate import WinRateRanker
from .elo import EloRanker
from .true_skill import TrueSkillRanker
from .eigenvector_centrality import EigenvectorCentralityRanker
from .bradley_terry import BradleyTerryRanker



# Master ranker
class MasterRanker:
    """Master ranker."""

    def __init__(self, ranker_name: str, *args, **kwargs) -> None:

        self.name = ranker_name

        self.ranker: BaseRanker
        if ranker_name == "win_rate":
            self.ranker = WinRateRanker(*args, **kwargs)
        elif ranker_name == "elo":
            self.ranker = EloRanker(*args, **kwargs)
        elif ranker_name == "true_skill":
            self.ranker = TrueSkillRanker(*args, **kwargs)
        elif ranker_name == "eigenvector_centrality":
            self.ranker = EigenvectorCentralityRanker(*args, **kwargs)
        elif ranker_name == "bradley_terry":
            self.ranker = BradleyTerryRanker(*args, **kwargs)
        else:
            raise ValueError(f"Unknown ranking strategy {ranker_name}")


    def add_principle(self, principle: str) -> None:
        """Add a principle to the ranker.
        
        Args:
            principle (str): principle to add.
        """
        self.ranker.add_principle(principle)

    def add_principles(self, principles: list[str]) -> None:
        """Add principles to the ranker.

        Args:
            principles (list[str]): principles to add.
        """ 
        self.ranker.add_principles(principles)


    def add_comparison(self, upvoted: str, downvoted: str, is_tie: bool=False) -> None:
        """Add a comparison to the ranker.

        Args:
            upvoted (str): principle upvoted.
            downvoted (str): principle downvoted.
            is_tie (bool, optional): whether the comparison is a tie. Defaults to False.
        """
        self.ranker.add_comparison(upvoted, downvoted, is_tie)

    def add_comparisons(self, comparisons: list[tuple[str, str, bool]]) -> None:
        """Add comparisons to the ranker.

        Args:
            comparisons (list[tuple[str, str, bool]]): comparisons to add.
        """
        self.ranker.add_comparisons(comparisons)


    @property
    def rating(self) -> dict:
        """Get the rating."""
        return self.ranker.rating

    @property
    def ranking(self) -> dict:
        """Get the ranking."""
        return self.ranker.ranking

    def get_result(self) -> dict:
        """Get the result of the algorithm.

        Returns:
            dict: result of the algorithm.
        """
        return self.ranker.get_result()
