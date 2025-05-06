import abc
import typing

import scipy



# Base ranker
class BaseRanker(abc.ABC):
    """Base class for rankers."""

    def __init__(self, ranker_name: str) -> None:
        """Initialize the ranker."""
        super().__init__()
    
        self.name: str = ranker_name

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

    def add_comparison(self, upvoted: str, downvoted: str, is_tie: bool=False, update_ranking: bool=True) -> None:
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

    def add_comparisons(self, comparisons: list[tuple[str, str, bool]], update_ranking: bool=True) -> None:
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
