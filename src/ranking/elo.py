from ._base import BaseRanker



# Elo ranker
class EloRanker(BaseRanker):
    """Elo ranker."""

    def __init__(self, r0: float=1_500, k: float=32) -> None:
        """Initialize the ranker.
        
        Args:
            r0 (float, optional): initial rating. Defaults to 1_500.
            k (float, optional): k-factor. Defaults to 32.
        """
        super().__init__(ranker_name="elo")

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
            return 1 / (1 + 10 ** ((rate_b - rate_a) / 400))

        rate_up, rate_down = self.principles_params[upvoted], self.principles_params[downvoted]
        expectation_up, expectation_down = compute_expectation(rate_up, rate_down), compute_expectation(rate_down, rate_up)

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
