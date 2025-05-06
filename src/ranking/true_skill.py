import trueskill

from ._base import BaseRanker



# TrueSkill ranker
class TrueSkillRanker(BaseRanker):
    """Trueskill ranker."""

    def __init__(self, mu0: float=25, sigma0: float=25/3) -> None:
        """Initialize the ranker.
        
        Args:
            mu0 (float, optional): initial mu. Defaults to 25.
            sigma0 (float, optional): initial sigma. Defaults to 25/3.
        """
        super().__init__(ranker_name="true_skill")

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
            self.principles_params[upvoted], self.principles_params[downvoted], 
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
