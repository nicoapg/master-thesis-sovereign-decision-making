from ._base import BaseRanker



# WinRate ranker
class WinRateRanker(BaseRanker):
    """Win rate ranker."""

    def __init__(self) -> None:
        """Initialize the ranker."""
        super().__init__(ranker_name="win_rate")


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

        def compute_rate(upvotes: int|float, tievotes: int|float, downvotes: int|float) -> float:
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

            numerator = upvotes + 0.5*tievotes
            denominator = upvotes + tievotes + downvotes

            return numerator / denominator

        self.rating = {
            principle: compute_rate(
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
                "votes": (
                    self.principles_params[principle]["upvotes"]
                    +
                    self.principles_params[principle]["tievotes"]
                    +
                    self.principles_params[principle]["downvotes"]
                ), 
                "rate": self.rating[principle], 
                "rank": self.ranking[principle], 
            }
            for principle in sorted(self.principles, key=lambda principle: self.ranking[principle])
        }
