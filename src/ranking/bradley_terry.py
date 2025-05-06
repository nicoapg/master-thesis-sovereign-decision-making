import numpy as np

from ._base import BaseRanker



# BradleyTerry ranker
class BradleyTerryRanker(BaseRanker):
    """Bradley-Terry ranker."""

    def __init__(self, max_iter: int=1_000, patience: int=5) -> None:
        """Initialize the ranker.

        Args:
            max_iter (int, optional): maximum number of iterations. Defaults to 1_000.
            patience (int, optional): number of iterations without improvement before stopping. Defaults to 5.
        """
        super().__init__(ranker_name="bradley_terry")

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
