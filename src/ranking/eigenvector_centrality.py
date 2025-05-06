import numpy as np

from ._base import BaseRanker



# EigenvectorCentrality ranker
class EigenvectorCentralityRanker(BaseRanker):
    """Eigenvector centrality ranker."""

    def __init__(self, max_iter: int=1_000, tol: float=1e-5) -> None:
        """Initialize the ranker.
        
        Args:
            max_iter (int, optional): maximum number of iterations. Defaults to 1_000.
            tol (float, optional): tolerance for convergence. Defaults to 1e-3.
        """
        super().__init__(ranker_name="eigenvector_centrality")

        self.max_iter = max_iter
        self.tol = tol

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
        m = np.divide(m, m+m.T, out=np.zeros_like(m), where=(m+m.T)!=0)

        # scale the loss ratio matrix
        d_max = np.max(np.sum(m, axis=1))
        m = m / d_max
        # add self-loops
        np.fill_diagonal(m, 1-np.sum(m, axis=1))
        
        # perform estimation through random-walk
        p0 = np.ones(len(self.principles)) / len(self.principles)
        for _ in range(self.max_iter):
            pk = p0 @ m
            if np.max(np.abs(p0-pk)) < self.tol:
                break
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
