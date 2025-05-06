from rankers import WinRateRanker
from typing import List, Dict, Tuple, Optional

def rank_principles(votes: List[Dict]) -> Tuple[List[int], Dict]:
    """
    Rank principles using the WinRateRanker algorithm.
    
    Args:
        votes: List of vote dictionaries containing:
            - lhs_id: ID of the left principle
            - rhs_id: ID of the right principle
            - choice: User's choice (1 for left, 2 for right, 3 for tie)
    
    Returns:
        Tuple containing:
        - List of principle IDs sorted by rank (highest to lowest)
        - Dictionary with detailed ranking information for each principle
    """
    # Initialize the ranker
    ranker = WinRateRanker()
    
    # Extract unique principle IDs from votes
    principle_ids = set()
    for vote in votes:
        principle_ids.add(str(vote["lhs_id"]))
        principle_ids.add(str(vote["rhs_id"]))
    
    # Add all principles to the ranker
    ranker.add_principles(list(principle_ids))
    
    # Process each vote
    for vote in votes:
        lhs_id = str(vote["lhs_id"])
        rhs_id = str(vote["rhs_id"])
        choice = vote["choice"]
        
        # Convert choice to upvoted/downvoted/tie format
        if choice == 1:  # Left principle chosen
            upvoted = lhs_id
            downvoted = rhs_id
            is_tie = False
        elif choice == 2:  # Right principle chosen
            upvoted = rhs_id
            downvoted = lhs_id
            is_tie = False
        else:  # Tie
            upvoted = lhs_id
            downvoted = rhs_id
            is_tie = True
            
        # Add the comparison
        ranker.add_comparison(upvoted, downvoted, is_tie)
    
    # Get the final ranking
    ranking = ranker.get_result()
    
    # Sort principles by rank (highest to lowest)
    sorted_principles = sorted(
        ranking.items(),
        key=lambda x: x[1]["rank"],
        reverse=True
    )
    
    # Extract just the IDs in ranked order
    ranked_ids = [int(principle_id) for principle_id, _ in sorted_principles]
    
    return ranked_ids, ranking

# Example usage:
if __name__ == "__main__":
    # Example votes
    votes = [
        {"lhs_id": 1, "rhs_id": 2, "choice": 1},  # Left principle chosen
        {"lhs_id": 2, "rhs_id": 3, "choice": 2},  # Right principle chosen
        {"lhs_id": 1, "rhs_id": 3, "choice": 3},  # Tie
    ]
    
    ranked_ids, detailed_ranking = rank_principles(votes)
    print("Ranked principle IDs:", ranked_ids)
    print("\nDetailed ranking information:")
    for principle_id, info in detailed_ranking.items():
        print(f"Principle {principle_id}:")
        print(f"  - Rank: {info['rank']}")
        print(f"  - Win Rate: {info['rate']:.3f}")
        print(f"  - Upvotes: {info['upvotes']}")
        print(f"  - Ties: {info['tievotes']}")
        print(f"  - Downvotes: {info['downvotes']}") 