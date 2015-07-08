# About
This is a solution for kaggle [CrowdFlower Search Results Relevance](https://www.kaggle.com/c/crowdflower-search-relevance) competition, score about 0.658 on public leaderboard

# Model
## Feature Engineering
- Similarity rates between query and product title and product description
- Query binary indicator 
- Topics modeling using TruncatedSVD on TfIdf matrix of products

## Predictive algorithm
SVM with rbf kernel
