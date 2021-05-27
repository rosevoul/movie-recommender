# Movie Recommender
Recommend movies to a user based on previously watched and rated movies.
Both non-personalized and personalized recommender systems are tested and compared.

> Non personalized approaches 

- [x] Popularity-based: recommends the top 10 most popular movies to each user
- [x] Average ranking: recommends the top 10 both top ranked and most popular movies to each user
- [x] Pair-wise: recommends the top 10 most popular movies to each user by taking into account the proximity (permutations) to the last movie the user watched

> Personalized approaches 

- [ ] Content-based recommendations: recommend  movies similar to previously watched ones
    - [x] Genre-based recommendations, Jaccard similarity for big data
- [ ] Collaborative Filter (matrix factorization approach using SVD) 
- [ ] Hybrid recommender 



# Dataset
The Movielens 20m dataset (ml-20m) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 20000263 ratings and 465564 tag applications across 27278 movies. These data were created by 138493 users between January 09, 1995 and March 31, 2015. This dataset was generated on March 31, 2015, and updated on October 17, 2016 to update links.csv and add genome-* files.

Users were selected at random for inclusion. All selected users had rated at least 20 movies. No demographic information is included. Each user is represented by an id, and no other information is provided.

The data are contained in six files, `genome-scores.csv`, `genome-tags.csv`, `links.csv`, `movies.csv`, `ratings.csv` and `tags.csv`. 

Download from:
https://grouplens.org/datasets/movielens/20m/ 


# How to run
<!-- TODO -->

# Next steps

For state-of-the-art recommender engines on Movielens 20m dataset
https://paperswithcode.com/sota/collaborative-filtering-on-movielens-100k