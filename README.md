# Evaluating the Robustness of Collaborative Filtering Recommender Systems against Attacks

By: Andrew Zhou, Shruthi Kundapura, Kenny Zhang

This is the Github Repository with our source code of analyzing recommender systems with different attacks.  
The Proposal and Final paper can be found in the docs folder.

Our goal is to analyse the five shilling attack methods((Li et al. 2017)[8]): Random Attack, Average
Attack, Bandwagon Attack, Segmented Attack, Sampling Attack on different recommender systems.
Recommender models chosen for evaluation are: User-based KNN model, Item-based KNN model,
Latent Factor model, Neural Matrix Factorization model (NeuMF)(Dziugaite et al. 2015)[3]). The
intent of these evaluations is to discover which of these attack types are more effective on what kind
of recommender system, and also to identify properties of the target items which may influence the
impact of these attacks.

[Final Paper](https://github.com/azhou5211/CSE547_Project/blob/main/docs/Recommender%20Systems%20against%20Attacks.pdf)
