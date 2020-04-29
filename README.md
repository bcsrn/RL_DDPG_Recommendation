# RL_DDPG_Recommendation
Project for Course : Reinforcement Learning

How to Run code:

Using Google Colab Platform:

Place data in Data Folder in drive folder RL Project.
The path to your data should look like - My Drive/RL Project/Data

The Notebook is in the RL Project Folder. Run the cells in sequence

Using an IDE:

Download the data from the link referred below
Change the paths in the python folder to the path to the data

On windows the path could look like: r'C:\Users\HOME\Documents\RLProject\Data\ml-1m\'

Install the dependencies - pytorch,torch,sklearn,numpy,pandas,scipy,matplotlib,tqdm they are mentioned at the top of the python file 

Run python -W ignore rl_actorcritic_ddpg_movie_recommendation.py

Implementation of Paper: Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling. 
Ref: Liu, Feng & Tang, Ruiming & Li, Xutao & Ye, Yunming & Chen, Haokun & Guo, Huifeng. (2018). Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling. 

Dataset Link: https://grouplens.org/datasets/movielens/1m/

Dataset Reference: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

Other References: 

1. @misc{RecNN,
  author = {M Scherbina},
  title = {RecNN: RL Recommendation with PyTorch},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/awarebayes/RecNN}},
}

2. https://github.com/shashist/recsys-rl/tree/35b12eb775baa4cb2dc573df3964c7f84cbcc98b

