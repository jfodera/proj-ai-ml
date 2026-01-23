# Lect 1 In-Class Notes

## Administrative stuff
- grade will be assessed on how well we can explain our performance. Not necc how good/bad it is. 
- Data for projects is located in lect-1.ipynb
  - along with resources for final projects 
- For homeworks, submit a .txt file with a link to your work as your submission 


## What is Machine Learning? 
- ML algorithm is the process that uncovers underlying relationship within the data
- Machine learning model is the result of a Machine Learning Algorithm. 
  - think of this model as a function with varying outputs given varying inputs
- Model is derived from historical data. 
  - when trained on differing data, the model itself will change

## ML Applications
- recognize objects from pictures
- predict location of robot 
- direct self-driving car. 

## Categorizing ML Models 
### Types of Learning 
- Supervised: Regression, Classification 
- Unsupervised: model is not trained on the y's 
- Reinforcement Learning
### Types of models
- Discriminative - learns the decision boundary in order to classify 
  - cannot generate
- Generative - Learns the distribution of data in order to correctly infer, and generate new date 


## Grouping ML Algorithms'
### 3 Main Ways
- first by their learning style 
- second is by *modeling paradigm*
  - what they assume about the data and what they choose to model 
    - *Note* - this is not by task or optimization method 
- third by grouping of algorithm by similarity in form or function 

## ML Algorithms Grouped by Style 
### Supervised Learning Algorithms 
- input data is called training data and each point has a known label or result 
  - like spam/not-spam for emails 
- model is culminated by forcing it to make predictions on training data and correcting when wrong 
  - training continues until model reaches desired accuracy on training data 
- Example problems: 
  - Classification 
  - Regression 
- Example Algorithms: 
  - **Logistic Regression** - returns the probability that an input x belongs to class 1
    - where linear regression typically outputs a continuous real-valued number but is helpful for creating a decsision boundary 
  - **Back Propagation Neural Network** - utilizes gradient descent over multiple layers to minimize prediction error 

### Unsupervised Leaerning ALgorithms 
- input data is not labeled and does not have a known result 
- Model is curated by analyzing and pulling out structures present in the input data 
  - could organize data by similarity 
  - to process data and reduce redundancy
  - or to extract general rules.
- Example problems: 
  - clustering 
  - dimensionality reduction
  - *association rule learning* - discovers interesting relationships, patterns, or correlations between variables in large datasets 
    - if X occurs, then Y tends to occur. 
- Example Algorithms 
  - *DBScan* - Density-Based Spacial Clustering of Applications with Noise
  - *K-Means* - partitions data into K clusters

### Semi-Supervised Learning 
- Input data is a mixture of labeled and unlabeled 
- typically a desired prediction problem but the model must learn the structures to organize the data as well as make predictions 
- Example Problems: 
  - classification
  - regression 
- Example Algorithms 
  - extensions to other flexible methods that make assumptions about how to model the unlabeled data


## Reinforcement Learning 
- Training ML models to make a sequence of decisions
- Agent learns to achieve goal in uncertain and potentially complex environment. 
- The AI faces a 'game-like' situation 
- trial and error based when model is solving problems 
- In order to train, the AI gets either rewards or penalties for the actions it performs 
  - the goal of the model is to maximize the reward 

## ML Algs Grouped by Modeling Paradigm
- Back to the discriminative and Generative models 
### Discriminative Models
- make predictions on unseen data by considering conditional probability 
- used to discriminate between various classes 
- In the form of P(y|x) 
- Meant to either model the conditional distribution (P(y|x)) or directly learn a decision boundary 
- Goal: predict y from x 
- Ex. Log regression, SVM (support vector machine), or neural network classifiers 
### Generative Models 
- models focus on the distribution of the dataset to make a call on how llikely a given sample is. 
- capture probability in the form of P(y,x) or P(x) if no lables 
- Model joint distribution p(x,y) often via p(x | y) * p(y) or model p(x) alone 
- Goal: learn how data is generated, so it can eventually do it itself 
  - Can classify using Bayes': 
  $$p(y \mid x)=\frac{p(x \mid y)\,p(y)}{p(x)}.$$
- Examples: Naive Byes, Gaussian Mixture Models, HMMs, VAEs, GANs, diffusion 
#### Problem formation 
- assumed labeled data spam or ham problem
- we want to determine probability of spam email (P(y=1|x))
- Discriminative way: model will assume some underlying functional form for P(y|x)
  - parameters of the model are then determined by using training data. 
- Generative Model: conditional probability P(y|x) is calculated by using the prior P(y) and likelihood P(x|y) from training data
  - this is applied to Bayes theorem to get final result 

## Machine Learning Algorithms Grouped by Similarity.






## Misc Definitions
- A general learning algorithm is one that improves its performance on a task by learning from data
- A **Deep** learning algorithim is one that uses neural networks with many layers to learn features 
- **Regression** - how one variable changes as another changes 
- **Paradigm** - modeling approach or methodology 

## Reads if have time 
- https://cloud.google.com/learn/artificial-intelligence-vs-machine-learning
