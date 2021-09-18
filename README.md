# Recommender Systems
## Recommender Systems Theory
<li>Given a dataset of triples (user, item, rating)</li>
<li>We can fit a model to the data f(u, m) --> r</li>
<li>The model will do 2 things:</li>
<li>#1 if the user u and movie m appeared in the dataset, then the predicted r should be close to the true r</li>
<li>predict what a user u would rate a movie m even if it didn't appear in datset</li>

## How to recommend:
<li>The model can predict ratings for unseen movies</li>
<li>for a given user, get a prediction for unseen movies</li>
<li>sort predicted ratings </li>
<li>recommed movies based on highest raitings.</li>


## Pseudocode:
<code> 
#stage 1<br>
u = Input(shape=(1,)) # u for user<br>
m = Input(shape=(1,)) # m for movie<br>

#stage 2: convert u and m to feature vectors using embeddings<br>
u_emb = Embedding(num_users, embedding_dim)(u)<br>
m_emb = Embedding(num_users, embedding_dim)(m)<br>

#stage 3: concatenate for a single feature vector<br>
x = Concat()(u_emb, m_emb)<br>

#stage 4: model using ANN<br>
x = Dense(512, activation='relu')(x)<br>
x = Dense(1)(x)<br>


</code>
