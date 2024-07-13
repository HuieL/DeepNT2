## epinions
The user's trust and distrust information is included in this dataset; users evaluate other users based on the quality of their reviews on the item.

Column Details: 

    MY_ID This stores Id of the member who is making the trust/distrust statement 
    OTHER_ID The other ID is the ID of the member being trusted/distrusted 
    VALUE Value = 1 for trust and -1 for distrust 
    CREATION It is the date on which the trust was made 
    
Raw data comes from `https://www.kaggle.com/datasets/masoud3/epinions-trust-network`

## twitter
This dataset consists of 'circles' (or 'lists') from Twitter. Twitter data was crawled from public sources. The dataset includes node features (profiles), circles, and ego networks.

    nodeId.edges : The edges in the ego network for the node 'nodeId'. Edges are undirected for facebook, and directed (a follows b) for twitter and gplus. The 'ego' node does not appear, but it is assumed that they follow every node id that appears in this file.
    nodeId.circles : The set of circles for the ego node. Each line contains one circle, consisting of a series of node ids. The first entry in each line is the name of the circle.
    nodeId.feat : The features for each of the nodes that appears in the edge file.
    nodeId.egofeat : The features for the ego user.
    nodeId.featnames : The names of each of the feature dimensions. Features are '1' if the user has this property in their profile, and '0' otherwise. This file has been anonymized for facebook users, since the names of the features would reveal private data.
    This dataset consists of 'circles' (or 'lists') from Twitter. Twitter data was crawled from public sources. The dataset includes node features (profiles), circles, and ego networks.

Raw data comes from `https://snap.stanford.edu/data/ego-Twitter.html`
