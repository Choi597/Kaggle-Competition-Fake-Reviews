Fake Reviews:
The training and validation datasets are similar (independent samples from the same P(Y,X) distribution). The test data labels "real review?" are all wrong (-1). Task: Correctly classify a review as real or fake given its characteristics. Each row contains a review. A review is characterized by:

A review label (1 for real reviews and 0 for fake reviews).
Category of the item.
Star rating.
Review text (variable-size).


Fake Reviews Task: Given the reviews, ratings, and category of different items, use predictive modeling to determine whether the review is real or fabricated.
Fake Reviews Dataset: We are given 3 Datasets (Reviews_Train, Reviews_Validation, Reviews_Test_Attributes). Reviews_Test_Attributes has 2249 Observations/rows and 5 columns, Reviews_Train has 37,200 Observations/Rows and 4 columns, and Reviews_Validation has 999 Observations/Rows and 4 columns
Knowledge Representation: Here, I used Logistic Regression, a Binary Classification algorithm used to classify categorical variables into exactly two classes. This would be appropriate here; Since we are categorizing reviews as either real or fake, we would need to assign a label (real/fake) to each entity in the test set given a training set with labeled entities. As a result, knowledge Representation would be Two Classes, labeled (y=0 and y=1).
