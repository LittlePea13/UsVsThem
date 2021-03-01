# UsVsThem
Repository for the paper Us vs. Them: A Dataset of Populist Attitudes, News Bias and Emotions that will be presented in EACL 2021.

Code to train and test found in [src/](src/). References to local data have been removed. Trained and tested on an older version of pytorch lightning with some manual fixes for DDP and not provided here. Branch three_task includes the code to train and test the three-task MTL model using emotions and group identification.

The public version of the dataset is available at [data/](data/). To comply with  GDPR we provide the Us Vs. Them dataset with just the Reddit comment body and the labels. For more information about the dataset or the extra data included in the original one such as Reddit metadata or the news articles that prompted the comments please contact one of the authors.

This research was funded by the H2020 project Democratic Efficacy and the Varieties of Populism in Europe (DEMOS) under H2020-EU.3.6.1.1. and H2020-EU.3.6.1.2. (grant agreement ID: 822590).
