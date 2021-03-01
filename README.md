# UsVsThem
Repository for the paper [Us vs. Them: A Dataset of Populist Attitudes, News Bias and Emotions](https://arxiv.org/abs/2101.11956) that will be presented in EACL 2021.

Code to train and test found in [src/](src/). References to local data have been removed. Trained and tested on an older version of pytorch lightning with some manual fixes for DDP and not provided here. Main includes the code to train and test the two-task MTL model using emotions or group identification and the STL model. Branch [three_task](https://github.com/LittlePea13/UsVsThem/tree/three_task) includes the code to train and test the three-task MTL model using emotions and group identification.

The public version of the dataset is available at [data/](data/). To comply with  GDPR we provide the Us Vs. Them dataset with just the Reddit comment body and the labels. For more information about the dataset or the extra data included in the original one such as Reddit metadata or the news articles that prompted the comments please contact one of the authors.

Code and dataset are made publicly available for the research community to be used. As a common courtesy your citation is appreciated: Huguet Cabot, P. L., Abadi, D., Fischer, A., & Shutova, E. (2021). Us vs. Them: A Dataset of Populist Attitudes, News Bias and Emotions. arXiv e-prints, arXiv-2101. 

    @misc{huguetcabot2021vs,
          title={Us vs. Them: A Dataset of Populist Attitudes, News Bias and Emotions}, 
          author={Huguet Cabot, Pere-Llu{\'\i}s  and
          Abadi, David  and
          Fischer, Agneta  and
          Shutova, Ekaterina},
          year={2021},
          eprint={2101.11956},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
    }


This research was funded by the H2020 project Democratic Efficacy and the Varieties of Populism in Europe (DEMOS) under H2020-EU.3.6.1.1. and H2020-EU.3.6.1.2. (grant agreement ID: [822590](https://cordis.europa.eu/project/id/822590)).
