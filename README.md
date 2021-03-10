# UsVsThem
This is the repository for our publication [_Us vs. Them: A Dataset of Populist Attitudes, News Bias and Emotions_](https://arxiv.org/abs/2101.11956), which will be presented at the upcoming EACL 2021 conference. 

Our Reddit dataset is made publicly available for the research community to be used. As a common courtesy your citation is very appreciated: 

Huguet Cabot, P. L., Abadi, D., Fischer, A., & Shutova, E. (2021). Us vs. Them: A Dataset of Populist Attitudes, News Bias and Emotions. arXiv e-prints, arXiv-2101. https://arxiv.org/abs/2101.11956

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

## Code

The code to train and test the dataset can be found in [src/](src/).

References to local data have been removed. 

Our dataset was trained and tested on an older version of _pytorch lightning_ with some manual fixes for DDP (Distributed Data Parallel) and not provided here. 

The branch [three_task](https://github.com/LittlePea13/UsVsThem/tree/three_task) includes the code to train and test the three-task MTL model using emotions and group identification.

## Data

The public version of our Reddit dataset is available on request. For this purpose, please contact [David Abadi](https://www.uva.nl/en/profile/a/b/d.r.abadi/d.r.abadi.html) (Department of Psychology, University of Amsterdam).

To comply with GDPR laws we provide our _Us Vs. Them_ dataset only with the Reddit comment body and the labels. 

For more information about the dataset or any original data, such as Reddit metadata or the news articles that prompted the comments, please contact us via email. 

## Funding statement
This research was funded by the H2020 project _Democratic Efficacy and the Varieties of Populism in Europe_ (DEMOS) under H2020-EU.3.6.1.1. and H2020-EU.3.6.1.2. (grant agreement ID: [822590](https://cordis.europa.eu/project/id/822590)) and supported by the European Union’s H2020 Marie Skłodowska-Curie project _Knowledge Graphs at Scale_ (KnowGraphs) under H2020-EU.1.3.1. (grant agreement ID: [860801](https://cordis.europa.eu/project/id/860801)).
