# UsVsThem
This is the repository for our publication [_Us vs. Them: A Dataset of Populist Attitudes, News Bias and Emotions_](http://dx.doi.org/10.18653/v1/2021.eacl-main.165), which was recently published and presented at the EACL 2021 conference. Our Reddit dataset is made publicly available for the research community to be used. As a common courtesy your citation is very appreciated:

Huguet Cabot, P. L. H., Abadi, D., Fischer, A., & Shutova, E. (2021, April). Us vs. Them: A Dataset of Populist Attitudes, News Bias and Emotions. In _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_ (pp. 1921–1945). http://dx.doi.org/10.18653/v1/2021.eacl-main.165

    @inproceedings{huguet-cabot-etal-2021-us,
    title = "Us vs. Them: A Dataset of Populist Attitudes, News Bias and Emotions",
    author = "Huguet Cabot, Pere-Llu{\'\i}s  and
      Abadi, David  and
      Fischer, Agneta  and
      Shutova, Ekaterina",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "http://dx.doi.org/10.18653/v1/2021.eacl-main.165",
    pages = "1921--1945"
    }

## Code

The code to train and test the dataset can be found in [src/](src/).

References to local data have been removed. 

Our dataset was trained and tested on an older version of _pytorch lightning_ with some manual fixes for DDP (Distributed Data Parallel) and not provided here. 

The branch [three_task](https://github.com/LittlePea13/UsVsThem/tree/three_task) includes the code to train and test the three-task MTL model using emotions and group identification.

## Data

The public version of our Reddit dataset is available on request only. For this purpose, please contact [Pere-Lluis Huguet Cabot](https://littlepea13.github.io) (Sapienza University of Rome).

To comply with GDPR laws we provide our _Us Vs. Them_ dataset exclusively with the Reddit comment body and the labels. 

For further information about our dataset or any original data, such as Reddit metadata or the news articles that prompted the comments, please contact us via email. 

## Funding statement

This research was funded by the H2020 project _Democratic Efficacy and the Varieties of Populism in Europe_ (DEMOS) under H2020-EU.3.6.1.1. and H2020-EU.3.6.1.2. (grant agreement ID: [822590](https://cordis.europa.eu/project/id/822590)) and supported by the European Union’s H2020 Marie Skłodowska-Curie project _Knowledge Graphs at Scale_ (KnowGraphs) under H2020-EU.1.3.1. (grant agreement ID: [860801](https://cordis.europa.eu/project/id/860801)).
