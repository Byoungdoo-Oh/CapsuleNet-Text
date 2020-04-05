# Capsule Networks for Text Classification

A Keras implementation of Capsule Networks in the paper:

[Kim, J., Jang, S., Park, E., & Choi, S. (2020). Text classification using capsules. Neurocomputing, 376, 214-221.](https://www.sciencedirect.com/science/article/pii/S0925231219314092)

- The layer of the Capsule Networks was implemented by reading the paper.
  - [Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between capsules. In Advances in neural information processing systems (pp. 3856-3866)](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf).
- In order to accurately understand and confirm the structure proposed in this paper, I confirmed and supplemented it through the [author's GitHub](https://github.com/TeamLab/text-capsule-network).
- In Testing
  - I tested this model with the [Naver Sentiment Movie Corpus (NSMC)](https://github.com/e9t/nsmc).
    - This is a corpus for representative emotional analysis in Korean.
  - I also tested with standard corpuses for text classification in English.
    - Movie Review (MR) dataset, etc.
- Version
  - Keras == 2.3.1
  - Tensorflow == 1.13.1
