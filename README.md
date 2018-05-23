# BiLSTM을 이용한 영화 리뷰 감성분석
**Word2Vec**을 통해 임베딩 된 네이버 영화 리뷰 데이터를 **BiLSTM**을 통해 긍정, 부정을 분류해 주는 프로젝트

## 1. 모델 구조도
![alt text](https://github.com/MSWon/Sentimental-Analysis/blob/master/pic_1.PNG "Model")

1. 정답이 있는 네이버 영화 리뷰 데이터 15만건(박은정님 제공)에 대해서 **품사 태깅**

2. 품사 태깅한 단어들에 대해 **Word2Vec**을 이용해 학습시킨 임베딩 벡터로 변환

3. 단어 벡터들을 **BiLSTM**에 넣어서 모든 state들에 대해서 **fully connected layer**와 **Softmax**함수를 이용해 분류

