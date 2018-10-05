# BiLSTM을 이용한 영화 리뷰 감성분석
**Word2Vec**을 통해 임베딩 된 네이버 영화 리뷰 데이터를 **BiLSTM**을 통해 긍정, 부정을 분류해 주는 프로젝트

## 1. 모델 구조도
![alt text](https://github.com/MSWon/Sentimental-Analysis/blob/master/pic/pic_1.PNG "Model")

1. 정답이 있는 네이버 영화 리뷰 데이터 15만건([박은정님 제공](https://github.com/e9t/nsmc))에 대해서 **품사 태깅**

2. 품사 태깅한 단어들에 대해 **Word2Vec**을 이용해 학습시킨 임베딩 벡터로 변환

3. 단어 벡터들을 **BiLSTM**에 넣어서 양쪽 끝 state들에 대해서 **fully connected layer**와 **Softmax**함수를 이용해 분류

## 2. 필요한 패키지

- [konlpy](http://konlpy.org/en/v0.4.4/)
- [tensorflow >= 1.5.0](https://www.tensorflow.org/)
- [gensim](https://radimrehurek.com/gensim/)

## 3. 데이터

- Training data : 영화 리뷰 데이터 15만건 [ratings_train.txt](https://github.com/e9t/nsmc)

- Test data : 영화 리뷰 데이터 5만건 [ratings_test.txt](https://github.com/e9t/nsmc)

## 4. 학습

![alt text](https://github.com/MSWon/Sentimental-Analysis/blob/master/pic/pic_2.png "Word2Vec Tensorboard")

1. **Word2Vec.py** , **Bi_LSTM.py**를 패키지 폴더로 이동 (C:\Users\jbk48\Anaconda3\Lib\site-packages)

2. **Word2Vec_train.py**로 품사 태깅한 단어들에 대해서 Word2Vec 학습 후 모델 저장 [Word2vec.model](https://drive.google.com/file/d/1Jxf_F_ibneTNRe_4glcWTYmj0TgLh8fP/view?usp=sharing)

3. **Word2Vec_Tensorboard.py**를 통해 시각화

4. **Bi_LSTM_train.py**를 통해 이진 분류기 학습

![alt text](https://github.com/MSWon/Sentimental-Analysis/blob/master/pic/pic_4.png "Accuracy graph")

   **epoch 4 이후에 overfitting이 되므로 epoch 4에서 early stopping을 한다.**

## 5. 결과

![alt text](https://github.com/MSWon/Sentimental-Analysis/blob/master/pic/pic_3.png "Result table")

- Bi_LSTM_test.py를 통해 test data에 대해서 성능 확인 (**86.52%**)

- Doc2Vec, Term-existance Naive Bayes에 의한 성능 보다 뛰어남([박은정](https://www.slideshare.net/lucypark/nltk-gensim))


![alt text](https://github.com/MSWon/Sentimental-Analysis/blob/master/pic/pic_4.png "Result")


- Grade_review.py를 통해 직접 입력한 문장에 성능 확인
