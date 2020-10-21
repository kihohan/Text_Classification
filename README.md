# Text_Classification

### Spm + W2v + CNN2d
---
![스크린샷 2020-09-02 오전 11 20 21](https://user-images.githubusercontent.com/40457277/91924676-52a39a00-ed0e-11ea-9bb4-86138cb5f096.png)   
![sample_model](https://user-images.githubusercontent.com/40457277/91927467-8ed9f900-ed14-11ea-9bde-84738a9b039a.png)   
![lime](https://user-images.githubusercontent.com/40457277/92676961-9ecd8a80-f35d-11ea-9c30-6a184794745f.PNG)   
---
- tutorial -> ipynb파일로 실행해 볼 수 있습니다.

- parameter은 간단하게 spm_vocab_size, embedding_layer, max_len 3개가 있으며(코드 내에서 다양하게 튜닝 가능) 아래와 같은 설정이 이상적으로 결과가 나오는 것으로 확인되었습니다. (참조) 

![스크린샷 2020-09-02 오전 11 38 01](https://user-images.githubusercontent.com/40457277/91925818-c777d380-ed10-11ea-85c0-2ff4eb648bc6.png)
- 기본적으로 data 폴드안에 spm모델을 생성 할 text파일과 classifcation할 pickle 데이터를 넣으시면 됩니다.
- spm text 파일은 classification 데이터 도메인에 맞는 '뛰어쓰기'가 잘된 파일을 학습시키는 것이 최선입니다.
- classification 파일은 아래와 같은 데이터 형식으로 디버그 됩니다.(두개의 컬럼이며, 본인 필요에 따라 코드를 수정하셔도 됩니다.)

![스크린샷 2020-09-02 오전 11 11 08](https://user-images.githubusercontent.com/40457277/91924399-8631f480-ed0d-11ea-80f6-e99231354dd3.png)
---
- src폴더 안의 .py파일은 data폴더 안에 학습시킬 spm text데이터와 classification할 pickle 데이터를 넣고 아래와 같은 순서로 디버그 하시면 out폴더에 모델이 생성됩니다.

> train_spm.py   
> train_w2v_and_model.py   
> prediction.py   

* reference
---

Sentencepiece: https://donghwa-kim.github.io/SPM.html  
Character-level Convolutional Networks for Text: Classification: https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf  
Convolutional Neural Networks for Sentence Classification: https://www.aclweb.org/anthology/D14-1181.pdf  
실습예제_1: https://reniew.github.io/26/  
실습예제_2: https://heung-bae-lee.github.io/2020/02/01/NLP_05/  
실습예제_3: https://buomsoo-kim.github.io/keras/2018/05/16/Easy-deep-learning-with-Keras-12.md/  
실습예제_4: https://buomsoo-kim.github.io/keras/2018/05/23/Easy-deep-learning-with-Keras-13.md/  
[NDC] 딥러닝으로 욕설 탐지하기: https://www.youtube.com/watch?v=K4nU7yXy7R8&t=1574s  
