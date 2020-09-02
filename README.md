# text_classification

## Spm + W2v + cnn2d

tutorial -> ipynb파일로 실행해 볼 수 있습니다.

기본적으로 data 폴드안에 spm모델을 생성 할 text파일과 classifcation할 pickle 데이터를 넣으시면 됩니다.

spm text 파일은 classification 데이터 도메인에 맞는 '뛰어쓰기'가 잘된 파일을 학습시키는 것이 최선입니다.

classification 파일은 아래와 같은 데이터 형식으로 디버그 됩니다.(두개의 컬럼이며, 본인 필요에 따라 코드를 수정하셔도 됩니다.)


mall_goods_name	master_tag\n
신원 자외선 칫솔살균기 SW-15A 노랑	구강가전\n
NS홈쇼핑 삼성전자 MC32K7056CT 세라믹조리실 오븐 32L 쇼핑도 건강하게 ...	주방가전\n
교세라 정품 TK-5154KY P6035cdn 10K 노랑	사무가전(프린터/복합기)\n
캐슬 Avon2 에이본2 리본트위터 북쉘프스피커	스피커\n


src폴더 안의 .py파일은 
