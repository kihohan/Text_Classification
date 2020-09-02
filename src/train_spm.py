import os
import sentencepiece as spm
'''
data  sample
===================================================================================
mall_goods_name
신원 자외선 칫솔살균기 SW-15A 노랑
NS홈쇼핑 삼성전자 MC32K7056CT 세라믹조리실 오븐 32L 쇼핑도 건강하게
교세라 정품 TK-5154KY P6035cdn 10K 노랑
캐슬 Avon2 에이본2 리본트위터 북쉘프스피커
듀얼 팁 속눈썹 고데기
신일 심지식히터 SCS-850P/석유난로/야외,캠핑,사무실
명정보기술 MyStor S900 M.2 2280 (256GB)
로지텍 G304 무선 게이밍 마우스 병행 밀봉박스새상품
쿠쿠 보온밥솥 분리형 6인용 CR-0655FR
[바보사랑]hoco 호코 M55 하이파이 인이어형 이어폰 3.5MM 고음질 고감도
===================================================================================
'''
def make_spm_tokenizer(train_text, vocab_size, model_prefix):
    templates = '--input={} --model_prefix={} --vocab_size={}'
    cmd = templates.format(train_text, model_prefix, vocab_size)
    spm.SentencePieceTrainer.train(cmd)
    
if __name__ == '__main__':
    path = os.getcwd()
    train_text = path + '/data/cate.txt'
    vocab_size = 30000
    model_prefix = path + '/out/cate_spm'
    make_spm_tokenizer(train_text, vocab_size, model_prefix)