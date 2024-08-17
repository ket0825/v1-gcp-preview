# 대분류 Asepct Category 매핑 Dictionary
# value (list) 내에 포함된 label을 key 값으로 변경
label_map_dict = {
    "디자인": ["커스터마이징", "그립감", "색감", "로고없음", "재질","디자인"], # 1 / 6
    "안전": ["인증", "발열", "과충전방지", "과전류","안전"], # 1 / 5
    "서비스": ["AS", "환불", "문의", "교환", "수리", "보험", "배송","서비스", "배송/포장/발송"], # 1 / 9
    "기능": ["멀티포트", "거치", "부착", "디스플레이", "잔량표시", "충전표시","기능"], # 1 / 7
    "충전": ["고속충전", "동시충전","저전력", "무선충전", "맥세이프", "배터리충전속도","충전"], # 1 / 7
    "휴대성": ["사이즈", "무게","휴대성"], # 1 / 3
    "기타":["기내반입", "수명", "친환경", "구성품", "케이블", "파우치", "케이스","기타"], # 1 / 8
    "배터리를충전하는호환성":["호환성","배터리를충전하는호환성"], # 1 / 2
    "배터리용량":["배터리용량"], # 1 / 1
}


# 대분류 Asepct Category Dictionary에 BIO tag 적용
label_list,label_changing_rule = [], {}
for key in label_map_dict.keys():
    if key != 'O':
        label_list.extend(['B-' + key, 'I-' + key])
    else:
        label_list.append('O')
for key, labels in label_map_dict.items():
    for label in labels:
        if key != label:
            for tag in ["B-", "I-"]:
                label_changing_rule[tag + label] = tag + key