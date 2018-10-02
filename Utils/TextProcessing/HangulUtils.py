# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/kss
'''

from __future__ import print_function
import os
from itertools import chain
from Utils.Hyperparams import hparams
from Utils.TextProcessing.KoG2P import runKoG2P

hangul_symbol_1 = [u"␀", u"␃", u'"', " ", "!", ",", ".", "?", 'aa', 'c0', 'cc', 'ch', 'ee', 'h0', 'ii', 'k0', 'kf', 'kh',
                   'kk',
                   'ks', 'lb', 'lh', 'lk', 'll', 'lm', 'lp',
                   'ls', 'lt', 'mf', 'mm', 'nc', 'nf', 'nh', 'nn', 'ng', 'oh', 'oo', 'p0', 'pf', 'ph', 'pp', 'ps', 'qq',
                   'rr',
                   's0',
                   'ss', 't0', 'tf', 'th', 'tt', 'uu', 'vv', 'wa', 'we', 'wi', 'wo', 'wq', 'wv', 'xi', 'xx', 'ya', 'ye',
                   'yo',
                   'yq', 'yu', 'yv']
hangul_symbol_2 = u'''␀␃ !,.?ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ'''
hangul_symbol_3 = u'''␀␃ !,.?ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄾㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'''  # HCJ
hangul_symbol_4 = u'''␀␃ !,.?ᄀᄂᄃᄅᄆᄇᄉᄋᄌᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆫᆮᆯᆷᆸᆺᆼᆽᆾᆿᇀᇁᇂ'''
hangul_symbol_5 = u'''␀␃ !,.?ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'''  # HCJ. single consonants only.
hangul_type = hparams.hangul_type

if hangul_type == 1:
    hangul_to_ids = {char: idx for idx, char in enumerate(hangul_symbol_1)}
    ids_to_hangul = {idx: char for idx, char in enumerate(hangul_symbol_1)}
elif hangul_type == 2:
    hangul_to_ids = {char: idx for idx, char in enumerate(hangul_symbol_2)}
    ids_to_hangul = {idx: char for idx, char in enumerate(hangul_symbol_2)}
elif hangul_type == 3:
    hangul_to_ids = {char: idx for idx, char in enumerate(hangul_symbol_3)}
    ids_to_hangul = {idx: char for idx, char in enumerate(hangul_symbol_3)}
elif hangul_type == 4:
    hangul_to_ids = {char: idx for idx, char in enumerate(hangul_symbol_4)}
    ids_to_hangul = {idx: char for idx, char in enumerate(hangul_symbol_4)}
elif hangul_type == 5:
    hangul_to_ids = {char: idx for idx, char in enumerate(hangul_symbol_5)}
    ids_to_hangul = {idx: char for idx, char in enumerate(hangul_symbol_5)}
else:
    hangul_to_ids = None
    ids_to_hangul = None
    raise Exception('Hangul type must in range (1:5)')


def hangul_to_sequence(dir, hangul_text, hangul_type=1):
    # load conversion dictionaries
    j2hcj, j2sj, j2shcj = load_j2hcj(), load_j2sj(), load_j2shcj()
    rulebook = os.path.join(dir, 'Utils/TextProcessing/rulebook.txt')
    hangul_text = number_to_hangul(hangul_text)
    hangul_text = hangul_text + u"␃"  # ␃: EOS
    if hangul_type == 1:
        hangul_text = runKoG2P(hangul_text, rulebook)
    else:
        if hangul_type == 3:
            hangul_text = [j2hcj[char] for char in hangul_text]
        elif hangul_type == 4:
            hangul_text = [j2sj[char] for char in hangul_text]
        elif hangul_type == 5:
            hangul_text = [j2shcj[char] for char in hangul_text]
    sequence = [hangul_to_ids[char] for char in hangul_text]
    return sequence

##m = hangul_to_sequence(dir='', hangul_text='사진좀봐라',hangul_type=1)


### match different representation of hangul
def load_j2hcj():
    '''
    Arg:
      jamo: A Hangul Jamo character(0x01100-0x011FF)

    Returns:
      A dictionary that converts jamo into Hangul Compatibility Jamo(0x03130 - 0x0318F) Character
    '''
    jamo = u'''␀␃ !,.?ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ'''
    hcj = u'''␀␃ !,.?ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄾㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'''

    assert len(jamo) == len(hcj)
    j2hcj = {j: h for j, h in zip(jamo, hcj)}
    return j2hcj


def load_j2sj():
    '''
    Arg:
      jamo: A Hangul Jamo character(0x01100-0x011FF)

    Returns:
      A dictionary that decomposes double consonants into two single consonants.
    '''
    jamo = u'''␀␃ !,.?ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ'''
    sj = u'''␀|␃| |!|,|.|?|ᄀ|ᄀᄀ|ᄂ|ᄃ|ᄃᄃ|ᄅ|ᄆ|ᄇ|ᄇᄇ|ᄉ|ᄉᄉ|ᄋ|ᄌ|ᄌᄌ|ᄎ|ᄏ|ᄐ|ᄑ|ᄒ|ᅡ|ᅢ|ᅣ|ᅤ|ᅥ|ᅦ|ᅧ|ᅨ|ᅩ|ᅪ|ᅫ|ᅬ|ᅭ|ᅮ|ᅯ|ᅰ|ᅱ|ᅲ|ᅳ|ᅴ|ᅵ|ᆨ|ᆨᆨ|ᆨᆺ|ᆫ|ᆫᆽ|ᆫᇂ|ᆮ|ᆯ|ᆯᆨ|ᆯᆷ|ᆯᆸ|ᆯᇀ|ᆯᇂ|ᆷ|ᆸ|ᆸᆺ|ᆺ|ᆺᆺ|ᆼ|ᆽ|ᆾ|ᆿ|ᇀ|ᇁ|ᇂ'''

    assert len(jamo) == len(sj.split("|"))
    j2sj = {j: s for j, s in zip(jamo, sj.split("|"))}
    return j2sj


def load_j2shcj():
    '''
    Arg:
      jamo: A Hangul Jamo character(0x01100-0x011FF)

    Returns:
      A dictionary that converts jamo into Hangul Compatibility Jamo(0x03130 - 0x0318F) Character.
      Double consonants are further decomposed into single consonants.
    '''
    jamo = u'''␀␃ !,.?ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ'''
    shcj = u'''␀|␃| |!|,|.|?|ㄱ|ㄱㄱ|ㄴ|ㄷ|ㄷㄷ|ㄹ|ㅁ|ㅂ|ㅂㅂ|ㅅ|ㅅㅅ|ㅇ|ㅈ|ㅈㅈ|ㅊ|ㅋ|ㅌ|ㅍ|ㅎ|ㅏ|ㅐ|ㅑ|ㅒ|ㅓ|ㅔ|ㅕ|ㅖ|ㅗ|ㅘ|ㅙ|ㅚ|ㅛ|ㅜ|ㅝ|ㅞ|ㅟ|ㅠ|ㅡ|ㅢ|ㅣ|ㄱ|ㄱㄱ|ㄱㅅ|ㄴ|ㄴㅈ|ㄴㅎ|ㄷ|ㄹ|ㄹㄱ|ㄹㅁ|ㄹㅂ|ㄹㅌ|ㄹㅎ|ㅁ|ㅂ|ㅂㅅ|ㅅ|ㅅㅅ|ㅇ|ㅈ|ㅊ|ㅋ|ㅌ|ㅍ|ㅎ'''

    assert len(jamo) == len(shcj.split("|"))
    j2shcj = {j: s for j, s in zip(jamo, shcj.split("|"))}
    return j2shcj


# 숫자를 입력받아서 한글로 출력하는 함수
# 1~999
def number_to_hangul(text):
    import re
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    for number in numbers:
        number_text = digit2txt(number)
        text = text.replace(number, number_text, 1)
    return text


def digit2txt(strNum):
    # 만 단위 자릿수
    tenThousandPos = 4
    # 억 단위 자릿수
    hundredMillionPos = 9
    txtDigit = ['', '십', '백', '천', '만', '억']
    txtNumber = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
    txtPoint = '쩜 '
    resultStr = ''
    digitCount = 0
    #자릿수 카운트
    for ch in strNum:
        # ',' 무시
        if ch == ',':
            continue
        #소숫점 까지
        elif ch == '.':
            break
        digitCount = digitCount + 1
    digitCount = digitCount-1
    index = 0
    while True:
        notShowDigit = False
        ch = strNum[index]
        #print(str(index) + ' ' + ch + ' ' +str(digitCount))
        # ',' 무시
        if ch == ',':
            index = index + 1
            if index >= len(strNum):
                break
            continue
        if ch == '.':
            resultStr = resultStr + txtPoint
        else:
            #자릿수가 2자리이고 1이면 '일'은 표시 안함.
            # 단 '만' '억'에서는 표시 함
            if(digitCount > 1) and (digitCount != tenThousandPos) and  (digitCount != hundredMillionPos) and int(ch) == 1:
                resultStr = resultStr + ''
            elif int(ch) == 0:
                resultStr = resultStr + ''
                # 단 '만' '억'에서는 표시 함
                if (digitCount != tenThousandPos) and  (digitCount != hundredMillionPos):
                    notShowDigit = True
            else:
                resultStr = resultStr + txtNumber[int(ch)]
        # 1억 이상
        if digitCount > hundredMillionPos:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount-hundredMillionPos]
        # 1만 이상
        elif digitCount > tenThousandPos:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount-tenThousandPos]
        else:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount]
        if digitCount <= 0:
            digitCount = 0
        else:
            digitCount = digitCount - 1
        index = index + 1
        if index >= len(strNum):
            break
    return resultStr
