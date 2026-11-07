"""Fast, conservative routing of common Korean commands from either STT or text.

Voice uses explicit commands. Mode selection arms one automatic photo.
"""
import re


def direct_intent(utterance):
    text = re.sub(r'[\s,.!?。\-–—]', '', utterance)
    if text in {'하나', '하나번', '일번', '1', '1번'}:
        return 'full'
    if text in {'둘', '둘번', '두번째', '이번', '2', '2번'}:
        return 'upper'
    if text in {'준비', '촬영준비', '준비시작'}:
        return 'prepare'
    if text in {'취소', '취소해', '취소해줘', '촬영취소', '촬영취소해', '촬영취소해줘',
                '찍지마', '사진찍지마', '찍지말아줘'}:
        return 'cancel'
    if text in {'종료', '종료해', '종료해줘', '프로그램종료', '프로그램종료해줘'}:
        return 'quit'
    if text in {'사진', '촬영', '찍어', '찍어줘', '사진찍어', '사진찍어줘', '촬영해', '촬영해줘'}:
        return 'shoot'
    if text in {'구도', '구도어때', '지금구도어때', '구도확인', '구도확인해줘'}:
        return 'status'
    # Recognize explicit phrases, not arbitrary text merely mentioning a mode.
    match = re.fullmatch(r'(?:나|나는|저|저는)?(전신|상반신)(?:사진)?'
                         r'(?:모드|모드로|으로|으로바꿔|으로바꿔줘|모드로바꿔줘|'
                         r'찍을거야|찍을게|찍고싶어|찍고싶어요|찍어줘|촬영할거야)?', text)
    if match:
        return 'full' if match.group(1) == '전신' else 'upper'
    return None
