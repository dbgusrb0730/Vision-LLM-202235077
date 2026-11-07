"""Load the actual local model and test Korean intent parsing without a camera."""
import argparse
from settings import load_config
from gemma import GemmaDirector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.laptop.json')
    args = parser.parse_args()
    print('로컬 Gemma 모델을 불러옵니다. CPU에서는 시간이 걸릴 수 있습니다.', flush=True)
    director = GemmaDirector(load_config(args.config))
    passed = True
    for text, expected in [('전신 사진 찍을 거야', 'full'), ('상반신으로 바꿔 줘', 'upper'),
                           ('찍어 줘', 'shoot'), ('사진 찍지 마. 취소해', 'cancel')]:
        actual = director.intent(text)
        print(f'{text} -> {actual} (expected {expected})', flush=True)
        passed &= actual == expected
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
