import numpy as np

np_arr = np.array([1, 2, 3, 4, 5])

# TODO: NumPy 연산 함수를 사용하여 배열을 정규화

np_arr_normalized = (np_arr - np_arr.mean()) / np_arr.std()
print(np_arr_normalized)

# TODO: NumPy 연산 함수를 사용하여 배열을 0.0 ~ 1.0 범위의 float32 배열로 변환
# TODO: 최종 배열의 최솟값 = 0.0, 최종 배열의 최댓값 = 1.0

np_arr_final = ((np_arr_normalized - np_arr_normalized.min()) / (np_arr_normalized.max() - np_arr_normalized.min())).astype(np.float32)
print(np_arr_final)
print(np_arr_final.dtype)


# Section Project 데이터
raw_students = [
    "zg0505_짱구",
    "cs0822_철수",
    "mg0910_맹구",
    "yr0605_유리",
    "hi0205_훈이"
    ]

raw_scores = np.array([
    [95, 82, 88],   # 짱구
    [70, 65, 90],   # 철수
    [98, 100, 92],  # 맹구
    [60, 75, 80],   # 유리
    [81, 88, 84],   # 훈이
])

subjects = ["국어", "수학", "과학"]
# TODO: list comprehension을 활용해 raw_students 리스트에서 이름만 추출한 리스트 생성
student_names = [student.split("_")[1] for student in raw_students]
print("1. 학생 명단: ", end="")
for idx, name in enumerate(student_names):
    sep = ", " if idx < len(student_names) - 1 else "\n\n"
    print(name, end=sep)

# TODO: 전산 시스템에 입력된 모든 성적들을 1차원 배열로 출력 (총 15개)
all_scores = raw_scores.flatten()
print("2. 전산 시스템에 입력된 모든 성적: ", end="")
for idx, score in enumerate(all_scores):
    sep = ", " if idx < len(all_scores) - 1 else "\n\n"
    print(f"{score}점", end=sep)

# TODO: 전체 15개 성적 중 85점 이상 개수
over_85 = (all_scores >= 85).sum()
print(f"3. 전체 15개 성적 중 85점 이상 개수: {over_85}개", end="\n\n")

# TODO: 학생별 평균 점수 출력
student_means = np.mean(raw_scores, axis=1)
print("4. 학생별 평균 점수: " + ", ".join(
    [f"{student_names[i]} {student_means[i]:.2f}점" for i in range(len(student_names))]
))

# TODO: 평균 점수가 가장 높은 학생 선택
top_student_idx = student_means.argmax()
top_student_name = student_names[top_student_idx]
print(f"5. 최고 평균 수석 학생: {top_student_name} ({student_means[top_student_idx]:.2f}점)")

# TODO: 전체 점수 기준, 최고 점수와 과목 그리고 해당 학생 계산 (modulo % 사용 추천)
ovr_max_score = all_scores.max()
ovr_max_idx = all_scores.argmax()
ovr_max_subject = subjects[ovr_max_idx % len(subjects)]
ovr_max_student = student_names[ovr_max_idx // len(subjects)]
print(f"6. 단일 과목 최고 점수: {ovr_max_score}점 (과목: {ovr_max_subject}, 학생: {ovr_max_student})")