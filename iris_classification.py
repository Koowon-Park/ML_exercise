# 필요한 라이브러리들을 가져옵니다.
import pandas as pd  # 데이터 조작 및 분석을 위한 라이브러리
from sklearn.model_selection import train_test_split  # 데이터를 훈련용과 테스트용으로 나누는 함수
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 모델
from sklearn.metrics import accuracy_score  # 모델의 정확도를 평가하는 함수

# 붓꽃(Iris) 데이터셋을 CSV 파일에서 읽어옵니다.
df = pd.read_csv('datasets/uciml/iris.csv')

# 입력(특성) 데이터와 목표(정답) 변수를 정의합니다.
# X는 꽃의 특성(꽃받침 길이/너비, 꽃잎 길이/너비)을 나타냅니다.
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
# y는 예측하고자 하는 목표인 품종을 나타냅니다.
y = df['Species']

# 전체 데이터를 훈련용 데이터(80%)와 테스트용 데이터(20%)로 나눕니다.
# random_state를 42로 설정하여 항상 동일한 방식으로 데이터가 나뉘도록 합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델을 생성합니다.
model = LogisticRegression()

# 훈련용 데이터를 사용하여 모델을 학습시킵니다.
model.fit(X_train, y_train)

# 학습된 모델을 사용하여 테스트 데이터의 품종을 예측합니다.
predictions = model.predict(X_test)

# 모델의 예측 정확도를 계산합니다.
# 실제 정답(y_test)과 모델의 예측(predictions)을 비교합니다.
accuracy = accuracy_score(y_test, predictions)

# 최종 결과를 출력합니다.
print("모델 예측 결과:")
print(predictions)
print(f"모델 정확도: {accuracy:.2f}") 
