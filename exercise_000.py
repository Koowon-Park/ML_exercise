#첫 실습: 선형 회귀 모델 만들기

from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 준비
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10, 20, 30, 40, 50])

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 예측
print(model.predict([[6]]))  # 결과: 약 60
