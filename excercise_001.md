1. 첫 실습: 선형 회귀 모델 만들기
``` python
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
```

2. 데이터셋 활용 실습
붓꽃(iris) 데이터 분류: 가장 유명한 입문용 예제

손글씨 숫자(MNIST): 이미지 분류 실습

타이타닉 생존자 예측: Kaggle에서 인기 있는 실습용 데이터

3. 실습 자료
beyondborder 블로그의 실습 가이드: 개념부터 코드까지 단계별 설명 https://beyondborder.tistory.com/entry/🚀-머신러닝-완벽-가이드-개념부터-실습까지

초보자용 머신러닝 튜토리얼: 실습 위주로 구성된 입문자 친화형 자료 https://memory-glory.tistory.com/12

머신러닝 실습 완벽가이드 : 초보자를 위한 단계별 튜토리얼 https://memory-glory.tistory.com/12

파이썬 코드 예제 : 초보자를 위한 머신러닝 알고리즘 https://memory-glory.tistory.com/12

머신러닝 기초 정리 예제 : https://han-py.tistory.com/330
