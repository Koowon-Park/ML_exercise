# 필요한 라이브러리들을 가져옵니다.
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터 로드
# seaborn 라이브러리에 내장된 타이타닉 데이터셋을 불러옵니다.
titanic_df = sns.load_dataset('titanic')

# 2. 데이터 전처리
# 예측에 사용할 특성(feature)과 목표(target) 변수를 선택합니다.
# 'survived'는 생존 여부(0=사망, 1=생존)이며, 우리가 예측하려는 목표입니다.
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

# 선택한 특성들과 목표 변수만 남깁니다.
df = titanic_df[features + [target]]

# 결측치(비어있는 값) 처리
# 'age' 열의 결측치는 전체 나이의 중간값으로 채웁니다.
df['age'].fillna(df['age'].median(), inplace=True)
# 'embarked' 열의 결측치는 가장 빈도가 높은 값으로 채웁니다.
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# 범주형(문자열) 데이터 숫자 변환
# 'sex'와 'embarked' 열을 모델이 이해할 수 있는 숫자 형태로 변환합니다.
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# 입력(X)과 정답(y) 데이터로 분리합니다.
X = df.drop(target, axis=1)
y = df[target]

# 3. 데이터 분할
# 훈련용 데이터(80%)와 테스트용 데이터(20%)로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 훈련
# 랜덤 포레스트 분류 모델을 생성하고 훈련시킵니다.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 예측 및 평가
# 테스트 데이터로 생존 여부를 예측합니다.
predictions = model.predict(X_test)

# 모델의 정확도를 계산하고 출력합니다.
accuracy = accuracy_score(y_test, predictions)

print(f"타이타닉 생존자 예측 모델의 정확도: {accuracy:.2f}") 