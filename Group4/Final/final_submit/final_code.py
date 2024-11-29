import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 읽기
df = pd.read_csv("C:/Users/wogns/PycharmProjects/pythonProject/data/seattle-weather.csv")

# 기본적인 데이터 확인
print(df.head(10))
print(df.info())
print(df.isna().sum().to_frame())

# 'weather' 열의 값에 대한 파이 차트 생성
df['weather'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, figsize=(7,7))

# 그래프를 화면에 띄우기 위해 plt.show() 호출
plt.title("Weather Distribution")
plt.ylabel("")  # y축 라벨 제거
plt.show()

df['is_sunny'] = df['weather'].map(lambda x: 1 if x == 'sun' else 0)
df['is_sunny'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, figsize=(7,7))

plt.title("Sunny Weather Distribution")
plt.ylabel("")  # y축 라벨 제거
plt.show()

df['date'] = pd.to_datetime(df['date'])
df['is_sunny'] = df['weather'].map(lambda x: 1 if x == 'sun' else 0)
df_time = df[['date', 'is_sunny', 'weather']].copy()
df_time.iloc[-100:]['is_sunny'].plot(figsize=(10, 6))
plt.title("Is it Sunny in the Last 100 Days?")
plt.xlabel("Date")
plt.ylabel("Is Sunny (1 = Sunny, 0 = Not Sunny)")
plt.grid(True)
plt.show()


df_time['day'] = df_time['date'].dt.day
df_time['month'] = df_time['date'].dt.month
df_time['year'] = df_time['date'].dt.year
df_time['day_of_week'] = df_time['date'].dt.dayofweek
plt.figure(figsize=(10, 6))
sns.countplot(data=df_time, x='year', hue='is_sunny')
plt.title("Sunny Days Distribution by Year")
plt.xlabel("Year")
plt.ylabel("Count of Days")
plt.legend(title="Is Sunny", labels=["Not Sunny", "Sunny"])
plt.show()

plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 그래프
sns.countplot(data=df_time, x='month', hue='is_sunny')
plt.title("Sunny Days Distribution by Month")
plt.xlabel("Month")
plt.ylabel("Count of Days")
plt.legend(title="Is Sunny", labels=["Not Sunny", "Sunny"])
plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 그래프
sns.countplot(data=df_time, x='day_of_week', hue='is_sunny')
plt.title("Sunny Days Distribution by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Count of Days")
plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.legend(title="Is Sunny", labels=["Not Sunny", "Sunny"])
plt.tight_layout()  # 레이아웃 조정
plt.show()

pd.plotting.autocorrelation_plot(df['is_sunny'])
plt.show()

df = df.sort_values(by='date')

# 'lag_30', 'lag_180', 'lag_365' 열 추가: 30일, 180일, 365일 전의 값
df['lag_30'] = df['is_sunny'].shift(30)
df['lag_180'] = df['is_sunny'].shift(180)
df['lag_365'] = df['is_sunny'].shift(365)

# 상관 행렬을 계산하고 heatmap을 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(df[['lag_30', 'lag_180', 'lag_365', 'is_sunny']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Lagged and Actual Sunny Days")
plt.show()

df['rolling_5'] = df['is_sunny'].rolling(35).sum()

# NaN 값을 처리 (첫 34개의 값은 NaN이므로 제거)
df = df.dropna(subset=['rolling_5'])

# countplot 그리기
plt.figure(figsize=(10, 6))
sns.countplot(x='rolling_5', hue='is_sunny', data=df)
plt.title('Countplot of Rolling Sum of Sunny Days (35 days window)')
plt.show()

y = df['is_sunny'][35:]  # 목표 변수 (햇살이 비친 날인지 여부)
X = df.drop(['date', 'is_sunny', 'weather', 'lag_180', 'lag_365', 'precipitation', 'temp_max', 'temp_min', 'wind'], axis=1)[35:]  # 특성들

# 날짜 정보를 특성으로 추가
X['dayofweek'] = df['date'].dt.dayofweek[35:]  # 요일
X['day'] = df['date'].dt.day[35:]  # 일
X['month'] = df['date'].dt.month[35:]  # 월
X['year'] = df['date'].dt.year[35:]  # 년

# X 확인
print(X)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# 첫 번째 예측 모델

# 시계열 데이터 교차 검증을 위한 TimeSeriesSplit 객체 생성 (5개 분할)
tscv = TimeSeriesSplit(5)

# 정확도 점수를 저장할 리스트
acc_score_rf = []  # 랜덤 포레스트의 정확도 점수
precision_score_rf = []  # 랜덤 포레스트의 정밀도
recall_score_rf = []  # 랜덤 포레스트의 재현율
f1_score_rf = []  # 랜덤 포레스트의 F1 점수

# 교차 검증 반복
for i, (train_ind, test_ind) in enumerate(tscv.split(X)):
    X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]  # 훈련 데이터와 목표 변수
    X_test, y_test = X.iloc[test_ind], y.iloc[test_ind]  # 테스트 데이터와 목표 변수

    # 2. 랜덤 포레스트 모델
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train, y_train)

    # 예측
    y_pred_rf = model_rf.predict(X_test)

    # 정확도, 정밀도, 재현율, F1 점수 계산
    acc_score_rf.append(accuracy_score(y_test, y_pred_rf))
    precision_score_rf.append(precision_score(y_test, y_pred_rf))
    recall_score_rf.append(recall_score(y_test, y_pred_rf))
    f1_score_rf.append(f1_score(y_test, y_pred_rf))

    # 각 반복에 대해 분류 보고서 출력 (랜덤포레스트)
    print(f'{i}th iter - Random Forest:')
    print(classification_report(y_test, y_pred_rf))

# 랜덤 포레스트 모델에 대한 정확도, 정밀도, F1 점수 평균 출력
print(f'Random Forest - Accuracy: {np.mean(acc_score_rf):.2f}')
print(f'Random Forest - Precision: {np.mean(precision_score_rf):.2f}')
print(f'Random Forest - Recall: {np.mean(recall_score_rf):.2f}')
print(f'Random Forest - F1 Score: {np.mean(f1_score_rf):.2f}')



# 두 번째 예측모델

# 이동 평균을 적용할 컬럼 목록
cols_to_apply_rolling = ['precipitation', 'temp_max', 'temp_min', 'wind']

# 각 컬럼에 대해 이동 평균을 계산 (window=3)
for col in cols_to_apply_rolling:
    df[f'{col}_rolling_3'] = df[col].rolling(window=3).mean()

# NaN 값 제거 (처음 2개 값은 NaN이므로 삭제)
df = df.dropna(subset=[f'{col}_rolling_3' for col in cols_to_apply_rolling])

# 훈련 데이터 X와 목표 변수 y 정의
y = df['is_sunny'][3:]  # 목표 변수 (햇살이 비친 날인지 여부)
X = df.drop(['date', 'is_sunny', 'weather', 'lag_180', 'lag_365', 'precipitation', 'temp_max', 'temp_min', 'wind'], axis=1)[3:]  # 특성들

# 이동 평균이 적용된 컬럼만 남기기
X = X[['precipitation_rolling_3', 'temp_max_rolling_3', 'temp_min_rolling_3', 'wind_rolling_3']]

# 시계열 데이터 교차 검증을 위한 TimeSeriesSplit 객체 생성 (5개 분할)
tscv = TimeSeriesSplit(5)

# 정확도 점수를 저장할 리스트
acc_score_rf = []  # 랜덤 포레스트의 정확도 점수
precision_score_rf = []  # 랜덤 포레스트의 정밀도
recall_score_rf = []  # 랜덤 포레스트의 재현율
f1_score_rf = []  # 랜덤 포레스트의 F1 점수

# 교차 검증 반복
for i, (train_ind, test_ind) in enumerate(tscv.split(X)):
    X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]  # 훈련 데이터와 목표 변수
    X_test, y_test = X.iloc[test_ind], y.iloc[test_ind]  # 테스트 데이터와 목표 변수

    # 2. 랜덤 포레스트 모델
    model_rf = RandomForestClassifier(class_weight='balanced')
    model_rf.fit(X_train, y_train)

    # 예측
    y_pred_rf = model_rf.predict(X_test)

    # 정확도, 정밀도, 재현율, F1 점수 계산
    acc_score_rf.append(accuracy_score(y_test, y_pred_rf))
    precision_score_rf.append(precision_score(y_test, y_pred_rf))
    recall_score_rf.append(recall_score(y_test, y_pred_rf))
    f1_score_rf.append(f1_score(y_test, y_pred_rf))

    # 각 반복에 대해 분류 보고서 출력 (랜덤포레스트)
    print(f'{i}th iter - Random Forest:')
    print(classification_report(y_test, y_pred_rf))

# 랜덤 포레스트 모델에 대한 정확도, 정밀도, 재현율, F1 점수 평균 출력
print(f'Random Forest - Accuracy: {np.mean(acc_score_rf):.2f}')
print(f'Random Forest - Precision: {np.mean(precision_score_rf):.2f}')
print(f'Random Forest - Recall: {np.mean(recall_score_rf):.2f}')
print(f'Random Forest - F1 Score: {np.mean(f1_score_rf):.2f}')



import xgboost as xgb

# 세 번째 예측모델

# 이동 평균을 적용할 컬럼 목록
cols_to_apply_rolling = ['precipitation', 'temp_max', 'temp_min', 'wind']

# 각 컬럼에 대해 이동 평균을 계산 (window=3)
for col in cols_to_apply_rolling:
    df[f'{col}_rolling_3'] = df[col].rolling(window=3).mean()

# NaN 값 제거 (처음 2개 값은 NaN이므로 삭제)
df = df.dropna(subset=[f'{col}_rolling_3' for col in cols_to_apply_rolling])

# 훈련 데이터 X와 목표 변수 y 정의
y = df['is_sunny'][35:]  # 목표 변수 (햇살이 비친 날인지 여부)
X = df.drop(['date', 'is_sunny', 'weather', 'lag_180', 'lag_365', 'precipitation', 'temp_max', 'temp_min', 'wind'], axis=1)[35:]  # 특성들

# 이동 평균이 적용된 컬럼만 남기기
X = X[['precipitation_rolling_3', 'temp_max_rolling_3', 'temp_min_rolling_3', 'wind_rolling_3']]

# 시계열 데이터 교차 검증을 위한 TimeSeriesSplit 객체 생성 (5개 분할)
tscv = TimeSeriesSplit(5)

# 정확도 점수를 저장할 리스트
acc_score_xgb = []  # XGBoost의 정확도 점수
precision_score_xgb = []  # XGBoost의 정밀도
recall_score_xgb = []  # XGBoost의 재현율
f1_score_xgb = []  # XGBoost의 F1 점수

# 교차 검증 반복
for i, (train_ind, test_ind) in enumerate(tscv.split(X)):
    X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]  # 훈련 데이터와 목표 변수
    X_test, y_test = X.iloc[test_ind], y.iloc[test_ind]  # 테스트 데이터와 목표 변수

    # 2. XGBoost 모델
    model_xgb = xgb.XGBClassifier(
        scale_pos_weight=10,  # 클래스 불균형을 처리하기 위해 scale_pos_weight 설정
        eval_metric='logloss',  # 평가 기준을 logloss로 설정
        random_state=42
    )
    model_xgb.fit(X_train, y_train)

    # 예측
    y_pred_xgb = model_xgb.predict(X_test)

    # 정확도, 정밀도, F1 점수 계산
    acc_score_xgb.append(accuracy_score(y_test, y_pred_xgb))
    precision_score_xgb.append(precision_score(y_test, y_pred_xgb))
    recall_score_xgb.append(recall_score(y_test, y_pred_xgb))
    f1_score_xgb.append(f1_score(y_test, y_pred_xgb))

    # 각 반복에 대해 분류 보고서 출력 (XGBoost)
    print(f'{i}th iter - XGBoost:')
    print(classification_report(y_test, y_pred_xgb))

# XGBoost 모델에 대한 정확도, 정밀도, F1 점수 평균 출력
print(f'XGBoost - Accuracy: {np.mean(acc_score_xgb):.2f}')
print(f'XGBoost - Precision: {np.mean(precision_score_xgb):.2f}')
print(f'XGBoost - Recall: {np.mean(recall_score_xgb):.2f}')
print(f'XGBoost - F1 Score: {np.mean(f1_score_xgb):.2f}')



from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# 네 번째 예측 모델

# 시계열 데이터 교차 검증을 위한 TimeSeriesSplit 객체 생성 (5개 분할)
tscv = TimeSeriesSplit(5)

# 하이퍼파라미터 그리드 정의
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# 랜덤 포레스트 모델
model_rf = RandomForestClassifier(random_state=42)

# GridSearchCV 설정
grid_search = GridSearchCV(
    estimator=model_rf,
    param_grid=param_grid,
    cv=tscv,  # TimeSeriesSplit 사용
    scoring='f1',  # F1 점수를 기준으로 최적화
    verbose=2,
    n_jobs=-1  # 병렬 실행
)

# GridSearchCV 실행
grid_search.fit(X, y)

# 최적 하이퍼파라미터 출력
print("Best parameters found: ", grid_search.best_params_)

# 최적 모델 성능 평가
best_model = grid_search.best_estimator_

# 교차 검증 결과 저장용 리스트
acc_score_rf = []  # 랜덤 포레스트의 정확도 점수
precision_score_rf = []  # 랜덤 포레스트의 정밀도
recall_score_rf = []  # 랜덤 포레스트의 재현율
f1_score_rf = []  # 랜덤 포레스트의 F1 점수

for i, (train_ind, test_ind) in enumerate(tscv.split(X)):
    X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
    X_test, y_test = X.iloc[test_ind], y.iloc[test_ind]

    # 최적화된 모델로 학습 및 예측
    best_model.fit(X_train, y_train)
    y_pred_rf = best_model.predict(X_test)

    # 성능 계산
    acc_score_rf.append(accuracy_score(y_test, y_pred_rf))
    precision_score_rf.append(precision_score(y_test, y_pred_rf))
    recall_score_rf.append(recall_score(y_test, y_pred_rf))
    f1_score_rf.append(f1_score(y_test, y_pred_rf))

    print(f'{i}th iter - Optimized Random Forest:')
    print(classification_report(y_test, y_pred_rf))

# 최적화된 랜덤 포레스트 모델에 대한 평균 성능 출력
print(f'Optimized Random Forest - Accuracy: {np.mean(acc_score_rf):.2f}')
print(f'Optimized Random Forest - Precision: {np.mean(precision_score_rf):.2f}')
print(f'Optimized Random Forest - Recall: {np.mean(recall_score_rf):.2f}')
print(f'Optimized Random Forest - F1 Score: {np.mean(f1_score_rf):.2f}')