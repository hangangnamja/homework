import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ----------------------------
# [1] 한글 폰트 설정
# ----------------------------
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# [2] 데이터 불러오기
# ----------------------------
df = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")

# ----------------------------
# [3] 결측치 처리
# ----------------------------
df = df.dropna(subset=["Year_of_Release", "Genre", "Global_Sales"])

# User_Score: 숫자로 변환, 'tbd' 등은 NaN 처리
df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")

# Critic_Score도 일부 NaN일 수 있음 → 평균으로 채움
df["User_Score"] = df["User_Score"].fillna(df["User_Score"].mean())
df["Critic_Score"] = df["Critic_Score"].fillna(df["Critic_Score"].mean())

# Year를 정수형으로
df["Year_of_Release"] = df["Year_of_Release"].astype(int)

# ----------------------------
# [4] 이상치 제거
# ----------------------------
df = df[df["Global_Sales"] < 50]

# ----------------------------
# [5] 장르별 판매량 합계 계산
# ----------------------------
genre_sales = df.groupby(["Year_of_Release", "Genre"])["Global_Sales"].sum().reset_index()

# ----------------------------
# [6] 머신러닝 모델: 장르별 판매량 예측
# ----------------------------
# 피처(X)와 타겟(y) 설정
X = genre_sales[["Year_of_Release", "Genre"]]  # 장르별 판매량 예측을 위한 특성
X = pd.get_dummies(X, drop_first=True)  # 장르는 범주형 변수이므로 더미 변수로 변환
y = genre_sales["Global_Sales"]  # 타겟은 판매량

# 학습용 데이터와 테스트용 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestRegressor 모델 훈련
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 테스트 데이터를 사용해 예측
y_pred = model.predict(X_test)

# 성능 평가 (MSE: 평균 제곱 오차)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# ----------------------------
# [7] Streamlit UI: 장르별 판매량 시각화
# ----------------------------

# 예측된 판매량과 실제 판매량을 합쳐서 시각화
genre_sales_pred = X_test.copy()
genre_sales_pred['Actual_Sales'] = y_test
genre_sales_pred['Predicted_Sales'] = y_pred

# 'Genre' 컬럼을 X_test에서 다시 가져오기
genre_sales_pred['Genre'] = genre_sales['Genre'].iloc[X_test.index].values

# Streamlit UI 설정
st.title("게임 장르별 판매량 예측")
st.write("각 장르별 실제 판매량과 예측 판매량을 확인하세요.")

# 장르별로 버튼 만들기
genres = df["Genre"].unique()

for genre in genres:
    if st.button(f"{genre} 장르"):
        genre_data = genre_sales_pred[genre_sales_pred["Genre"] == genre]

        # 실제 판매량과 예측 판매량 시각화
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.lineplot(data=genre_data, x="Year_of_Release", y="Actual_Sales", label="Actual Sales", ax=ax, marker='o')
        sns.lineplot(data=genre_data, x="Year_of_Release", y="Predicted_Sales", label="Predicted Sales", ax=ax,
                     linestyle="--", marker='x')

        ax.set_title(f"{genre} 장르: 실제 판매량 vs 예측 판매량")
        ax.set_xlabel("출시 연도")
        ax.set_ylabel("글로벌 판매량 (백만 단위)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig)
