import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, gaussian_kde

# 페이지 설정
st.set_page_config(page_title="Sunspot Data Analysis", layout="wide")
st.title("🌞 Sunspot Data Advanced Analysis & Visualization")

# 데이터 로드
df = pd.read_csv("./sunspots_for_prophet.csv")
df["YEAR"] = df["YEAR"].astype(int)
df["date"] = pd.to_datetime(df["YEAR"], format="%Y")
df = df.set_index("date")
df = df.sort_index()
sunactivity = df["y"].dropna()

# -------------------------------------------------------
# 1. 기본 통계 요약 및 분포 분석
# -------------------------------------------------------
st.header("1️⃣ 기본 통계 요약 및 분포 분석")

st.subheader("📊 기본 통계 요약")
st.dataframe(df.describe())

skew_val = skew(sunactivity)
kurt_val = kurtosis(sunactivity)
st.write(f"**데이터 왜도 (Skewness)**: {skew_val:.4f}")
st.write(f"**데이터 첨도 (Kurtosis)**: {kurt_val:.4f}")

st.subheader("📈 분포 시각화")
fig1, ax1 = plt.subplots()
ax1.hist(sunactivity, bins=30, density=True, alpha=0.6, label='Histogram', color='skyblue')
xs = np.linspace(sunactivity.min(), sunactivity.max(), 200)
density = gaussian_kde(sunactivity)
ax1.plot(xs, density(xs), color='red', label='Density')
ax1.set_title("Sunspot Activity Distribution")
ax1.set_xlabel("SUNACTIVITY")
ax1.set_ylabel("Density")
ax1.legend()
st.pyplot(fig1)

# -------------------------------------------------------
# 2. 결측치 및 이상치 확인
# -------------------------------------------------------
st.header("2️⃣ 결측치 및 이상치 확인")

st.subheader("🧩 결측치 확인")
st.write(df.isnull().sum())

st.subheader("🚨 이상치 탐지 (IQR 방법)")
Q1 = sunactivity.quantile(0.25)
Q3 = sunactivity.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

st.write(f"**하한 (Lower bound)**: {lower_bound:.2f}")
st.write(f"**상한 (Upper bound)**: {upper_bound:.2f}")

outliers = df[(df["y"] < lower_bound) | (df["y"] > upper_bound)]
st.markdown("**탐지된 이상치 (연도와 값):**")
st.dataframe(outliers[["YEAR", "y"]])

# -------------------------------------------------------
# 3. 심화 시각화: 다중 서브플롯
# -------------------------------------------------------
st.header("3️⃣ 심화 시각화 (서브플롯)")

fig2, axs = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle("Sunspots Data Advanced Visualization", fontsize=20)

# (a) 라인 차트
axs[0, 0].plot(df.index, df["y"], color='blue')
axs[0, 0].set_title("Sunspot Activity Over Time")
axs[0, 0].set_xlabel("Year")
axs[0, 0].set_ylabel("SUNACTIVITY")
axs[0, 0].grid(True)

# (b) 히스토그램 + 커널 밀도 추정
axs[0, 1].hist(sunactivity, bins=30, density=True, alpha=0.6, color='gray', label='Histogram')
axs[0, 1].plot(xs, density(xs), color='red', label='Density')
axs[0, 1].set_title("Distribution of Sunspot Activity")
axs[0, 1].set_xlabel("SUNACTIVITY")
axs[0, 1].set_ylabel("Density")
axs[0, 1].legend()
axs[0, 1].grid(True)

# (c) 1900-2000년 박스플롯
df_20th = df["1900":"2000"]
if not df_20th.empty:
    axs[1, 0].boxplot(df_20th["y"], vert=False)
    axs[1, 0].set_title("Boxplot (1900-2000)")
    axs[1, 0].set_xlabel("SUNACTIVITY")
else:
    axs[1, 0].text(0.5, 0.5, "No data between 1900 and 2000", ha='center', va='center')
    axs[1, 0].set_title("Boxplot (1900-2000)")

# (d) 산점도 + 회귀선
years = df["YEAR"]
sunvals = df["y"]
axs[1, 1].scatter(years, sunvals, s=10, alpha=0.5, label='Data Points')
trend_coef = np.polyfit(years, sunvals, 1)
trend = np.poly1d(trend_coef)
axs[1, 1].plot(years, trend(years), color='red', label='Trend Line')
axs[1, 1].set_title("Trend of Sunspot Activity")
axs[1, 1].set_xlabel("Year")
axs[1, 1].set_ylabel("SUNACTIVITY")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
st.pyplot(fig2)
