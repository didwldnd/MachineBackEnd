import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

file_path = "data.csv"

# 1. 데이터 로딩
df = pd.read_csv(file_path)
X = df.drop(columns=["fail"])
y = df["fail"]

# 2. 이상치 처리 (중앙값 치환)
for col in X.columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    median = X[col].median()
    X[col] = X[col].apply(lambda x: median if x < lower or x > upper else x)

# 3. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. 저장
dump(model, "model.pkl")
dump(scaler, "scaler.pkl")
dump(list(X.columns), "columns.pkl")