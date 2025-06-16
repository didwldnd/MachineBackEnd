from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 모델과 컬럼만 로드 (scaler는 제거!)
try:
    model = load("model.pkl")
    scaler = load("scaler.pkl")  # ✅ 반드시 추가
    columns = load("columns.pkl")

    logger.info("✅ model.pkl, columns.pkl 로드 완료")
except Exception as e:
    logger.error(f"❌ 모델 로딩 실패: {e}")

# 입력 데이터 구조
class InputData(BaseModel):
    feature_values: list[float]

# DB 설정
Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    footfall = Column(Float)
    tempMode = Column(Float)
    AQ = Column(Float)
    USS = Column(Float)
    CS = Column(Float)
    VOC = Column(Float)
    RP = Column(Float)
    IP = Column(Float)
    Temperature = Column(Float)
    prediction = Column(Integer)
    probability = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

engine = create_engine("sqlite:///predictions.db", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)

# 🔍 예측 API
@app.post("/predict")
def predict(data: InputData):
    try:
        logger.info(f"🔥 입력 데이터: {data.feature_values}")
        X_df = pd.DataFrame([data.feature_values], columns=columns)

        # ✅ 스케일링
        X_scaled = scaler.transform(X_df)

        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]

        db = SessionLocal()
        record = PredictionRecord(
            footfall=X_df.iloc[0][0],
            tempMode=X_df.iloc[0][1],
            AQ=X_df.iloc[0][2],
            USS=X_df.iloc[0][3],
            CS=X_df.iloc[0][4],
            VOC=X_df.iloc[0][5],
            RP=X_df.iloc[0][6],
            IP=X_df.iloc[0][7],
            Temperature=X_df.iloc[0][8],
            prediction=int(pred),
            probability=round(float(proba), 4)
        )
        db.add(record)
        db.commit()
        db.close()

        return {
            "prediction": int(pred),
            "probability": round(float(proba), 4)
        }

    except Exception as e:
        logger.error(f"❌ 예측 실패: {e}")
        return {"error": str(e)}


# 이력 조회 API
@app.get("/history")
def get_history():
    db = SessionLocal()
    records = db.query(PredictionRecord).order_by(PredictionRecord.id.desc()).all()
    db.close()

    return [
        {
            "id": r.id,
            "footfall": r.footfall,
            "tempMode": r.tempMode,
            "AQ": r.AQ,
            "USS": r.USS,
            "CS": r.CS,
            "VOC": r.VOC,
            "RP": r.RP,
            "IP": r.IP,
            "Temperature": r.Temperature,
            "prediction": r.prediction,
            "probability": r.probability,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        for r in records
    ]

from fastapi.responses import JSONResponse

# 특성 중요도 API
@app.get("/feature-importance")
def get_feature_importance():
    try:
        # 모델이 feature_importances_ 속성을 가지고 있는지 확인
        if not hasattr(model, "feature_importances_"):
            raise AttributeError("모델에 feature_importances_ 속성이 없습니다")

        importances = model.feature_importances_

        importance_list = []
        for i, (name, score) in enumerate(
            sorted(zip(columns, importances), key=lambda x: x[1], reverse=True), start=1
        ):
            importance_list.append({
                "feature_name": name,
                "importance": round(float(score), 4),
                "rank": i
            })

        return JSONResponse(content=importance_list)

    except Exception as e:
        logger.error(f"❌ 특성 중요도 로딩 실패: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
