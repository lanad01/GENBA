from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS 설정 (Streamlit에서 API 호출 가능하도록 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 현재 페이지 상태를 저장할 변수
current_page = {"page": "home"}

class PageRequest(BaseModel):
    page: str

@app.get("/get_page")
def get_page():
    """현재 페이지 상태 반환"""
    return current_page

@app.post("/set_page")
def set_page(data: PageRequest):
    """페이지 상태 변경"""
    current_page["page"] = data.page
    return {"message": "Page updated", "current_page": current_page}
