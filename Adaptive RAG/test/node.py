from state import QAState
import re
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import get_model_name, LLMs


# 최신 LLM 모델 이름 가져오기
MODEL_NAME = get_model_name(LLMs.GPT4)

#유틸 함수
def clean_product_name(query):
    query = re.sub(r"(에 대해.*|알려줘|무엇입니까|뭔가요|뭐야|어떻게.*|사용법|부작용|효능|복용.*|섭취.*|투여.*)", "", query)
    query = re.sub(r"[^\w가-힣]", "", query)
    query = re.sub(r"(은|는|이|가|을|를)$", "", query)
    return query.strip()

def normalize(text):
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^\w가-힣]", "", text)
    return re.sub(r"\s+", "", text.strip().lower())


#라우팅 함수
# 사용자 쿼리를 가장 관련성 높은 데이터 소스로 라우팅하는 데이터 모델
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    # 데이터 소스 선택을 위한 리터럴 타입 필드
    datasource: Literal["MEDICINE_RECOMMENDATION", "MEDICINE_INFO"] = Field(
        ...,
        description="Given a user question choose to route it to medication recommendation or medication search in document.",
    )


# LLM 초기화 및 함수 호출을 통한 구조화된 출력 생성
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# 시스템 메시지와 사용자 질문을 포함한 프롬프트 템플릿 생성
system = """
사용자의 질문을 분석해서 다음 중 어떤 datasource에 해당하는지 판단해주세요.

질문: {question}

datasource:
1. MEDICINE_RECOMMENDATION: 질병 증상을 기반으로 약품 추천을 원하는 경우
   - 예: "두통이 있는데 어떤 약을 먹어야 하나요?", "감기 증상에 좋은 약 추천해주세요"
   
2. MEDICINE_INFO: 특정 약품에 대한 정보를 알고 싶어하는 경우
   - 예: "타이레놀 부작용이 뭐예요?", "게보린의 효능을 알려주세요"

다음 JSON 형식으로 답변해주세요:
{{
    "datasource": "MEDICINE_RECOMMENDATION" 또는 "MEDICINE_INFO",
    "reason": "판단 근거",
    "condition": ["증상1", "증상2"] (MEDICINE_RECOMMENDATION인 경우),
    "category": "약품명", "약품 종류" (MEDICINE_INFO인 경우),
    "requested_fields": ["사용자가 알고싶어하는 요청 항목"] (효능, 부작용 등등)
}}
"""

# Routing 을 위한 프롬프트 템플릿 생성
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# 프롬프트 템플릿과 구조화된 LLM 라우터를 결합하여 질문 라우터 생성
question_router = route_prompt | structured_llm_router


    

#사용자 질문 전처리 및 분석 노드
def preprocess_query(state):
    query = state["query"]
    cleaned = clean_product_name(query)
    normalized = normalized(cleaned)
    return {"cleaned_query": cleaned, "normalized_query": normalized}



#질문 라우팅 노드
def route_question(state):
    # 질문 가져오기
    query = state["normalized_query"]
    # 질문 라우팅
    source = question_router.invoke({"question": query})
    # 질문 라우팅 결과에 따른 노드 라우팅
    if source.datasource == "MEDICINE_RECOMMENDATION":
        return "medi_recommend"
    elif source.datasource == "MEDICINE_INFO":
        return "doc_search"  #pdf, excel 등 document 검색 노드로
    

#사용자 병력에 기반한 약품 추천 노드