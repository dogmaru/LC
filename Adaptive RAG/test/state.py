###기존 state.py (State import error)

# from typing import Optional, List
# from langgraph.graph import State
# from langchain_core.documents import Document

# class QAState(State):
#     """
#     LangGraph 기반 의약품 QA 시스템의 전체 상태를 관리하는 클래스입니다.

#     각 노드에서 참조하거나 수정할 수 있는 주요 필드들은 다음과 같습니다:

#     - 사용자 입력 관련
#         - query: 사용자 원 질문
#         - cleaned_query: 약품명 정제 결과
#         - normalized_query: 약품명 정규화 결과

#     - 전처리 정보
#         - condition: 병력 정보 (ex. 위장염)
#         - category: 약물 카테고리 (ex. 감기약)
#         - requested_fields: 사용자가 궁금해하는 항목들 (효능, 부작용 등)

#     - 추천 흐름
#         - recommendation_answer: 병력 기반 약품 추천 결과

#     - 관련성 판단
#         - is_medicine_related: 약 관련 질문인지 여부 판단 결과

#     - 과거 질문 활용
#         - previous_context: 이전 질문 맥락

#     - 검색 결과
#         - pdf_results: PDF 검색 결과 문서 리스트
#         - excel_results: Excel 검색 결과 문서 리스트
#         - external_raw: 외부 검색의 원문 텍스트
#         - external_parsed: 외부 검색을 LLM이 정제한 JSON 구조

#     - 후처리 및 평가
#         - reranked_docs: 리랭킹 결과 문서 리스트
#         - relevant_docs: 문서 평가 후 관련성 높은 문서 리스트
#         - hallucination_flag: 환각 여부 판단 결과
#         - re_query: 재검색 시 새로 생성된 질문

#     - 최종 생성 결과
#         - final_answer: LLM이 생성한 최종 응답 텍스트
#      """ 
#     # 사용자 입력 관련
#     query: str  # 원 질문
#     cleaned_query: Optional[str] = None  # 약품명 정제
#     normalized_query: Optional[str] = None  # 약품명 정규화

#     # 전처리 정보
#     condition: Optional[str] = None  # 병력
#     category: Optional[str] = None  # 약물 카테고리
#     requested_fields: Optional[List[str]] = None  # 사용자가 요청한 항목들

#     # 추천 흐름
#     recommendation_answer: Optional[str] = None  # 병력 기반 추천 응답

#     # 관련성 판단
#     is_medicine_related: Optional[bool] = None  # 약 관련 여부

#     # 과거 질문 활용
#     previous_context: Optional[str] = None  # 과거 문맥 기억용

#     # 검색 결과
#     pdf_results: Optional[List[Document]] = None
#     excel_results: Optional[List[Document]] = None
#     external_raw: Optional[str] = None  # 외부 검색 결과 원문
#     external_parsed: Optional[dict] = None  # 외부 검색 JSON 정제 결과

#     # 후처리 및 평가
#     reranked_docs: Optional[List[Document]] = None
#     relevant_docs: Optional[List[Document]] = None  # 문서 평가 후 채택된 문서
#     hallucination_flag: Optional[bool] = None  # 환각 여부
#     re_query: Optional[str] = None  # 재검색 시 질문 수정 결과

#     # 최종 생성 결과
#     final_answer: Optional[str] = None


###Claude 수정본
from typing import Optional, List, TypedDict
from langchain_core.documents import Document

class QAState(TypedDict, total=False):
    """
    LangGraph 기반 의약품 QA 시스템의 전체 상태를 관리하는 클래스입니다.

    각 노드에서 참조하거나 수정할 수 있는 주요 필드들은 다음과 같습니다:

    - 사용자 입력 관련
        - query: 사용자 원 질문
        - cleaned_query: 약품명 정제 결과
        - normalized_query: 약품명 정규화 결과

    - 전처리 정보
        - condition: 병력 정보 (ex. 위장염)
        - category: 약물 카테고리 (ex. 감기약)
        - requested_fields: 사용자가 궁금해하는 항목들 (효능, 부작용 등)

    - 추천 흐름
        - recommendation_answer: 병력 기반 약품 추천 결과

    - 관련성 판단
        - is_medicine_related: 약 관련 질문인지 여부 판단 결과

    - 과거 질문 활용
        - previous_context: 이전 질문 맥락

    - 검색 결과
        - pdf_results: PDF 검색 결과 문서 리스트
        - excel_results: Excel 검색 결과 문서 리스트
        - external_raw: 외부 검색의 원문 텍스트
        - external_parsed: 외부 검색을 LLM이 정제한 JSON 구조

    - 후처리 및 평가
        - reranked_docs: 리랭킹 결과 문서 리스트
        - relevant_docs: 문서 평가 후 관련성 높은 문서 리스트
        - hallucination_flag: 환각 여부 판단 결과
        - re_query: 재검색 시 새로 생성된 질문

    - 최종 생성 결과
        - final_answer: LLM이 생성한 최종 응답 텍스트
     """ 
    # 사용자 입력 관련 (query만 필수, 나머지는 선택)
    query: str  # 원 질문
    cleaned_query: Optional[str]  # 약품명 정제
    normalized_query: Optional[str]  # 약품명 정규화

    # 전처리 정보
    condition: Optional[List[str]]  # 병력
    category: Optional[str]  # 약물 카테고리
    requested_fields: Optional[List[str]]  # 사용자가 요청한 항목들

    # 추천 흐름
    recommendation_answer: Optional[str]  # 병력 기반 추천 응답

    # 관련성 판단
    is_medicine_related: Optional[bool]  # 약 관련 여부

    # 과거 질문 활용
    previous_context: Optional[str]  # 과거 문맥 기억용

    # 검색 결과
    pdf_results: Optional[List[Document]]
    excel_results: Optional[List[Document]]
    external_raw: Optional[str]  # 외부 검색 결과 원문
    external_parsed: Optional[dict]  # 외부 검색 JSON 정제 결과

    # 후처리 및 평가
    reranked_docs: Optional[List[Document]]
    relevant_docs: Optional[List[Document]]  # 문서 평가 후 채택된 문서
    hallucination_flag: Optional[bool]  # 환각 여부
    re_query: Optional[str]  # 재검색 시 질문 수정 결과

    # 최종 생성 결과
    final_answer: Optional[str]