import os
import re
import difflib
import json
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader

# 유틸 함수
def normalize(text):
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^\w가-힣]", "", text)
    return re.sub(r"\s+", "", text.strip().lower())

def clean_product_name(query):
    query = re.sub(r"(에 대해.*|알려줘|무엇입니까|뭔가요|뭐야|어떻게.*|사용법|부작용|효능|복용.*|섭취.*|투여.*)", "", query)
    query = re.sub(r"[^\w가-힣]", "", query)
    query = re.sub(r"(은|는|이|가|을|를)$", "", query)
    return query.strip()

def is_product_match(query, target_name):
    nq, nt = normalize(query), normalize(target_name)
    return nq in nt or nt in nq or difflib.SequenceMatcher(None, nq, nt).ratio() > 0.85

def extract_field_from_docs(docs, label):
    pattern = rf"\[{label}\]:\s*((?:.|\n)*?)(?=\n\[|\Z)"
    candidates = []
    for doc in docs:
        match = re.search(pattern, doc.page_content)
        if match:
            candidates.append(match.group(1).strip())
    if candidates:
        return max(candidates, key=len)
    return "정보 없음"

# LLM 초기화
load_dotenv(".env")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["TAVILY_API_KEY"] = "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"
llm = ChatOpenAI(model="gpt-4o", temperature=0)
hf_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=hf_model, top_n=5)
splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=50)

condition_risk_map = {
    "위장염": ["위장 자극", "속쓰림", "구토", "소화불량"],
    "간질환": ["간독성", "간 손상"],
    "고혈압": ["혈압 상승", "심장 부담"]
}

def extract_condition_and_category(query):
    prompt = f"""
다음 사용자 질문에서 병력과 약물 종류를 반드시 JSON 형식으로 추출하세요.

반드시 아래 예시처럼 JSON만 출력하세요. 자연어 설명 없이 JSON만 반환해야 합니다.

예시:
{{
  "병력": "위장염",
  "약물종류": "감기약"
}}

질문: "{query}"

반환:
"""
    try:
        response = llm.invoke(prompt)
        raw_content = response.content.strip()
        print("📤 LLM 응답 내용 확인:", raw_content)

        # ✅ LLM이 ```json ... ``` 으로 감싸서 보냈을 경우 제거
        if raw_content.startswith("```"):
            raw_content = re.sub(r"```[a-zA-Z]*", "", raw_content).strip()  # ```json 제거
            raw_content = raw_content.replace("```", "").strip()            # 마지막 ``` 제거

        return json.loads(raw_content)  # 이제 파싱 성공!
    except Exception as e:
        print("❌ JSON 파싱 실패:", e)
        return {"병력": "", "약물종류": ""}

def generate_recommendation_response(condition, category, candidates):
    prompt = f"""
당신은 건강 상태에 맞는 약을 추천해주는 건강 상담사입니다.
사용자는 현재 '{condition}'이 있고, '{category}'에 도움이 되는 약을 찾고 있습니다.

아래는 병력에 따라 부작용 위험이 낮은 추천 후보들입니다. 각 약에 대해 부드럽고 신뢰감 있게 설명해주세요.
추천 이유도 병력과 관련지어 주세요.

추천 후보:
{json.dumps(candidates, ensure_ascii=False)}

답변:
"""
    return llm.invoke(prompt).content

# 필드 감지 (LLM 기반)
def detect_requested_fields_llm(query):
    prompt = f"""
아래의 질문에서 사용자가 궁금해하는 항목이 무엇인지 추론하세요.
항목은 [\"효능\", \"부작용\", \"사용법\"] 중 해당되는 것만 포함하세요.
결과는 JSON 배열 형식으로 주세요.

질문: \"{query}\"
결과:
"""
    try:
        response = llm.invoke(prompt)
        return json.loads(response.content.strip())
    except:
        return ["효능", "부작용", "사용법"]

# 답변 생성 (LLM 표현)
def generate_response_llm(name, fields, eff, side, usage):
    context = {"제품명": name, "효능": eff, "부작용": side, "사용법": usage}
    field_info = {k: v for k, v in context.items() if k in fields and v not in ["정보 없음", ""]}
    prompt = f"""
당신은 따뜻하고 정확한 건강 상담사입니다.
아래에 주어진 약품 정보(JSON 형식)를 사용자에게 부드럽고 신뢰감 있게 설명해주세요.
제공된 정보만 사용하고 추가적인 추론은 하지 마세요.

요청 항목: {fields}
정보:
{json.dumps(field_info, ensure_ascii=False)}

답변:
"""
    return llm.invoke(prompt).content

# 외부 검색 결과 요약 (LLM JSON 구조)
def summarize_structured_json(text):
    prompt = f"""
다음 약품 관련 텍스트에서 항목별 정보를 JSON 형식으로 정리해줘.
항목은 '제품명', '효능', '부작용', '사용법'이며, 없으면 \"정보 없음\"으로 표기해줘.

텍스트:
{text}

결과 형식:
{{
  "제품명": "...",
  "효능": "...",
  "부작용": "...",
  "사용법": "..."
}}
"""
    try:
        response = llm.invoke(prompt)
        return json.loads(response.content)
    except:
        return {"제품명": "", "효능": "정보 없음", "부작용": "정보 없음", "사용법": "정보 없음"}

# PDF 인덱싱
pdf_path = "C:\\Users\\jung\\Desktop\\pdf\\한국에서 널리 쓰이는 일반의약품 20선.pdf"
pdf_docs_raw = PyPDFLoader(pdf_path).load()
pdf_structured_docs, pdf_product_index = [], {}
for doc in pdf_docs_raw:
    blocks = re.findall(r"(\d+\.\s*.+?)(?=\n\d+\.|\Z)", doc.page_content, re.DOTALL)
    for block in blocks:
        name_match = re.match(r"\d+\.\s*([^\n(]+)", block)
        if name_match:
            name = name_match.group(1).strip()
            eff = re.search(r"주요 효능[:：]\s*(.*?)(?:\n|일반적인 부작용[:：])", block, re.DOTALL)
            side = re.search(r"일반적인 부작용[:：]\s*(.*?)(?:\n|성인 기준 복용법[:：])", block, re.DOTALL)
            usage = re.search(r"성인 기준 복용법[:：]\s*(.*?)(?:\n|$)", block, re.DOTALL)
            content = f"[제품명]: {name}\n[효능]: {eff.group(1).strip() if eff else '정보 없음'}\n[부작용]: {side.group(1).strip() if side else '정보 없음'}\n[사용법]: {usage.group(1).strip() if usage else '정보 없음'}"
            for chunk in splitter.split_text(content):
                doc_obj = Document(page_content=chunk, metadata={"제품명": name})
                pdf_structured_docs.append(doc_obj)
                pdf_product_index.setdefault(normalize(name), []).append(doc_obj)
pdf_vectordb = FAISS.from_documents(pdf_structured_docs, OpenAIEmbeddings())
pdf_retriever = ContextualCompressionRetriever(
    base_retriever=pdf_vectordb.as_retriever(search_type="similarity", k=20),
    base_compressor=compressor
)

# Excel 인덱싱
excel_files = [rf"C:\\Users\\jung\\Desktop\\11\\e약은요정보검색{i}.xlsx" for i in range(1, 11)]
required_columns = ["제품명", "이 약의 효능은 무엇입니까?", "이 약은 어떤 이상반응이 나타날 수 있습니까?", "이 약은 어떻게 사용합니까?"]
doc_list, product_names, product_names_normalized = [], [], []
for file in excel_files:
    if not os.path.exists(file): continue
    df = pd.read_excel(file)
    if not all(col in df.columns for col in required_columns): continue
    df = df[required_columns].fillna("정보 없음")
    for _, row in df.iterrows():
        name = row["제품명"].strip()
        product_names.append(name)
        product_names_normalized.append(normalize(name))
        content = f"[제품명]: {name}\n[효능]: {row['이 약의 효능은 무엇입니까?']}\n[부작용]: {row['이 약은 어떤 이상반응이 나타날 수 있습니까?']}\n[사용법]: {row['이 약은 어떻게 사용합니까?']}"
        for chunk in splitter.split_text(content):
            doc_list.append(Document(page_content=f"[제품명]: {name}\n{chunk}", metadata={"제품명": name}))
vectordb = FAISS.from_documents(doc_list, OpenAIEmbeddings())
excel_retriever = ContextualCompressionRetriever(
    base_retriever=vectordb.as_retriever(search_type="similarity", k=20),
    base_compressor=compressor
)

# 외부 검색 및 평가
search_agent = initialize_agent(tools=[TavilySearchResults()], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, verbose=False)
eval_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["answer", "query"],
        template="질문: {query}\n답변: {answer}\n\n신뢰도 평가:\n- 신뢰도: [높음 / 중간 / 낮음]\n- 이유:"
    )
)

ragas_records = []

def _log_answer(query, docs, fields, override_name=None):
    name = override_name or extract_field_from_docs(docs, "제품명") or clean_product_name(query)
    answer = generate_response_llm(
        name, fields,
        extract_field_from_docs(docs, "효능"),
        extract_field_from_docs(docs, "부작용"),
        extract_field_from_docs(docs, "사용법")
    )
    ragas_records.append({
        "query": query,
        "contexts": [doc.page_content for doc in docs],
        "answer": answer
    })
    return answer

def qa_agent(query):
    original_query = query
    condition_data = extract_condition_and_category(query)
    print("🧪 병력 추출 결과:", condition_data)
    condition = condition_data.get("병력", "").strip()
    category = condition_data.get("약물종류", "").strip()

    if condition and category:
        risk_keywords = condition_risk_map.get(condition, [])
        keyword_candidates = []

        for doc in pdf_structured_docs:
            name = doc.metadata.get("제품명", "")
            content = doc.page_content
            if category in content or any(k in content for k in ["감기", "기침", "콧물", "해열"]):
                if not any(risk in content for risk in risk_keywords):
                    rec = {
                        "제품명": name,
                        "효능": extract_field_from_docs([doc], "효능"),
                        "부작용": extract_field_from_docs([doc], "부작용"),
                        "사용법": extract_field_from_docs([doc], "사용법")
                    }
                    keyword_candidates.append(rec)

                    print("🔎 추천 후보 약물 개수:", len(keyword_candidates))  # 여기를 추가하세요
                    print("📋 추천 후보 예시:", keyword_candidates[:2])  # 일부 예시 보기

        if keyword_candidates:
            return generate_recommendation_response(condition, category, keyword_candidates[:3])
    stripped = clean_product_name(query)
    normalized_query = normalize(stripped)
    requested_fields = detect_requested_fields_llm(original_query)

    matches = pdf_product_index.get(normalized_query)
    if not matches:
        suggestion = difflib.get_close_matches(normalized_query, list(pdf_product_index.keys()), n=1, cutoff=0.85)
        if suggestion:
            matches = pdf_product_index[suggestion[0]]
    if matches:
        return _log_answer(query, matches, requested_fields)

    pdf_results = pdf_retriever.get_relevant_documents(stripped)
    matches = [doc for doc in pdf_results if is_product_match(stripped, doc.metadata.get("제품명", ""))]
    if matches:
        return _log_answer(query, matches, requested_fields)

    exact_product = None
    if normalized_query in product_names_normalized:
        idx = product_names_normalized.index(normalized_query)
        exact_product = product_names[idx]
    else:
        suggestions = difflib.get_close_matches(normalized_query, product_names_normalized, n=1, cutoff=0.85)
        if suggestions:
            idx = product_names_normalized.index(suggestions[0])
            exact_product = product_names[idx]

    if exact_product:
        results = excel_retriever.get_relevant_documents(exact_product)
        filtered = [doc for doc in results if normalize(doc.metadata.get("제품명", "")) == normalized_query]
        return _log_answer(query, filtered or results, requested_fields, override_name=exact_product)

    search_query = f"site:mfds.go.kr OR site:health.kr {stripped}"
    raw = search_agent.run(search_query)

    with ThreadPoolExecutor() as executor:
        summary_data = executor.submit(summarize_structured_json, raw).result()
        evaluation = executor.submit(eval_chain.invoke, {"answer": json.dumps(summary_data, ensure_ascii=False), "query": query}).result()

    answer = generate_response_llm(
        summary_data["제품명"] or stripped,
        requested_fields,
        summary_data["효능"],
        summary_data["부작용"],
        summary_data["사용법"]
    ) + f"\n\n🔬 평가\n{evaluation['text']}"

    ragas_records.append({
        "query": original_query,
        "contexts": [raw],
        "answer": answer
    })
    return answer


demo = gr.Interface(
    fn=qa_agent,
    inputs=gr.Textbox(lines=2, placeholder="예: 슬림락정의 사용법은?"),
    outputs="text",
    title="💊 의약품 정보 상담 에이전트 (LLM 기반 RAG with 평가)",
    description="PDF → Excel → 외부 검색 순서로 정보를 제공합니다."
)
demo.launch()

with open("ragas_records.json", "w", encoding="utf-8") as f:
    json.dump(ragas_records, f, ensure_ascii=False, indent=2)
