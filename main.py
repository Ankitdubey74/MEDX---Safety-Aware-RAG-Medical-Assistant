import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation import run_quick_eval
from tools import symptom_risk_score, bmi_calculator, emergency_flag
from query_rewriter import rewrite_query
from failure_logger import log_model_failure
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ✅ Fixed Document class
class Document:
    def __init__(self, page_content, metadata=None, doc_id=None):
        self.page_content = page_content
        self.metadata = metadata or {}
        self.id = doc_id or id(self)

# 🔹 Embeddings + Vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="vectorstore", embedding_function=embeddings)

# 🔹 LLM
MODEL_PATH = r"D:\\PROJECTS\\MEDX\\phi2-medx-merged\\phi2-medx-merged"

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token  # VERY IMPORTANT

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,   # CPU SAFE
    device_map=None,             # CPU
    trust_remote_code=True
)

model.eval()


def llm_invoke(prompt: str) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768   # CPU friendly
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,   # CPU friendly
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )




# 🔹 Vector retriever
vector_retriever = db.as_retriever(search_kwargs={"k": 5})

# 🔹 BM25 retriever
all_docs = [Document(d) if not isinstance(d, Document) else d for d in db.get()["documents"]]
bm25_retriever = BM25Retriever.from_documents(all_docs, k=5)

# 🔹 Simple Hybrid retriever
class SimpleHybridRetriever:
    def __init__(self, retrievers, weights=None):
        self.retrievers = retrievers
        self.weights = weights or [1/len(retrievers)]*len(retrievers)

    def get_relevant_documents(self, query):
        docs = []
        for r in self.retrievers:
            docs.extend(r._get_relevant_documents(query, run_manager=None))
        # Deduplicate
        seen = set()
        unique_docs = []
        for d in docs:
            if d.page_content not in seen:
                unique_docs.append(d)
                seen.add(d.page_content)
        return unique_docs[:3]

hybrid_retriever = SimpleHybridRetriever([vector_retriever, bm25_retriever])

def simple_stem(word: str) -> str:
    """
    Very lightweight stemmer for medical terms
    """
    if word.endswith("ies"):
        return word[:-3] + "y"   # allergies → allergy
    if word.endswith("s") and len(word) > 4:
        return word[:-1]         # symptoms → symptom
    return word

def normalize_text(text: str) -> set:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    stemmed = [simple_stem(t) for t in tokens]
    return set(stemmed)

def context_confidence(query, docs):
    if not docs:
        return 0.0

    query_terms = normalize_text(query)
    matched_terms = set()

    for d in docs:
        doc_terms = normalize_text(d.page_content)
        matched_terms |= (query_terms & doc_terms)

    return len(matched_terms) / max(len(query_terms), 1)

def is_self_reported(text: str) -> bool:
    """
    Check if user is talking about their own symptoms
    """
    indicators = [
        "i have", "i am", "i feel", "my", 
        "since", "from morning", "experiencing"
    ]
    text = text.lower()
    return any(indicator in text for indicator in indicators)

def is_weak_query(query: str) -> bool:
    """
    Weak query = too short OR unclear
    """
    tokens = query.lower().split()

    if len(tokens) <= 3:
        return True

    weak_keywords = ["signs", "problem", "issue", "condition"]
    if any(w in tokens for w in weak_keywords):
        return True

    return False

# 🔹 Query function (TOOL CALLING ADDED)
def ask_medx(question, return_contexts=False):

    # 🔴 LEVEL-0: EMERGENCY TOOL
    if is_self_reported(question) and any(
        e in question.lower()
        for e in ["cannot breathe", "unconscious", "severe pain"]
    ):
        tool_result = emergency_flag(question)
        return f"**EMERGENCY CHECK:** {tool_result['action']}"

    # 🟠 LEVEL-0: RISK SCORING TOOL
    if is_self_reported(question) and any(
        s in question.lower()
        for s in ["chest pain", "shortness", "nausea", "dizzy"]
    ):
        tool_result = symptom_risk_score(question)
        return (
            f"**Risk Assessment:** {tool_result['risk_level']} "
            f"(Score: {tool_result['risk_score']:.1%})\n"
            f"{tool_result['recommendation']}"
        )

    # 🟢 LEVEL-0: BMI TOOL
    if any(k in question.lower() for k in ["bmi", "weight", "height"]):
        numbers = re.findall(r"\d+", question)
        if len(numbers) >= 2:
            weight = float(numbers[0])
            height = float(numbers[1])
            tool_result = bmi_calculator(weight, height)
            return (
                f"**BMI:** {tool_result['bmi']}\n"
                f"**Category:** {tool_result['category']}\n"
                f"**Health Risk:** {tool_result['health_risk']}\n"
                f"{tool_result['advice']}"
            )

    # 🔹 LEVEL-1: Rewrite weak query only
    if is_weak_query(question):
        rewritten_question = rewrite_query(question)
        print(f"[REWRITE] {question} → {rewritten_question}")
    else:
        rewritten_question = question

    # 🔹 Retrieve documents
    docs = hybrid_retriever.get_relevant_documents(rewritten_question)
    confidence = context_confidence(rewritten_question, docs)

    # 🔴 LEVEL-3: No usable context
    if confidence < 0.15:
        answer = "Based on the available documents, limited information is available."
        return (answer, []) if return_contexts else answer

    # 🟡 LEVEL-2: Weak but usable
    elif confidence < 0.35:
        mode = "LIKELY"

    # 🟢 LEVEL-1: Strong context
    else:
        mode = "STRICT"


    if mode in ["WEAK", "LIKELY"]:
      log_model_failure(
        query=rewritten_question,
        confidence=confidence,
        mode=mode,
        used_tools=[]
       )

    contexts = [d.page_content for d in docs]
    context_text = "\n\n".join(contexts)

    # 🔹 Prompt (UNCHANGED)
    prompt = f"""
SYSTEM:
You are MedX, a medical retrieval-based assistant.

MODE: {mode}

RULES:
- Use ONLY the provided CONTEXT
- Do NOT say "limited information" if symptoms are present in context
- Extract symptoms directly if disease name matches
- Bullet points only
- No explanations
- No hallucinations

CONTEXT:
{context_text}

QUESTION:
{rewritten_question}

ANSWER:
"""

    answer = llm_invoke(prompt)

    if return_contexts:
        return answer, contexts
    return answer

# 🔹 Test
if __name__ == "__main__":
    print(ask_medx("Allergy signs?"))
    
    print("\n" + "="*60)
    run_quick_eval(ask_medx, num_samples=2)

