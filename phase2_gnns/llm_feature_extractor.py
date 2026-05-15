import os
from xml.parsers.expat import model
import requests
from dotenv import load_dotenv
from collections import defaultdict
from newspaper import Article
import numpy as np
import textstat
import torch

# LangChain & Ollama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch

# Transformers / Torch
from transformers import pipeline, AutoTokenizer ,AutoModelForSequenceClassification
import torch

#MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_NAME = "Stremie/bert-base-uncased-clickbait"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Import prompts from the separate file
import   phase1.prompts_new as prompts
# 1. Load the variables from your .env file
load_dotenv() 

# 2. Retrieve the key securely
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# --- Configuration ---
# You can set these via environment variables or hardcode them here
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PAGERANK_API_KEY = os.getenv("PAGERANK_API_KEY")

SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL")

llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant",
    api_key= os.environ["GROQ_API_KEY"]
)

# --- 1. Utilities ---

def get_article_content(url):
    """Downloads and parses article content."""
    article = Article(url)
    article.download()
    article.parse()
    return {
        "text": article.text,
        "headline": article.title,
        "url": url
    }

def get_token_count(text, tokenizer):
    return len(tokenizer.encode(text, truncation=False))

def recursive_chunking(text, tokenizer, text_splitter, chunk_list, max_tokens=512):
    """Recursively split text into <= max_tokens chunks."""
    if get_token_count(text, tokenizer) <= max_tokens:
        chunk_list.append(text)
    else:
        sub_chunks = text_splitter.split_text(text)
        for sub_chunk in sub_chunks:
            recursive_chunking(sub_chunk, tokenizer, text_splitter, chunk_list, max_tokens)

# --- 2. Sentiment Analysis ---

def analyze_sentiment(text, model_name=SENTIMENT_MODEL):
    """
    Chunks text, runs sentiment analysis, and returns aggregated score.
    """
    print("--- Running Sentiment Analysis ---")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=-1) # -1 for CPU, 0 for GPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, 
        separators=["\n\n", "\n", ". ", ", ", " ", "-"]
    )
    
    final_chunks = []
    recursive_chunking(text, tokenizer, text_splitter, final_chunks, 400)
    
    chunk_sentiments = []
    sentiment_details = []
    
    for chunk in final_chunks:
        # Truncation ensures safety if chunking slightly missed
        result = sentiment_pipeline(chunk, truncation=True, max_length=512)
        chunk_sentiments.append(result)
        sentiment_details.append(result[0]['score'])
        
    # Majority Voting Logic
    scores = defaultdict(float)
    for chunk in chunk_sentiments:
        c = chunk[0]
        scores[c['label']] += c['score']

    final_label = max(scores, key=scores.get)
    return {
        "final_label": final_label,
        "scores": dict(scores),
        "chunk_details": chunk_sentiments
    }

# --- 3. LLM-Based Feature Extraction Steps ---

def run_llm_json_chain(template_text, inputs):
    """Helper to run a JSON parsing chain."""
    prompt = ChatPromptTemplate.from_template(template_text)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain.invoke(inputs)

def run_llm_str_chain(template_text, inputs):
    """Helper to run a string output chain."""
    # Note: Using from_messages for the claim templates as per original code style
    prompt = ChatPromptTemplate.from_messages([
        ("system", template_text),
        ("human", "S: {text}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(inputs)

def analyze_stance(headline, body_text):
    print("--- Running Stance Detection ---")
    return run_llm_json_chain(prompts.head_body_stance_template, {
        "headline": headline, 
        "body_text": body_text
    })

def analyze_subjectivity(text):
    print("--- Running Subjectivity Analysis ---")
    return run_llm_json_chain(prompts.subjectivity_template, {"text": text})

def analyze_hedging(text):
    print("--- Running Hedging Detection ---")
    return run_llm_json_chain(prompts.hedge_template, {"text": text})

'''
def analyze_clickbait(headline: str):

    print("--- Running Clickbait Detection (BERT) ---") 
    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    clickbait_score = probs[0][1].item()
    return {
        "clickbait_detected": clickbait_score > 0.5,
        "clickbait_score": round(clickbait_score, 3)
    }
'''
clickbait_analyzer = pipeline("text-classification", model="Stremie/bert-base-uncased-clickbait")

def analyze_clickbait(headline):
    """
    Uses BERT to determine if a headline is clickbait.
    Returns a score between 0 and 1.
    """
    try:
        # The model returns a list like: [{'label': 'LABEL_1', 'score': 0.98}]
        result = clickbait_analyzer(headline)[0]
        
        # Determine if it's clickbait (LABEL_1 is usually clickbait)
        is_clickbait = 1 if result['label'] == 'LABEL_1' else 0
        score = result['score'] if is_clickbait == 1 else 1 - result['score']

        return {
            "clickbait_detected": bool(is_clickbait),
            "clickbait_score": round(score, 4)
        }
    except Exception as e:
        print(f"Error in BERT analysis: {e}")
        return {"clickbait_detected": False, "clickbait_score": 0.0}

def analyze_readability(text):
    """
    Returns standard readability metrics using classical NLP formulas.
    """
    print("--- Running Readability Analysis (Flesch) ---")

    # Safety check
    if not text or len(text.strip()) < 50:
        return {
            "flesch_reading_ease": None,
            "flesch_kincaid_grade": None,
            "readability_level": "unknown"
        }

    fre = textstat.flesch_reading_ease(text)
    fkgl = textstat.flesch_kincaid_grade(text)

    # Human-readable bucket (useful later)
    if fre >= 70:
        level = "easy"
    elif fre >= 30:
        level = "moderate"
    else:
        level = "hard"

    return {
        "flesch_reading_ease": round(fre, 2),
        "flesch_kincaid_grade": round(fkgl, 2),
        "readability_level": level
    }

# --- 4. Source & Fact Checking ---

def get_pagerank(url):
    print("--- Fetching PageRank ---")
    try:
        domain = url.split("//")[-1].split("/")[0]
        api_url = "https://openpagerank.com/api/v1.0/getPageRank"
        headers = {"API-OPR": PAGERANK_API_KEY}
        params = [("domains[]", domain)]
        response = requests.get(api_url, headers=headers, params=params)
        data = response.json()
        
        # Safe extraction
        rank_data = data.get("response", [])
        if rank_data and isinstance(rank_data, list):
             # Original code returned integer, can be 0-10 or similar
            return rank_data[0].get("page_rank_integer", 0)
        return 0
    except Exception as e:
        print(f"PageRank Error: {e}")
        return 0

def extract_and_verify_claims(text):
    print("--- Extracting and Verifying Claims ---")
    
    # 1. Extract Claim from Article
    # Note: formatting inputs for ChatPromptTemplate.from_messages
    article_claim = run_llm_str_chain(prompts.claim_extraction_template, {"text": text})
    print(f"Extracted Article Claim: {article_claim[:100]}...")
    
    # 2. Extract Keywords for Search
    # Simple extraction logic from notebook
    keywords = ""
    for k in article_claim.split("\n"):
        if ":" in k:
            kw = k.split(":")[-1].strip()
            keywords += kw + " "
    if not keywords:
        keywords = article_claim[:300] # Fallback
        
    # 3. Search Tavily
    tool = TavilySearch(max_results=5)
    
    search_results = tool.invoke({"query": f"what is the latest news about {keywords.strip()}"})
    
    # 4. Get Content of Top Result
    evidence_text = []
    results = search_results.get("results", [])
    
    if not results:
        print("No search results found to verify claims.")
        evidence_text = "No external evidence found."
    else:
        # Loop through the results until we find one that works
        for i, result in enumerate(results):
            url = result['url']
            try:
                print(f"Attempting to scrape evidence from source #{i+1}: {url}")
                web_article = Article(url)
                web_article.download()
                web_article.parse()
                if web_article.text and len(web_article.text) > 50:
                    evidence_text.append((i+1, web_article.text))
                    print(f"Success! Scraped evidence from: {url}")
                 
                else:
                    print(f"Source #{i+1} returned empty content. Trying next...")
                    
            except Exception as e:
                print(f"Error scraping source #{i+1} ({url}): {e}")
                continue  
        print("total successful evidence sources:", len(evidence_text))

    
    if not evidence_text:
        evidence_text = "Could not retrieve external evidence from any of the search results."
    ver_results = []
    evidence_list = []
    for idx, ev_text in evidence_text:
        print(f"Extracting claim from evidence source #{idx}...")
        web_claim = run_llm_str_chain(prompts.web_content_template, {"text": ev_text})
        verify_prompt = ChatPromptTemplate.from_template(prompts.claim_verify_template)
        verify_chain = verify_prompt | llm | JsonOutputParser()
        verification_result = verify_chain.invoke({
        "article_claim": article_claim,
        "web_claim": web_claim
        })
        verification_result['source_id'] = idx
        if verification_result['verdict'] == "Irrelevant":
            print(f"Source #{idx} deemed Irrelevant. Skipping.")
        else:
            ver_results.append(verification_result)
            evidence_list.append(verification_result['evidence_quote'])

    if not ver_results:
        final_verdict = "Not Enough Information"
        final_confidence = 0.0
    else:
    # 1. Initialize Score Trackers
     score_map = {"Supported": 0.0, "Refuted": 0.0, "Mixed": 0.0}
     count_map = {"Supported": 0, "Refuted": 0, "Mixed": 0}

    # 2. Weighted Voting Loop
     for res in ver_results:
        v = res.get('verdict')
        c = res.get('confidence_score', 0.5) # Default to 0.5 if missing
        
        if v in score_map:
            score_map[v] += c  # Add the confidence score to the bucket
            count_map[v] += 1

     print(f"Voting Scores: {score_map}")

    # 3. Determine the Winner
    # We find the category with the highest accumulated confidence score
     winner = max(score_map, key=score_map.get)
     total_score = sum(score_map.values())
    
    # Calculate a final normalized confidence
    # (Score of Winner) / (Total Score of all votes)
    # This lowers confidence if there is disagreement (e.g. lots of Refuted vs Supported)
     if total_score > 0:
        final_confidence = score_map[winner] / total_score
     else:
        final_confidence = 0.0

    # 4. Handling Contested Scenarios (Optional Logic)
    # If 'Supported' and 'Refuted' are very close, declare it 'Disputed'
     if (score_map['Supported'] > 0 and score_map['Refuted'] > 0):
        # Check if the difference is small (e.g., less than 20% difference)
        diff = abs(score_map['Supported'] - score_map['Refuted'])
        if diff < (total_score * 0.2): 
            winner = "Disputed"

     final_verdict = winner

# 5. Construct Final Output
    final_output = {
    "final_verdict": final_verdict,
    "aggregate_confidence": round(final_confidence, 2),
    "evidence_count": len(ver_results),
    "vote_breakdown": count_map,
    "all_evidence": ver_results, # Keep the raw data for drill-down
    "evidence_quotes": evidence_list
    }

    return final_output
   


def run_final_judge(data_packet):
    print("--- Running Final Judge ---")
    prompt = ChatPromptTemplate.from_template(prompts.master_judge_template)
    chain = prompt | llm | JsonOutputParser()
    return chain.invoke(data_packet)

# --- Main Orchestrator ---

def main(url):
    # 1. Get Content
    article_data = get_article_content(url)
    text = article_data['text']
    headline = article_data['headline']
    
    if not text:
        print("No content found.")
        return

    # 2. Run Parallel/Sequential Analyses
    sentiment_res = analyze_sentiment(text)
    stance_res = analyze_stance(headline, text)
    subj_res = analyze_subjectivity(text)
    hedge_res = analyze_hedging(text)
    clickbait_res = analyze_clickbait(headline)
    readability_res = analyze_readability(text)
    pagerank_score = get_pagerank(url)
    verification_res = extract_and_verify_claims(text)
    print("the verification result :" , verification_res)

    # 3. Prepare Data Packet for Judge
    # Mapping keys from individual results to Judge input format
    judge_input = {
        "sentiment_scores": sentiment_res['scores'],
        "subjectivity_type": subj_res.get("final_category"),
        "subjectivity_score": subj_res.get("score"),
        "hedging_detected": hedge_res.get("hedging_detected"),
        "hedging_score": hedge_res.get("hedging_score"),
        "clickbait_detected": clickbait_res.get("clickbait_detected"),
        "clickbait_score": clickbait_res.get("clickbait_score"),
        "readability_score": readability_res.get("flesch_reading_ease"),
        "readability_complexity": readability_res.get("readability_level"),
        "pagerank_score": pagerank_score,
        "verdict": verification_res.get("final_verdict"),
        "verify_conf": verification_res.get("aggregate_confidence"),
        "evidence_quote" : verification_res.get("evidence_quotes"),
        "stance_label": stance_res.get("stance_label"),
        "dissonance_score": stance_res.get("dissonance_score")
    }

    # 4. Final Verdict
    final_verdict = run_final_judge(judge_input)
    
    print("\n" + "="*30)
    print("FINAL VERACITY REPORT")
    print("="*30)
    import json
    print(json.dumps(final_verdict, indent=2))

if __name__ == "__main__":
    # Test URL
    target_url = input("Enter URL: ")
    # Or ask user: target_url = input("Enter URL: ")
    main(target_url)