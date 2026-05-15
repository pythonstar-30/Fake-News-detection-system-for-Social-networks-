import os
import pandas as pd
from xml.parsers.expat import model
import requests
from dotenv import load_dotenv
from collections import defaultdict
from newspaper import Article
import numpy as np
import textstat
import torch
from typing import Optional
from pydantic import BaseModel, Field
import streamlit as st 
import plotly.graph_objects as go
import math
# LangChain & Ollama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch

# Transformers / Torch
from transformers import pipeline, AutoTokenizer ,AutoModelForSequenceClassification
import torch

#MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_NAME = "Stremie/bert-base-uncased-clickbait"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


import   prompts_new as prompts

load_dotenv() 


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PAGERANK_API_KEY = os.getenv("PAGERANK_API_KEY")

SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL")

llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant",
    api_key= os.environ["GROQ_API_KEY"]
)



class StanceOutput(BaseModel):
    """Stance detection result."""
    stance_label: str = Field(description="One of: Favor, Against, Neutral")
    dissonance_score: float = Field(description="0.0 to 1.0, higher means body contradicts or is unrelated to headline")
    reasoning: str = Field(description="Brief explanation of the stance decision")

class SubjectivityOutput(BaseModel):
    """Subjectivity analysis result."""
    entities_identified: list[str] = Field(description="List of named entities found")
    final_category: str = Field(description="Objective or Subjective")
    score: float = Field(description="0.0 to 1.0")
    reasoning: str = Field(description="Explanation of the classification")

class HedgingOutput(BaseModel):
    """Hedging detection result."""
    hedging_detected: bool = Field(description="Whether hedging language was found")
    hedging_score: float = Field(description="0.0 to 1.0")
    reasoning: str = Field(description="Step-by-step explanation and hedge words found")

class ClaimExtractionOutput(BaseModel):
    """Structured claim/summary extraction."""
    article_claim: str = Field(description="Single concise paragraph summarizing core factual content with named entities and dates preserved")

class VerificationItemOutput(BaseModel):
    """Single source verification result."""
    source_id: int = Field(description="Source index 1, 2, 3...")
    step_by_step_analysis: str = Field(description="Brief comparison of key entities, actions, numbers, timeline")
    evidence_quote: Optional[str] = Field(default=None, description="Exact quote from source; null if irrelevant")
    verdict: str = Field(description="One of: Supported, Refuted, Mixed, Irrelevant")
    confidence_score: float = Field(description="0.0 to 1.0")

class VerificationListOutput(BaseModel):
    """List of verification results for each source."""
    results: list[VerificationItemOutput] = Field(description="Verification result for each evidence source")

class FinalJudgeOutput(BaseModel):
    """Final veracity judgment."""
    veracity_label: str = Field(description="One of: Real, Fake, Probable Real, Probable Fake")
    reliability_score: float = Field(description="0.0 to 1.0, how much to trust this article")
    data_quality_weight: float = Field(description="0.0 to 1.0, how useful is this data for training")
    justification: str = Field(description="Concise summary (max 6 sentences) referencing PageRank, Fact-Check, and linguistic signals")



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
        result = sentiment_pipeline(chunk, truncation=True, max_length=512)
        chunk_sentiments.append(result)
        sentiment_details.append(result[0]['score'])
        
    # Majority Voting Logic
    scores = defaultdict(float)
    for chunk in chunk_sentiments:
        c = chunk[0]
        scores[c['label']] += c['score']

    total_score = sum(scores.values())
    normalized_scores = {}
    
    if total_score > 0:
        for label, score in scores.items():
            normalized_scores[label] = score / total_score
    else:
        normalized_scores = dict(scores)

 
    final_label = max(normalized_scores, key=normalized_scores.get) if normalized_scores else "UNKNOWN"


    return {
        "final_label": final_label,
        "scores": dict(normalized_scores),
        "chunk_details": chunk_sentiments
    }


def run_llm_structured_chain(template_text, inputs, schema_class):
    """Run chain that compulsorily returns JSON (dict) matching the Pydantic schema."""
    prompt = ChatPromptTemplate.from_template(template_text)
    structured_llm = llm.with_structured_output(schema_class)
    chain = prompt | structured_llm
    result = chain.invoke(inputs)
    return result.model_dump() if hasattr(result, "model_dump") else result

def analyze_stance(headline, body_text):
    print("--- Running Stance Detection ---")
    return run_llm_structured_chain(
        prompts.head_body_stance_template,
        {"headline": headline, "body_text": body_text},
        StanceOutput,
    )

def analyze_subjectivity(text):
    print("--- Running Subjectivity Analysis ---")
    return run_llm_structured_chain(prompts.subjectivity_template, {"text": text}, SubjectivityOutput)

def analyze_hedging(text):
    print("--- Running Hedging Detection ---")
    return run_llm_structured_chain(prompts.hedge_template, {"text": text}, HedgingOutput)


CLAIM_EXTRACTION_JSON_TEMPLATE = """Task: Summarize the core factual content of text S into a single concise paragraph.
Constraint 1: Preserve all Named Entities (People, Countries, Organizations) and specific dates/events.
Constraint 2: Do not output reasoning, thoughts, or steps.
Return ONLY a valid JSON object with exactly one key: "article_claim", whose value is that summary paragraph. No other text.

S: {text}
"""

def extract_article_claim(text):
    """Extract article claim as structured JSON (article_claim string)."""
    print("--- Extracting Article Claim (structured JSON) ---")
    out = run_llm_structured_chain(CLAIM_EXTRACTION_JSON_TEMPLATE, {"text": text}, ClaimExtractionOutput)
    if isinstance(out, dict):
        return out.get("article_claim", "")
    return getattr(out, "article_claim", "")

clickbait_analyzer = pipeline("text-classification", model="Stremie/bert-base-uncased-clickbait")

def analyze_clickbait(headline):
    """
    Uses BERT to determine if a headline is clickbait.
    Returns a score between 0 and 1.
    """
    if not headline or not str(headline).strip():
        print("--- Running Clickbait Detection: skipped (empty headline) ---")
        return {"clickbait_detected": False, "clickbait_score": 0.0}
    print("--- Running Clickbait Detection (BERT) ---")
    try:
        
        result = clickbait_analyzer(headline)[0]
        
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



def get_pagerank(url):
    print("--- Fetching PageRank ---")
    try:
        domain = url.split("//")[-1].split("/")[0]
        api_url = "https://openpagerank.com/api/v1.0/getPageRank"
        headers = {"API-OPR": PAGERANK_API_KEY}
        params = [("domains[]", domain)]
        response = requests.get(api_url, headers=headers, params=params)
        data = response.json()
        
        rank_data = data.get("response", [])
        if rank_data and isinstance(rank_data, list):
            return rank_data[0].get("page_rank_integer", 0)
        return 0
    except Exception as e:
        print(f"PageRank Error: {e}")
        return 0

def extract_and_verify_claims(text , target_url=""):
    print("\n--- Extracting and Verifying Claims (Optimized) ---")

    article_claim = extract_article_claim(text)

    print(f"Extracted Article Claim: {(article_claim or '')[:120]}...")

    

    keywords = " ".join(article_claim.split()[:40])  

    tool = TavilySearch(max_results=3)
    search_results = tool.invoke({
        "query": f"latest news about {keywords}"
    })

    results = search_results.get("results", [])

    evidence_text = []
    url_mapping = {}

    for i, result in enumerate(results):
        url = result["url"]
        if target_url and url.rstrip('/') == target_url.rstrip('/'):
            print(f"Skipping source #{i+1}: Matches original article.")
            continue
        try:
            print(f"Scraping source #{i+1}: {url}")
            web_article = Article(url)
            web_article.download()
            web_article.parse()

            if web_article.text and len(web_article.text) > 100:
                trimmed = " ".join(web_article.text.split()[:800]) 
                evidence_text.append((i+1, trimmed))
                url_mapping[i+1] = url
                print(f" Success from source #{i+1}")

        except Exception as e:
            print(f"Error scraping source #{i+1}: {e}")
            continue

    if not evidence_text:
        return {
            "final_verdict": "Not Enough Information",
            "aggregate_confidence": 0.0,
            "evidence_count": 0,
            "vote_breakdown": {},
            "all_evidence": [],
            "evidence_quotes": []
        }
   
    combined_evidence = ""
    for idx, ev in evidence_text:
        combined_evidence += f"\n\nSOURCE {idx}:\n{ev}"

    verify_prompt = ChatPromptTemplate.from_template("""
You are a senior investigative fact-checker.

Your task is to verify the truthfulness of the ARTICLE CLAIM using multiple independent EVIDENCE SOURCES.

Treat the evidence sources as reliable factual reporting.

===========================
ARTICLE CLAIM:
{article_claim}
===========================

Below are independent evidence sources labeled as SOURCE 1, SOURCE 2, etc.

{combined_evidence}

===========================
ANALYSIS INSTRUCTIONS
===========================

For EACH source separately:

1. Identify key factual elements in the ARTICLE CLAIM:
   - Main actors
   - Actions
   - Location
   - Time references
   - Numbers or scale
   - Stated motive or cause (if any)

2. Identify the key factual elements present in the SOURCE.

3. Compare both carefully:
   - Are the same actors involved?
   - Are the actions described the same?
   - Do locations and timelines match?
   - Are numbers consistent?
   - Is any crucial context missing or altered?

4. Determine verdict for that source:
   - "Supported"
   - "Refuted"
   - "Mixed"
   - "Irrelevant"

A claim is:
- "Supported" if the source clearly confirms the claim’s core facts.
- "Refuted" if the source clearly contradicts the core facts.
- "Mixed" if partially correct but misleading, exaggerated, outdated, or missing some of the context.
- "Irrelevant" if the source does not meaningfully address the claim.

===========================
OUTPUT FORMAT
===========================

Return a JSON object with a single key "results" that is a list of entries.

Each entry must follow this structure:
  - "source_id": integer (1, 2, 3...)
  - "step_by_step_analysis": string (brief comparison of key entities, actions, numbers, timeline)
  - "evidence_quote": string or null (exact quote from source; null if irrelevant)
  - "verdict": one of "Supported", "Refuted", "Mixed", "Irrelevant"
  - "confidence_score": float 0.0 to 1.0

Do NOT add explanations outside the JSON.
""")

    verify_chain = verify_prompt | llm | JsonOutputParser()
    verification_result_obj = verify_chain.invoke({
        "article_claim": article_claim,
        "combined_evidence": combined_evidence
    })
   
    if isinstance(verification_result_obj, list):
        verification_results = verification_result_obj
    else:
        verification_results = verification_result_obj.get("results", [])

   
    score_map = {"Supported": 0.0, "Refuted": 0.0, "Mixed": 0.0}
    count_map = {"Supported": 0, "Refuted": 0, "Mixed": 0}

    ver_results = []
    evidence_list = []

    for res in verification_results:

        source_id = res.get("source_id")
        res["url"] = url_mapping.get(source_id, "")

        if res["verdict"] != "Irrelevant":
            v = res.get("verdict")
            c = res.get("confidence_score", 0.5)

            if v in score_map:
                score_map[v] += c
                count_map[v] += 1

            ver_results.append(res)
            evidence_list.append(res.get("evidence_quote"))

    if not ver_results:
        return {
            "final_verdict": "Not Enough Information",
            "aggregate_confidence": 0.0,
            "evidence_count": 0,
            "vote_breakdown": count_map,
            "all_evidence": [],
            "evidence_quotes": []
        }

    winner = max(score_map, key=score_map.get)
    total_score = sum(score_map.values())

    final_confidence = (
        score_map[winner] / total_score if total_score > 0 else 0.0
    )

    
    if score_map["Supported"] > 0 and score_map["Refuted"] > 0:
        diff = abs(score_map["Supported"] - score_map["Refuted"])
        if diff < (total_score * 0.2):
            winner = "Disputed"

   
    final_output = {
        "final_verdict": winner,
        "aggregate_confidence": round(final_confidence, 2),
        "evidence_count": len(ver_results),
        "vote_breakdown": count_map,
        "all_evidence": ver_results,
        "evidence_quotes": evidence_list
    }

    return final_output

def run_final_judge(data_packet):
    print("--- Running Final Judge ---")
    prompt = ChatPromptTemplate.from_template(prompts.master_judge_template)
    structured_judge_llm = llm.with_structured_output(FinalJudgeOutput)
    chain = prompt | structured_judge_llm
    result = chain.invoke(data_packet)
    return result.model_dump() if hasattr(result, "model_dump") else result



def main(url):
  
    article_data = get_article_content(url)
    text = article_data['text']
    headline = article_data['headline']
    
    if not text:
        print("No content found.")
        return

    sentiment_res = analyze_sentiment(text)
    stance_res = analyze_stance(headline, text)
    subj_res = analyze_subjectivity(text)
    hedge_res = analyze_hedging(text)
    clickbait_res = analyze_clickbait(headline)
    readability_res = analyze_readability(text)
    pagerank_score = get_pagerank(url)
    verification_res = extract_and_verify_claims(text, target_url=url)  
    print("the verification result :" , verification_res)

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

    final_verdict = run_final_judge(judge_input)
    
    return {
        "headline": headline,
        "final_verdict": final_verdict,
        "verification_res": verification_res,
        "judge_input": judge_input
    }

def run_ui():
    st.set_page_config(page_title="AI Fact Checker", page_icon="🕵️‍♂️", layout="wide")
    
    st.title("📰 AI Article Fact Checker")
    st.markdown("Analyze news articles for stance, subjectivity, clickbait, and factual accuracy using LLMs and Agentic Search.")

    # Input section
    target_url = st.text_input("Enter Article URL:", placeholder="https://example.com/news-article")

    if st.button("Analyze Article", type="primary"):
        if not target_url:
            st.warning("⚠️ Please enter a valid URL.")
            return

        # Use a spinner to show the user that the backend is working
        with st.spinner("Scraping, analyzing, and fact-checking... This might take a minute."):
            try:
                results = main(target_url)
                
                if not results:
                    st.error("Could not extract content from the URL.")
                    return

                # --- DISPLAY RESULTS ---
                st.success("Analysis Complete!")
                
                final_verdict = results["final_verdict"]
                veracity_label = final_verdict.get("veracity_label", "Unknown")
                
                # Top Level Verdict Section
                st.header(f"Headline: {results['headline']}")
                
                # Determine color based on label
                if "Fake" in veracity_label:
                    st.error(f"🚨 Verdict: **{veracity_label}**")
                elif "Real" in veracity_label:
                    st.success(f"✅ Verdict: **{veracity_label}**")
                else:
                    st.warning(f"⚠️ Verdict: **{veracity_label}**")

                gauge_chart = create_veracity_gauge(veracity_label)
                st.plotly_chart(gauge_chart, use_container_width=True)

                st.write(f"**Justification:** {final_verdict.get('justification', 'No justification provided.')}")

                st.divider()

                # Metrics Row
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Reliability Score", f"{final_verdict.get('reliability_score', 0) * 100:.1f}%")
                col2.metric("Data Quality Weight", f"{final_verdict.get('data_quality_weight', 0):.2f}")
                col3.metric("Fact-Check Confidence", f"{results['judge_input'].get('verify_conf', 0) * 100:.1f}%")
                col4.metric("PageRank", results['judge_input'].get('pagerank_score', 0))

                # Detailed Breakdown in Expanders
                with st.expander("🔍 Linguistic & Sentiment Analysis"):
                    st.markdown("### Metric Breakdown")
                    
                    judge = results.get("judge_input", {})
                    
                    # Helper functions to format the data cleanly
                    def format_bool(val):
                        if val is None: return "N/A"
                        return "Yes" if val else "No"

                    def format_pct(val):
                        if val is None: return "N/A"
                        try:
                            return f"{float(val) * 100:.1f}%"
                        except (ValueError, TypeError):
                            return str(val)

                    # NEW: Helper to format the sentiment dictionary
                    def format_sentiment(scores):
                        if not isinstance(scores, dict):
                            return str(scores) if scores else "N/A"
                        
                        parts = []
                        for key, value in scores.items():
                            # Compound scores are usually -1 to 1, so we format them differently
                            if key.lower() == 'compound':
                                parts.append(f"Compound: {value:.2f}")
                            else:
                                parts.append(f"{key.capitalize()}: {format_pct(value)}")
                        
                        return " | ".join(parts)

                    # Map exactly to your backend dictionary
                    analysis_data = [
                        {
                            "Metric": "Stance", 
                            "Result": str(judge.get("stance_label", "N/A")).title(), 
                            "Score / Confidence": f"Dissonance: {format_pct(judge.get('dissonance_score'))}" 
                        },
                        {
                            "Metric": "Subjectivity", 
                            "Result": str(judge.get("subjectivity_type", "N/A")).title(), 
                            "Score / Confidence": format_pct(judge.get("subjectivity_score"))
                        },
                        {
                            "Metric": "Clickbait", 
                            "Result": format_bool(judge.get("clickbait_detected")), 
                            "Score / Confidence": format_pct(judge.get("clickbait_score"))
                        },
                        {
                            "Metric": "Hedging (Uncertainty)", 
                            "Result": format_bool(judge.get("hedging_detected")), 
                            "Score / Confidence": format_pct(judge.get("hedging_score"))
                        },
                        {
                            "Metric": "Readability", 
                            "Result": str(judge.get("readability_complexity", "N/A")).title(), 
                            "Score / Confidence": f"Flesch Score: {judge.get('readability_score', 'N/A')}" 
                        },
                        {
                            "Metric": "Sentiment", 
                            "Result": "Analyzed", 
                            "Score / Confidence": format_sentiment(judge.get("sentiment_scores")) 
                        }
                    ]
                    df = pd.DataFrame(analysis_data)
                    
                    df.index = df.index + 1 
                    
                    header_styles = [{
                        'selector': 'th',
                        'props': [
                            ('color', 'black'), 
                            ('font-weight', 'bold')
                        ]
                    }]

                   
                    styled_df = df.style.set_properties(**{
                        'color': 'black',
                        'font-weight': 'bold'
                    }).set_table_styles(header_styles)
                    
                   
                    st.table(styled_df)


                with st.expander("🧾 Evidence & Source Verification"):
                    verification_data = results["verification_res"]
                    
                    st.markdown(f"###  Final Evidence Verdict: **{verification_data.get('final_verdict', 'N/A')}**")
                    st.write(f"**Sources Checked:** {verification_data.get('evidence_count', 0)}")
                    st.divider()
                    
                    for i, ev in enumerate(verification_data.get("all_evidence", [])):
                        
                        source_id = ev.get('source_id', f"Source {i+1}")
                        verdict = ev.get('verdict', 'Unknown')
                        
                        st.markdown(f"#### {source_id} - Verdict: {verdict}")
                       
                        source_url = ev.get('url', ev.get('source_url', ''))
                        if source_url:
                            st.markdown(f"🔗 **Link:** [{source_url}]({source_url})")
                        
                        st.write(f"**Analysis:** {ev.get('step_by_step_analysis', 'No analysis provided.')}")
                        
                        if ev.get('evidence_quote'):
                            st.info(f"**Quote:** \"{ev.get('evidence_quote')}\"")
                            
                        st.divider()

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
def create_veracity_gauge(veracity_label):
    label_lower = str(veracity_label).lower()
    
    # Map the label to a point on a 0-100 scale
    if "probable fake" in label_lower:
        val = 37.5
    elif "fake" in label_lower:
        val = 12.5
    elif "probable real" in label_lower:
        val = 62.5
    elif "real" in label_lower:
        val = 87.5
    else:
        val = 50  
        
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Veracity Meter", 'font': {'size': 20}},
        gauge={
            'axis': {
                'range': [0, 100], 
                'tickvals': [12.5, 37.5, 62.5, 87.5], 
                'ticktext': ["Fake", "Probable Fake", "Probable Real", "Real"]
            },
            'bar': {'thickness': 0}, 
            'steps': [
                {'range': [0, 25], 'color': "#FF4B4B"},   # Streamlit Red
                {'range': [25, 50], 'color': "#FFA421"},  # Orange
                {'range': [50, 75], 'color': "#90EE90"},  # Light Green
                {'range': [75, 100], 'color': "#00C853"}  # Dark Green
            ],
           
            'threshold': {
                'line': {'color': "black", 'width': 7}, # Bold black marker
                'thickness': 0.8, # Spans across 80% of the color band's thickness
                'value': val
            }
        }
    ))
    
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

if __name__ == "__main__":
    # Streamlit requires running via the CLI command `streamlit run script.py`.
    # This block ensures that if you accidentally run `python script.py`, it tells you what to do.
    import sys
    if "streamlit" not in sys.modules:
        print("To run the UI, please execute: streamlit run <your_script_name>.py")
    else:
        run_ui()