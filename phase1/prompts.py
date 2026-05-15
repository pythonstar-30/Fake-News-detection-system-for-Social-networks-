# prompts.py

# 1. Stance Detection
head_body_stance_template = """
<|user|>
You are a forensic linguistic auditor. Your task is to determine if an article's BODY supports or contradicts its HEADLINE.

### INPUT DATA:
**HEADLINE:** "{headline}"
**BODY TEXT:** "{body_text}"

### STANCE DEFINITIONS:
- **FAVOR (Agree):** The body text provides facts and context that directly confirm the headline's claim.
- **AGAINST (Disagree):** The body text explicitly contradicts or debunks the claim made in the headline.
- **NEUTRAL (Unrelated/Discuss):** The body text mentions the same topic but does not take a position, or it discusses a different aspect entirely (making the headline "unrelated").

### EVALUATION LOGIC:
1. Identify the core claim of the Headline.
2. Check if the Body contains evidence for that specific claim.
3. If the Body is about a different topic, label as NEUTRAL/UNRELATED.

### OUTPUT FORMAT:
Provide your output EXCLUSIVELY in a JSON object:
{{
    "stance_label": "Favor/Against/Neutral",
    "dissonance_score": 0.0 to 1.0 (Higher means the body is UNRELATED or CONTRADICTS the head),
    "reasoning": "Identify the specific claim in the head and whether the body addresses it."
}}
<|end|>
<|assistant|>
"""

# 2. Subjectivity Analysis (Zcot-In)
subjectivity_template = """
    <|user|>
    You are a professional linguistic analyst. Your task is to classify the provided text 'S' as 'Objective' or 'Subjective' based on universal journalistic principles.

    ### CLASSIFICATION CRITERIA:

    1. **OBJECTIVE (Hard News/Reporting):**
       - **Anchors:** Uses Named Entities (People, Places, Organizations, Dates) as the primary subjects.
       - **Attribution:** Uses "Speech Acts" to relay information (e.g., "X said", "According to Y", "Official records show"). 
       - **Nature:** Describes verifiable actions, events, or official statements. Even if the event is intense or tragic, the language remains reportorial and descriptive.

    2. **SUBJECTIVE (Editorial/Opinion):**
       - **First-Person/Direct Bias:** Uses the author's voice (e.g., "I believe", "We must", "It is my view").
       - **Unattributed Adjectives:** Uses "value-laden" words without quoting a source (e.g., "The disgusting behavior", "A magnificent achievement", "This shameful act").
       - **Speculation:** Makes claims about people's internal thoughts, feelings, or future intentions that cannot be verified.

    ### STEP-BY-STEP ANALYSIS:
    1. **Identify Entities:** List the people, organizations, or locations mentioned.
    2. **Scan for Attribution:** Identify if the claims are linked to a specific source or if they are "naked" claims by the author.
    3. **Tone Evaluation:** Check if adjectives are used to describe a person's character (Subjective) or a person's verifiable action (Objective).

    S: {text}

    Provide your output EXCLUSIVELY in a JSON object:
    {{
        "entities_identified": ["Entity1", "Entity2"],
        "final_category": "Objective/Subjective",
        "score": 0.0 to 1.0,
        "reasoning": "Explain the decision by identifying if the text relies on named entities and attributed statements (Objective) or author-driven opinion/adjectives (Subjective)."
    }}
    <|end|>
    <|assistant|>
    """

# 3. Hedging Detection
hedge_template = """
    You are an expert in linguistic analysis. Your task is to determine if the text 'S' contains hedging language (expressions of uncertainty, caution, or vagueness).

    Think step-by-step:
    1.  First, scan the text for words or phrases that soften a claim, express uncertainty, or show caution (e.g., "seems", "possible", "could suggest", "probably").
    2.  Next, analyze the impact of these phrases on the overall certainty of the text.
    3.  Finally, based on the presence and strength of these hedging phrases, determine if hedging is present and assign a score.

    Provide your output **exclusively in a single JSON object** with the following three keys:
    - "hedging_detected": A boolean value (true or false).
    - "hedging_score": A float between 0.0 (no hedging) and 1.0 (very high hedging).
    - "reasoning": A concise, step-by-step explanation of your analysis, identifying the specific hedge words found.

    S: {text}
    """

# 4. Clickbait Detection
clickbait_template = """
    You are an expert media analyst. Your task is to determine if the text 'S', typically a headline or title, uses clickbait techniques.

    Think step-by-step:
    1.  First, scan the text for common clickbait patterns: withholding information ("You Won't Believe..."), creating a curiosity gap ("The Secret Method..."), using exaggeration, or making sensational claims.
    2.  Next, analyze the impact of these phrases. Do they create a strong emotional response or an urge to click to find out more?
    3.  Finally, based on the presence and strength of these techniques, determine if the text is clickbait and assign a score.

    Provide your output **exclusively in a single JSON object** with the following three keys:
    - "clickbait_detected": A boolean value (true or false).
    - "clickbait_score": A float between 0.0 (not clickbait) and 1.0 (very strong clickbait).
    - "reasoning": A concise, step-by-step explanation of your analysis, identifying the specific clickbait techniques found.

    S: {text}
    """

# 5. Readability Analysis
readability_template = """
    You are an expert readability analyst, specifically in the context of fake news detection. Your task is to analyze the readability of the text 'S'.

    Think step-by-step:
    1.  First, analyze sentence structure, vocabulary complexity, and the use of jargon.
    2.  Next, consider that in the context of fake news, text that is overly simplistic and repetitive can be a red flag, just as text that is overly complex can be. A moderate, clear style is often associated with legitimate news.
    3.  Finally, assign a readability score and a classification based on your analysis.

    Provide your output **exclusively in a single JSON object** with the following three keys:
    - "readability_score": A float between 0.0 (very difficult to read) and 1.0 (very easy to read).
    - "complexity_level": A classification of the text's complexity ("Simple", "Moderate", or "Complex").
    - "reasoning": A concise, step-by-step explanation of your analysis.

    S: {text}
    """

# 6. Claim Extraction (Article)
claim_extraction_template = """Task: Summarize the core factual content of text S.
    Constraint 1: Output ONLY a single, concise paragraph.
    Constraint 2: Do not output reasoning, thoughts, or steps.
    Constraint 3: Ensure all Named Entities (People, Countries, Organizations) and specific dates/events are preserved.
    Output format: [Summary Text]
    
    S: {text}
"""

# 7. Web Content Summary (Evidence)
web_content_template = """Task: Extract the core factual content from the provided web search text S.
    
    Rules:
    1. Output ONLY the factual summary.
    2. Do NOT provide reasoning, steps, or introductory text.
    3. Retain all dates, numbers, and proper names (people, organizations, locations).
    4. Focus on the final resolution or the most recent status of the event.
    
    Output Format: [Factual Summary]
    
    S: {text}
"""

# 8. Claim Verification
claim_verify_template = """
    You are an expert fact-checker. Your task is to verify the truthfulness of the "Article Claim" using the "Web Claim" as the ground truth.

    Assume the "Web Claim" is accurate and verified information from reliable sources.
     **Analysis Steps:**
    1.  **Analyze Key Facts:** Identify the key factual points in both claims (actions, subjects, outcomes).
    2.  **Fact Comparison:** Compare the facts from the "Article Claim" against the "Web Claim".
    3.  **Verdict Formulation:** Based on the comparison, determine if the "Article Claim" is Supported, Refuted, Mixed or Irrelevant. A claim is "Mixed" if it contains some correct information but is misleading, out of date, or missing crucial context found in the ground truth.

    **Input Data:**
    ---------------------
    **Claim to Verify:** {article_claim}

    **Evidence Text (Ground Truth):** {web_claim}
    ---------------------
    **Output Format:**
    Provide a valid JSON object with the following keys:
    - "step_by_step_analysis": (String) Briefly compare key entities, actions, and numbers.
    - "evidence_quote": (String) A string or portion from the Evidence Text that supports your verdict. If irrelevant, return null.
    - "verdict": (String) Choose strictly from ["Supported", "Refuted", "Mixed", "Irrelevant"].
    - "confidence_score": (Float) 0.0 to 1.0. 

    **Verdict Definitions:**
    - "Supported": The evidence explicitly confirms the claim.
    - "Refuted": The evidence explicitly contradicts the claim.
    - "Mixed": The claim is partially correct but leaves out context or gets details wrong.
    - "Irrelevant": The evidence text does not contain enough information to verify this claim.
    """  

# 9. Master Judge
master_judge_template = """
You are the Final Veracity Arbiter. Your task is to determine if a news article is Real, Fake, or somewhere in between based on the following specific data signals.

### INPUT DATA SIGNALS:
1. **Sentiment Analysis:** {sentiment_scores}
2. **Subjectivity Type**: {subjectivity_type}
3. **Subjectivity Score**: {subjectivity_score}
4. **Subjectivity Reasoning**: {subjectivity_reasoning}
5. **Hedging Detected**: {hedging_detected} 
6. **Hedging Score** : {hedging_score}
7. ** Hedge Reasoning ** : {hedging_reasoning}
8. **Clickbait Detected**: {clickbait_detected} 
9. **Clickbait Score**: {clickbait_score}
10. **Clickbait Reasoning**: {clickbait_reasoning}
11. **Readability Score**: {readability_score} 
12. **Readability Complexity **: {readability_complexity}
13. ** Readability Reasoning** : {readability_reasoning}
14. **Source Authority (PageRank):** {pagerank_score}
15.**Fact-Check Verification Verdict**: {verdict} 
16.**Confidence **: {verify_conf}
17.**Stance Analysis label ** :{stance_label} 
18.** Stance Dissonace Score**: {dissonance_score}


### JUDGMENT CATEGORIES & LOGIC:

- **Real:** 
    * Fact check verdict : if the fact check verdict is "Supported" with a a high confidence score (e.g., >0.8), this is the 
    strongest signal of authenticity and should heavily influence the final label.
    * Tone: High Objectivity.
    * Language: Low or no Hedging; low Clickbait.
    * Readability: Balanced (not extremely simple/childish nor overly convoluted).
    * Authority: High PageRank combined with a "Supported" Fact-Check Verdict.
    *Stance : Article body supports the headline claim (stance_label : Favor) with low dissonance score.

- **Probable Real:** * Fact-Check Verdict is "Supported" but PageRank is low (e.g., local news).
    * OR PageRank is high but Fact-Check is "Mixed" (minor contextual errors in a reliable source).
    * Stance : Article body mostly supports the headline claim (stance_label : Favor) with moderate dissonance score.

- **Probable Fake:** * Fact-Check is "Mixed" or "Refuted" + High Clickbait.
    * Language: Highly Subjective and heavy use of Hedging (avoiding direct responsibility for claims).
    * Stance : Article body contradicts or is unrelated to the headline claim (stance_label : Against/Neutral) with high dissonance score.

- **Fake:** * Fact-Check Verdict is "Refuted" + Low Source Authority (PageRank).
    * Tone: Heavily Biased/Subjective and high Clickbait score.
    * Note: Even if PageRank is high, a "Refuted" Fact-Check Verdict for a specific claim can override source authority.
    * Stance : Article body contradicts the headline claim (stance_label : Against) with very high dissonance score.

### OUTPUT FORMAT (Strict JSON):
{{
    "veracity_label": "Real/Fake/Probable Real/Probable Fake",
    "reliability_score": 0.0 to 1.0 (How much should we trust this specific article?),
    "data_quality_weight": 0.0 to 1.0 (How clean/useful is this data point for training a model?),
    "justification": "Provide a concise summary (max 6 sentences) explaining how you balanced the 9 signals. Specifically reference how PageRank, Fact-Check, and linguistic style (Subjectivity/Readability) influenced the final label."
}}
"""