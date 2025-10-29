import os
import httpx
import json
import asyncio
from bs4 import BeautifulSoup
import textstat
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

async def fetch_page(client, url):
    """Asynchronously fetches the content of a URL."""
    try:
        response = await client.get(url, follow_redirects=True, timeout=15.0)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.text
    except httpx.RequestError as exc:
        return f"An error occurred while requesting {exc.request.url!r}: {exc}"

def analyze_heading_structure(soup):
    """Analyzes heading tags (h1-h6) for hierarchy violations."""
    findings = []
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    h1_tags = soup.find_all('h1')
    if len(h1_tags) == 0:
        findings.append("❌ **Error:** No `<h1>` tag found.")
    elif len(h1_tags) > 1:
        findings.append(f"⚠️ **Warning:** Found {len(h1_tags)} `<h1>` tags. There should only be one.")

    if len(headings) > 1:
        last_level = int(headings[0].name[1])
        for i in range(1, len(headings)):
            current_level = int(headings[i].name[1])
            if current_level > last_level + 1:
                findings.append(f"❌ **Hierarchy Error:** Skipped from `<h{last_level}>` to `<h{current_level}>`. Text: \"{headings[i].get_text(strip=True)[:50]}...\"")
            last_level = current_level
            
    if not findings:
        findings.append("✅ **Success:** Heading structure is logical.")
        
    return findings

def audit_semantic_html(soup):
    """Performs a basic audit of semantic HTML tag usage."""
    findings = []
    
    for tag in soup.find_all(['strong', 'b']):
        if not tag.get_text(strip=True):
            findings.append("⚠️ **Warning:** Found an empty `<strong>` or `<b>` tag.")
            
    for list_tag in soup.find_all(['ul', 'ol']):
        invalid_children = [child.name for child in list_tag.children if child.name and child.name != 'li']
        if invalid_children:
            findings.append(f"❌ **Structure Error:** Found a `<{list_tag.name}>` tag with invalid direct children: {invalid_children}. Only `<li>` tags are allowed.")

    if not findings:
        findings.append("✅ **Success:** Basic semantic HTML looks good.")
        
    return findings

def analyze_readability(text):
    """Calculates readability score."""
    score = textstat.flesch_reading_ease(text)
    return {"flesch_reading_ease": score}

async def get_topical_gaps(client, title, text_content):
    """Uses Groq API to find topical gaps and generate Q&A pairs."""
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not found."}
    
    prompt = f"""
    An article's main topic is "{title}".
    1. Identify key sub-topics or common questions related to this topic that are missing from the article text provided below. Please note that these questions or topics should be related to amazon and the business being discussed. Basically, both the questions and answers should be found within the article, we're just cleaning it up and presenting it better.
    2. Based ONLY on the missing topics, generate 3-5 relevant question and answer pairs suitable for an FAQ section.
    3. VERY IMPORTANT: Format the output as a clean list, with each question starting with "Q:" and each answer starting with "A:". Do not add any other conversational text or introduction. This text will be customer facing, so please prepare it as a clear marketing copy. 

    ARTICLE TEXT TO ANALYZE:
    {text_content[:4000]}
    """
    payload = {"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    try:
        response = await client.post(GROQ_API_URL, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        
        qna_text = data['choices'][0]['message']['content']
        qna_pairs = []
        for line in qna_text.strip().split('\n'):
            if line.startswith('Q:'):
                qna_pairs.append({'question': line[2:].strip(), 'answer': ''})
            elif line.startswith('A:') and qna_pairs:
                qna_pairs[-1]['answer'] = line[2:].strip()
        
        return {"raw_text": qna_text, "structured_qna": qna_pairs}
    except Exception as e:
        return {"error": f"An error occurred with the LLM API: {e}", "raw_text": "", "structured_qna": []}

def audit_for_schema(soup):
    """Checks for existing Article and FAQPage schema."""
    found_schema = {'Article': False, 'FAQPage': False}
    script_tags = soup.find_all('script', type='application/ld+json')
    for tag in script_tags:
        try:
            data = json.loads(tag.string)
            graph = data.get('@graph', [data])
            for item in graph:
                schema_type = item.get('@type')
                if schema_type == 'Article':
                    found_schema['Article'] = True
                elif schema_type == 'FAQPage':
                    found_schema['FAQPage'] = True
        except (json.JSONDecodeError, AttributeError):
            continue
    return found_schema

def analyze_meta_tags(soup):
    """Analyzes title and meta description for length and content."""
    findings = {
        'title': {'text': '', 'length': 0, 'status': 'Missing'},
        'meta_description': {'text': '', 'length': 0, 'status': 'Missing'}
    }

    # 1. Analyze Title Tag
    title_tag = soup.find('title')
    if title_tag:
        text = title_tag.get_text(strip=True)
        length = len(text)
        status = 'Good'
        if length == 0:
            status = 'Empty'
        elif length > 65:
            status = '❌ Too long'
        
        findings['title'] = {'text': text, 'length': length, 'status': status}
    
    # 2. Analyze Meta Description
    meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
    if meta_desc_tag:
        text = meta_desc_tag.get('content', '').strip()
        length = len(text)
        status = 'Good'
        if length == 0:
            status = 'Empty'
        elif length > 160:
            status = '❌ Too long'
        
        findings['meta_description'] = {'text': text, 'length': length, 'status': status}

    return findings

async def get_title_recommendations(client, current_title, h1_text, text_content):
    """Uses the LLM to generate SEO-friendly title recommendations."""
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not found."}

    ### ▼▼▼ MODIFIED PROMPT ▼▼▼ ###
    prompt = f"""
    You are an expert SEO copywriter.
    The current title tag is: "{current_title}"
    The main H1 heading is: "{h1_text}"

    Your task is to generate 3 improved, SEO-friendly title tags based on the article's content.
    
    GUIDELINES:
    1. CRITICAL: All 3 titles MUST be 60 characters or less. This is a strict technical limit. Do not go over.
    2. They must be compelling for human readers.
    3. They must be machine-readable, incorporating key entities and semantic concepts from the article text.
    4. Do not just add keywords. Capture the user's intent.

    List only the 3 new title suggestions, each on a new line.
    Verify each one is 60 characters or less. Do not add any other text.

    ARTICLE TEXT SNIPPET FOR CONTEXT:
    {text_content[:2000]}
    """
    ### ▲▲▲ END OF MODIFIED PROMPT ▲▲▲ ###
    
    payload = {
        "model": "llama-3.1-8b-instant", # Using the correct model
        "messages": [{"role": "user", "content": prompt}], 
        "temperature": 0.7
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    try:
        response = await client.post(GROQ_API_URL, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        suggestions = response.json()['choices'][0]['message']['content']
        return {"suggestions": suggestions, "error": None}
    except Exception as e:
        return {"error": f"An error occurred with the LLM API: {e}", "suggestions": ""}

def generate_article_schema(soup, url):
    """Generates basic Article JSON-LD schema."""
    title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "No H1 Title Found"
    schema = {"@context": "https://schema.org", "@type": "Article", "headline": title, "mainEntityOfPage": {"@type": "WebPage", "@id": url}}
    return json.dumps(schema, indent=4)

def generate_faq_schema(qna_pairs):
    """Generates FAQPage JSON-LD schema from structured Q&A pairs."""
    if not qna_pairs:
        return None
    main_entity = []
    for pair in qna_pairs:
        if pair['question'] and pair['answer']:
            main_entity.append({"@type": "Question", "name": pair['question'], "acceptedAnswer": {"@type": "Answer", "text": pair['answer']}})
    if not main_entity:
        return None
    schema = {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": main_entity}
    return json.dumps(schema, indent=4)

async def get_content_structure_recommendations(client, text_content):
    """Uses the LLM to suggest headings for long-form text."""
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not found."}

    cleaned_text = ' '.join(text_content.split())
    
    ### ▼▼▼ MODIFIED PROMPT ▼▼▼ ###
    prompt = f"""
    Analyze the following article text, which is long and lacks sufficient headings. Your task is to improve its scannability and structure by suggesting headings.

    1. Read through the text and identify **major** logical breaks where a new, **substantial** sub-topic begins.
    2. Suggest a concise and descriptive heading for these breaks. Use H2s for major topics and H3s for sub-topics.
    3. **CRITICAL RULE:** Do not suggest new headings that are too close together. Ensure there are at least 1-2 paragraphs of substantial content *between* each suggested heading to avoid over-structuring the article.
    4. Present your recommendations as a outline using text headers like "H2" or "h3". Do not rewrite the original text. Please include the first sentence of the paragraph that is after the heading to idenyify location. 

    Example Output:
    ## New Suggested H2
    ### New Suggested H3
    ## Another New Suggested H2

    ARTICLE TEXT TO ANALYZE:
    {cleaned_text[:6000]}
    """
    ### ▲▲▲ END OF MODIFIED PROMPT ▲▲▲ ###
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
    }
    headers = { "Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json" }

    try:
        response = await client.post(GROQ_API_URL, json=payload, headers=headers, timeout=45.0)
        response.raise_for_status()
        data = response.json()
        return {"heading_suggestions": data['choices'][0]['message']['content']}
    except Exception as e:
        return {"error": f"An error occurred with the LLM API: {e}", "heading_suggestions": ""}

async def analyze_url(url):
    """Main analysis orchestrator for a single URL."""
    async with httpx.AsyncClient() as client:
        html_content = await fetch_page(client, url)

        if html_content.startswith("An error occurred"):
            return {"url": url, "error": html_content}

        soup = BeautifulSoup(html_content, 'html.parser')
        main_content = soup.find('main') or soup.find('article') or soup.body
        text_content = main_content.get_text()
        
        title = soup.find('title').string if soup.find('title') else "No Title Found"
        h1_tag = soup.find('h1')
        h1_text = h1_tag.get_text(strip=True) if h1_tag else title

        # Run all async LLM calls concurrently
        (topical_findings, 
         structure_recommendations,
         title_recommendations) = await asyncio.gather(
            get_topical_gaps(client, title, text_content),
            get_content_structure_recommendations(client, text_content),
            get_title_recommendations(client, title, h1_text, text_content)
        )

        # Run all synchronous analyses
        meta_tag_findings = analyze_meta_tags(soup) # <-- NEW
        heading_findings = analyze_heading_structure(soup)
        semantic_findings = audit_semantic_html(soup)
        readability_findings = analyze_readability(text_content)
        existing_schema = audit_for_schema(soup)

        # Generate recommendations
        recommendations = {"article_schema": None, "faq_schema": None}
        if not existing_schema['Article']:
            recommendations['article_schema'] = generate_article_schema(soup, url)
        
        if not existing_schema['FAQPage'] and topical_findings.get("structured_qna"):
            recommendations['faq_schema'] = generate_faq_schema(topical_findings["structured_qna"])

        # Compile all results
        return {
            "url": url,
            "title": title,
            "meta_analysis": { # <-- NEW SECTION
                "tags": meta_tag_findings,
                "llm_suggestions": title_recommendations
            },
            "structural_integrity": {"headings": heading_findings, "semantics": semantic_findings},
            "readability": readability_findings,
            "topical_gaps": {"raw_text": topical_findings.get("raw_text")},
            "existing_schema": existing_schema,
            "recommendations": recommendations,
            "content_structure": structure_recommendations
        }
    ##llama-3.1-8b-instant