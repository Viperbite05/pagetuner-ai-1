import streamlit as st
import asyncio
import pandas as pd
import io
from analyzer import analyze_url # Our core logic is imported

# --- Page Configuration ---
# Set the page to be wide
st.set_page_config(
    page_title="PageTuner AI",
    layout="wide"
)

# --- CSV Generation Function ---
# We copy this helper function directly from our Flask app
def flatten_results_for_csv(results):
    """Converts the nested result dictionaries into a flat list for pandas."""
    flat_data = []
    for result in results:
        if result.get('error'):
            row = {'URL': result.get('url'), 'Error': result.get('error')}
            flat_data.append(row)
            continue
            
        # Get the new meta analysis data
        meta_analysis = result.get('meta_analysis', {})
        tags = meta_analysis.get('tags', {})
        title_info = tags.get('title', {})
        meta_info = tags.get('meta_description', {})
        llm_sugs = meta_analysis.get('llm_suggestions', {})

        row = {
            'URL': result.get('url'),
            'Title': result.get('title'),
            'Title Text': title_info.get('text'), # <-- NEW
            'Title Length': title_info.get('length'), # <-- NEW
            'Title Status': title_info.get('status'), # <-- NEW
            'Meta Description Text': meta_info.get('text'), # <-- NEW
            'Meta Description Length': meta_info.get('length'), # <-- NEW
            'Meta Description Status': meta_info.get('status'), # <-- NEW
            'LLM Title Suggestions': llm_sugs.get('suggestions', ''), # <-- NEW
            'Readability (Flesch Ease)': result.get('readability', {}).get('flesch_reading_ease'),
            'Structural Integrity - Headings': "\n".join(result.get('structural_integrity', {}).get('headings', [])),
            'Structural Integrity - Semantics': "\n".join(result.get('structural_integrity', {}).get('semantics', [])),
            'Existing Article Schema?': result.get('existing_schema', {}).get('Article'),
            'Existing FAQ Schema?': result.get('existing_schema', {}).get('FAQPage'),
            'Generated Article Schema': result.get('recommendations', {}).get('article_schema'),
            'Generated FAQ Schema': result.get('recommendations', {}).get('faq_schema'),
            'Content Structure Suggestions': result.get('content_structure', {}).get('heading_suggestions'),
            'Identified Content Gaps': result.get('topical_gaps', {}).get('raw_text')
        }
        flat_data.append(row)
    return flat_data

# --- Async Runner ---
# Streamlit runs in a way that needs this helper to call our async code
async def run_analysis(urls):
    return await asyncio.gather(*(analyze_url(url) for url in urls))

# --- PageTuner AI Dashboard ---
st.title("🤖 PageTuner AI")
st.caption("On-page optimization for technical and semantic structure.")

# --- URL Input ---
urls_text = st.text_area("Enter URLs (one per line, max 500)", height=200, placeholder="https://www.example.com/article1\nhttps://www.example.com/article2")

# --- Run Analysis Button ---
if st.button("Analyze URLs", type="primary"):
    urls = [url.strip() for url in urls_text.splitlines() if url.strip()]
    
    if not urls:
        st.error("Please enter at least one URL.")
    elif len(urls) > 500:
        st.error("Maximum of 500 URLs allowed.")
    else:
        # Run the analysis with a spinner
        with st.spinner(f"Analyzing {len(urls)} page(s)... This may take a moment."):
            try:
                # Run our async functions
                analysis_results = asyncio.run(run_analysis(urls))
                # Store results in Streamlit's session state
                st.session_state['results'] = analysis_results
                st.success("Analysis complete!")
            except Exception as e:
                st.exception(f"An unexpected error occurred: {e}")

# --- Display Results ---
# Check if results exist in the session state
if 'results' in st.session_state:
    results = st.session_state['results']
    
    st.header("Analysis Report", divider="blue")

    # --- Download Button ---
    try:
        flat_data = flatten_results_for_csv(results)
        df = pd.DataFrame(flat_data)
        # Create an in-memory CSV
        csv_output = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download CSV Report",
            data=csv_output,
            file_name="pagetuner_report.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Could not prepare CSV for download: {e}")
        
    # --- Individual Page Reports ---
    for result in results:
        if result.get('error'):
            with st.expander(f"❌ Error Analyzing: {result.get('url')}", expanded=True):
                st.error(result.get('error'))
            continue

        # Use an expander for each URL
        with st.expander(f"✅ {result.get('title')}"):
            st.link_button("Open URL in New Tab", result.get('url'))
            
            # Create two columns for a cleaner layout
            col1, col2 = st.columns(2)

            with col1:
                # --- NEW SECTION: Title & Meta Analysis ---
                st.subheader("Title & Meta Analysis")
                meta_data = result.get('meta_analysis', {})
                tags_data = meta_data.get('tags', {})
                llm_data = meta_data.get('llm_suggestions', {})
                
                # Title
                title_info = tags_data.get('title', {})
                st.metric(f"Title Length ({title_info.get('status')})", f"{title_info.get('length')} / 65")
                st.markdown("**Current Title**")
                st.code(title_info.get('text'), language=None)
                
                # Meta Description
                meta_info = tags_data.get('meta_description', {})
                st.metric(f"Meta Desc. Length ({meta_info.get('status')})", f"{meta_info.get('length')} / 160")
                st.markdown("**Current Meta Desc**")
                st.code(meta_info.get('text'), language=None)
                
                # LLM Suggestions
                st.markdown("**LLM Title Recommendations:**")
                if llm_data and not llm_data.get('error'):
                    st.code(llm_data.get('suggestions'), language=None)
                else:
                    st.info("No title suggestions were generated.")
                # --- END OF NEW SECTION ---
                # Section 1: Recommendations
                st.subheader("Recommendations & Generated Assets")
                st.markdown("**Article Schema:**")
                if result.get('existing_schema', {}).get('Article'):
                    st.success("Article Schema already detected on page.")
                elif result.get('recommendations', {}).get('article_schema'):
                    st.warning("Article Schema missing. Generated schema below:")
                    st.code(result.get('recommendations').get('article_schema'), language="json")
                
                st.markdown("**FAQ Schema:**")
                if result.get('existing_schema', {}).get('FAQPage'):
                    st.success("FAQ Schema already detected on page.")
                elif result.get('recommendations', {}).get('faq_schema'):
                    st.warning("FAQ Schema missing. Generated schema below:")
                    st.code(result.get('recommendations').get('faq_schema'), language="json")
                
                # Section 2: Content Structure
                st.subheader("Content Structure Recommendations")
                if result.get('content_structure') and not result.get('content_structure').get('error'):
                    # Use st.markdown to render the H2/H3s
                    st.markdown(result.get('content_structure').get('heading_suggestions'))
                else:
                    st.info("Could not generate heading suggestions.")

            with col2:
                # Section 3: Structural Integrity
                st.subheader("Structural Integrity")
                for finding in result.get('structural_integrity', {}).get('headings', []):
                    st.markdown(finding) # Use markdown to render emoji/bold
                for finding in result.get('structural_integrity', {}).get('semantics', []):
                    st.markdown(finding)
                
                # Section 4: Readability
                st.subheader("Readability")
                st.metric("Flesch Reading Ease", result.get('readability', {}).get('flesch_reading_ease'))
                
                # Section 5: Topical Gaps
                st.subheader("Identified Content Gaps (LLM Output)")
                st.text(result.get('topical_gaps', {}).get('raw_text'))