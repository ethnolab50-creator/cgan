"""
Gradio interface for RAG model deployment on Hugging Face Spaces
"""

import os
import gradio as gr
from pathlib import Path
import sys

from rag_model import DynamicPricingRAG

# Initialize RAG model
rag = DynamicPricingRAG()

# Load data
data_path = Path(__file__).parent / "dynamic_pricing.csv"
if not data_path.exists():
    # Try alternative path
    data_path = "dynamic_pricing.csv"

print(f"Loading data from {data_path}")
documents = rag.load_data(str(data_path))

# Create vectorstore
print("Creating vectorstore...")
rag.create_vectorstore(documents, cache_dir="./vectorstore")

# Get HF token from environment
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    print("WARNING: HUGGINGFACEHUB_API_TOKEN not set. Some features may not work.")

# Setup QA chain
try:
    print("Setting up QA chain...")
    rag.setup_qa_chain(hf_token=hf_token)
except Exception as e:
    print(f"Error setting up QA chain: {e}")
    print("Will attempt to use a simpler retrieval approach")


def answer_question(question: str) -> str:
    """
    Answer questions about dynamic pricing
    
    Args:
        question: User's question
        
    Returns:
        Generated answer
    """
    if not question.strip():
        return "Please enter a question."
    
    try:
        result = rag.query(question)
        answer = result.get("answer", "No answer generated")
        return answer
    except Exception as e:
        return f"Error processing question: {str(e)}"


def get_dataset_info() -> str:
    """Get information about the dataset"""
    try:
        stats = rag.get_statistics()
        info = f"""
**Dataset Information:**

- **Total Records**: {stats['total_records']}
- **Average Price**: ${stats['avg_price']:.2f}
- **Price Range**: ${stats['min_price']:.2f} - ${stats['max_price']:.2f}
- **Vehicle Types**: {', '.join(stats['vehicle_types'])}
- **Locations**: {', '.join(stats['locations'])}
- **Loyalty Status**: {', '.join(stats['loyalty_statuses'])}
"""
        return info
    except Exception as e:
        return f"Error retrieving dataset info: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Dynamic Pricing RAG Model") as demo:
    gr.Markdown("# 🚕 Dynamic Pricing RAG Model")
    gr.Markdown("Ask questions about dynamic ride pricing patterns and get AI-generated insights!")
    
    with gr.Tab("Query"):
        gr.Markdown("### Ask questions about the pricing data")
        
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="E.g., What factors influence higher prices? How do ratings affect pricing?",
            lines=3
        )
        
        submit_btn = gr.Button("Get Answer", variant="primary")
        
        answer_output = gr.Textbox(
            label="Answer",
            lines=8,
            interactive=False
        )
        
        submit_btn.click(
            fn=answer_question,
            inputs=[question_input],
            outputs=answer_output
        )
    
    with gr.Tab("Dataset Info"):
        gr.Markdown("### Dataset Overview")
        info_output = gr.Markdown(get_dataset_info())
        
        refresh_btn = gr.Button("Refresh Info")
        refresh_btn.click(
            fn=get_dataset_info,
            outputs=info_output
        )
    
    with gr.Tab("About"):
        gr.Markdown("""
### About This RAG Model

This is a Retrieval-Augmented Generation (RAG) model for analyzing dynamic pricing data.

**How it works:**
1. **Data Loading**: Loads dynamic pricing CSV data (1000+ records)
2. **Embeddings**: Converts text into vector embeddings using sentence transformers
3. **Retrieval**: Finds relevant pricing records based on your question
4. **Generation**: Uses an LLM to generate human-readable answers

**Features:**
- Natural language queries about pricing patterns
- Retrieves relevant examples from the dataset
- AI-powered insights and analysis
- Real-time, no setup required

**Example Questions:**
- "What is the average price for premium vehicles?"
- "How does customer loyalty affect pricing?"
- "Which time of day has the highest prices?"
- "How do ratings impact the ride cost?"

**Technology Stack:**
- LangChain for RAG pipeline
- Hugging Face Transformers for embeddings and LLM
- FAISS for similarity search
- Gradio for web interface
""")


if __name__ == "__main__":
    demo.launch(share=True)
