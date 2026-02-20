---
title: Dynamic Pricing RAG Model
emoji: 🚕
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: AI-powered dynamic pricing analysis with natural language queries
models:
  - mistralai/Mistral-7B-Instruct-v0.1
datasets:
  - dynamic_pricing
tags:
  - rag
  - nlp
  - pricing
  - langchain
  - gradio
  - retrieval-augmented-generation
---

# Dynamic Pricing RAG Model

A Retrieval-Augmented Generation (RAG) model for analyzing dynamic pricing data with natural language queries.

## 🚀 Features

- **Natural Language Queries**: Ask questions about pricing patterns in plain English
- **Intelligent Retrieval**: Retrieves relevant pricing records using TF-IDF similarity search
- **Data Analysis**: Generates statistics, averages, correlations, and insights
- **Dataset Overview**: Quick statistics about the pricing data
- **Easy to Use**: Simple web interface powered by Gradio

## 📊 Example Questions

- "What is the average price for premium vehicles in urban areas?"
- "How does customer loyalty status affect pricing?"
- "What time of day has the highest prices?"
- "How do ratings impact ride costs?"
- "What factors influence pricing?"

## 🛠 Technical Stack

- **LangChain**: RAG pipeline orchestration
- **Gradio**: Web interface
- **Scikit-learn**: TF-IDF vectorization and similarity search
- **Pandas**: Data processing
- **Hugging Face**: Model hosting and deployment

## 📁 Project Structure

```
Project1/
├── app.py                    # Gradio web interface
├── rag_model.py             # RAG implementation
├── requirements.txt         # Python dependencies
├── dynamic_pricing.csv      # Dataset (1000+ records)
├── README.md               # This file
└── .gitignore             # Git configuration
```

## 🔧 How It Works

1. **Data Loading**: Loads dynamic pricing CSV with 1000+ records
2. **Vector Creation**: Converts records into TF-IDF vectors
3. **Query Processing**: Transforms user query into vector
4. **Similarity Search**: Finds top 5 most similar pricing records
5. **Analysis**: Extracts and analyzes price data, ratings, correlations
6. **Answer Generation**: Provides intelligent insights based on retrieved data

## 💡 Use Cases

- Analyze pricing patterns and trends
- Understand factors affecting ride costs
- Explore customer loyalty and vehicle type impacts
- Generate data-driven pricing insights
- Educational demonstrations of RAG systems

## 🚀 Deployment

This space is automatically deployed on Hugging Face Spaces. Any updates to the repository will trigger redeployment.

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will be available at `http://localhost:7860`

## 📝 License

MIT License - Feel free to use and modify this project

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share use cases

## 📖 Learn More

- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Gradio Documentation](https://www.gradio.app)
- [LangChain Documentation](https://docs.langchain.com)
- [RAG Systems Guide](https://huggingface.co/docs/transformers/rag)

---

**Created with ❤️ for data analysis and AI-powered insights**
