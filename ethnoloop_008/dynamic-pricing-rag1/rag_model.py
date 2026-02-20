"""
RAG Model for Dynamic Pricing Analysis
Retrieves relevant pricing data and generates insights using an LLM
"""

import pandas as pd
import os
from pathlib import Path
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class DynamicPricingRAG:
    """RAG model for dynamic pricing data analysis"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        """
        Initialize RAG model
        
        Args:
            model_name: HuggingFace model name for LLM
        """
        self.model_name = model_name
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = None
        self.qa_chain = None
        self.llm = None
        self.data_df = None
        
    def load_data(self, csv_path: str):
        """Load CSV data and convert to documents"""
        print(f"Loading data from {csv_path}...")
        self.data_df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.data_df)} records")
        
        # Convert dataframe rows to documents
        documents = []
        for idx, row in self.data_df.iterrows():
            doc_text = self._format_row_as_text(row)
            doc = Document(
                page_content=doc_text,
                metadata={"row_id": idx}
            )
            documents.append(doc)
        
        return documents
    
    def _format_row_as_text(self, row: pd.Series) -> str:
        """Format a data row as readable text"""
        text = f"""
Dynamic Pricing Record:
- Number of Riders: {row['Number_of_Riders']}
- Number of Drivers: {row['Number_of_Drivers']}
- Location: {row['Location_Category']}
- Customer Loyalty: {row['Customer_Loyalty_Status']}
- Past Rides: {row['Number_of_Past_Rides']}
- Rating: {row['Average_Ratings']}
- Time: {row['Time_of_Booking']}
- Vehicle: {row['Vehicle_Type']}
- Duration: {row['Expected_Ride_Duration']} minutes
- Historical Cost: ${row['Historical_Cost_of_Ride']:.2f}
"""
        return text.strip()
    
    def create_vectorstore(self, documents: List[Document], cache_dir: str = "./vectorstore"):
        """Create vector store using TF-IDF for similarity search"""
        print("Creating vectorstore using TF-IDF...")
        
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store documents
        self.documents = documents
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        # Create TF-IDF vectorizer
        print(f"Vectorizing {len(texts)} documents...")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"✓ Vectorstore created with {len(texts)} documents")
        
        return self
    
    def _retrieve_similar_docs(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve similar documents using TF-IDF similarity"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("Vectorstore not initialized")
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top k indices with better filtering
        top_indices = np.argsort(similarities)[-k*2:][::-1]  # Get more candidates
        
        # Filter by similarity threshold and remove very similar duplicates
        filtered_indices = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Lower threshold to get more diverse results
                filtered_indices.append(idx)
            if len(filtered_indices) >= k:
                break
        
        if not filtered_indices:
            # Fallback to top k even if low similarity
            filtered_indices = top_indices[:k]
        
        # Return documents
        return [self.documents[i] for i in filtered_indices]
    
    def _extract_keyword_filters(self, query: str) -> dict:
        """Extract keywords and potential filters from query"""
        query_lower = query.lower()
        filters = {}
        
        # Check for vehicle types
        if 'premium' in query_lower:
            filters['vehicle'] = 'Premium'
        elif 'economy' in query_lower:
            filters['vehicle'] = 'Economy'
        
        # Check for locations
        if 'urban' in query_lower:
            filters['location'] = 'Urban'
        elif 'suburban' in query_lower:
            filters['location'] = 'Suburban'
        elif 'rural' in query_lower:
            filters['location'] = 'Rural'
        
        # Check for loyalty status
        if 'gold' in query_lower:
            filters['loyalty'] = 'Gold'
        elif 'silver' in query_lower:
            filters['loyalty'] = 'Silver'
        elif 'regular' in query_lower:
            filters['loyalty'] = 'Regular'
        
        return filters
    
    def _generate_answer(self, question: str, docs: List[Document]) -> str:
        """Generate intelligent answer based on retrieved documents"""
        if not docs:
            return "No matching data found for your query."
        
        # Extract relevant info from documents
        prices = []
        locations = []
        vehicles = []
        ratings = []
        
        for doc in docs:
            content = doc.page_content
            # Extract price
            if 'Historical Cost:' in content:
                try:
                    price_str = content.split('Historical Cost: $')[1].split('\n')[0]
                    prices.append(float(price_str))
                except:
                    pass
            # Extract location  
            if 'Location:' in content:
                try:
                    location = content.split('Location: ')[1].split('\n')[0]
                    locations.append(location)
                except:
                    pass
            # Extract vehicle
            if 'Vehicle:' in content:
                try:
                    vehicle = content.split('Vehicle: ')[1].split('\n')[0]
                    vehicles.append(vehicle)
                except:
                    pass
            # Extract rating
            if 'Rating:' in content:
                try:
                    rating_str = content.split('Rating: ')[1].split('\n')[0]
                    ratings.append(float(rating_str))
                except:
                    pass
        
        # Generate smart answer based on question
        question_lower = question.lower()
        answer = ""
        
        if 'average' in question_lower or 'mean' in question_lower:
            if prices:
                avg_price = np.mean(prices)
                answer = f"Based on the retrieved data, the average price is ${avg_price:.2f}.\n"
                if vehicles:
                    answer += f"Vehicles included: {', '.join(set(vehicles))}.\n"
                if locations:
                    answer += f"Locations: {', '.join(set(locations))}.\n"
        
        elif 'highest' in question_lower or 'maximum' in question_lower or 'max' in question_lower:
            if prices:
                max_price = max(prices)
                min_price = min(prices)
                answer = f"Maximum price: ${max_price:.2f}, Minimum price: ${min_price:.2f}.\n"
        
        elif 'rating' in question_lower or 'affect' in question_lower or 'influence' in question_lower:
            if ratings:
                avg_rating = np.mean(ratings)
                answer = f"Average rating in retrieved records: {avg_rating:.2f}/5.0\n"
            if prices and ratings:
                correlation = np.corrcoef(ratings, prices)[0, 1]
                answer += f"Correlation between rating and price: {correlation:.3f}\n"
            if vehicles:
                answer += f"Vehicle types analyzed: {', '.join(set(vehicles))}.\n"
        
        elif 'factor' in question_lower or 'what' in question_lower:
            answer = "Key factors influencing pricing based on the data:\n"
            if vehicles:
                answer += f"- Vehicle Type: {', '.join(set(vehicles))}\n"
            if locations:
                answer += f"- Location: {', '.join(set(locations))}\n"
            if prices:
                answer += f"- Price Range: ${min(prices):.2f} - ${max(prices):.2f}\n"
            if ratings:
                answer += f"- Customer Ratings: Average {np.mean(ratings):.2f}\n"
        
        elif 'premium' in question_lower:
            premium_prices = prices if any('Premium' in d.page_content for d in docs) else []
            if premium_prices:
                answer = f"Premium vehicles average: ${np.mean(premium_prices):.2f}\n"
            else:
                answer = "Premium vehicle pricing information:\n"
            if any('Premium' in d.page_content for d in docs):
                answer += "Found premium vehicle records in the dataset.\n"
        
        elif 'economy' in question_lower:
            economy_prices = prices if any('Economy' in d.page_content for d in docs) else []
            if economy_prices:
                answer = f"Economy vehicles average: ${np.mean(economy_prices):.2f}\n"
            else:
                answer = "Economy vehicle pricing information:\n"
            if any('Economy' in d.page_content for d in docs):
                answer += "Found economy vehicle records in the dataset.\n"
        
        else:
            # Default answer for unknown queries
            answer = f"Retrieved {len(docs)} relevant records.\n"
            if prices:
                answer += f"Price range: ${min(prices):.2f} - ${max(prices):.2f}, Average: ${np.mean(prices):.2f}\n"
            if vehicles:
                answer += f"Vehicles: {', '.join(set(vehicles))}\n"
            if locations:
                answer += f"Locations: {', '.join(set(locations))}\n"
        
        return answer
    
    def setup_qa_chain(self, hf_token: str = None):
        """Setup QA chain with LLM"""
        if self.documents is None:
            raise ValueError("Documents not initialized. Call create_vectorstore first.")
        
        print(f"Initializing LLM: {self.model_name}...")
        
        if hf_token is None:
            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        try:
            from langchain_community.llms import HuggingFaceHub
            
            llm = HuggingFaceHub(
                repo_id=self.model_name,
                model_kwargs={"temperature": 0.7, "max_length": 500},
                huggingfacehub_api_token=hf_token
            )
            
            # Store LLM
            self.llm = llm
            self.qa_chain = "ready"
            
            print("✓ QA chain ready")
            return self.qa_chain
        except Exception as e:
            print(f"Warning: Could not setup LLM: {e}")
            print("Using retrieval-only mode...")
            self.qa_chain = "retrieval_only"
            return self.qa_chain
    
    def query(self, question: str) -> dict:
        """
        Query the RAG model
        
        Args:
            question: User question about pricing
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.documents is None:
            raise ValueError("Documents not initialized. Call create_vectorstore first.")
        
        print(f"\nQuery: {question}")
        
        try:
            # Retrieve relevant documents
            docs = self._retrieve_similar_docs(question, k=5)
            
            # Generate intelligent answer
            answer = self._generate_answer(question, docs)
            
            # If LLM is available and capable, use it for enhancement
            if hasattr(self, 'llm') and self.llm:
                try:
                    # Use LLM to enhance answer
                    context = "\n---\n".join([doc.page_content for doc in docs])
                    prompt = f"Based on the following pricing data:\n\n{context}\n\nQuestion: {question}\n\nProvide a detailed answer:"
                    llm_answer = self.llm(prompt)
                    answer = llm_answer
                except Exception as e:
                    print(f"LLM enhancement skipped: {e}")
                    # Keep the generated answer
                    pass
            
            return {
                "question": question,
                "answer": answer,
                "sources": docs
            }
        except Exception as e:
            print(f"Query error: {e}")
            import traceback
            traceback.print_exc()
            
            # Final fallback
            docs = self._retrieve_similar_docs(question, k=5)
            answer = self._generate_answer(question, docs)
            
            return {
                "question": question,
                "answer": answer,
                "sources": docs
            }
    
    def get_statistics(self) -> dict:
        """Get basic statistics about the dataset"""
        if self.data_df is None:
            raise ValueError("Data not loaded")
        
        return {
            "total_records": len(self.data_df),
            "avg_price": float(self.data_df['Historical_Cost_of_Ride'].mean()),
            "max_price": float(self.data_df['Historical_Cost_of_Ride'].max()),
            "min_price": float(self.data_df['Historical_Cost_of_Ride'].min()),
            "vehicle_types": self.data_df['Vehicle_Type'].unique().tolist(),
            "locations": self.data_df['Location_Category'].unique().tolist(),
            "loyalty_statuses": self.data_df['Customer_Loyalty_Status'].unique().tolist(),
        }


def main():
    """Example usage"""
    # Initialize RAG
    rag = DynamicPricingRAG()
    
    # Load data
    csv_path = "./dynamic_pricing.csv"
    documents = rag.load_data(csv_path)
    
    # Create vectorstore
    rag.create_vectorstore(documents)
    
    # Setup QA chain
    try:
        rag.setup_qa_chain()
    except Exception as e:
        print(f"Warning: Could not setup LLM: {e}")
        print("Continuing with retrieval-only mode...")
    
    # Example queries
    queries = [
        "What is the average price for premium vehicles in urban areas?",
        "How does rating affect pricing?",
        "What factors influence high-priced rides?",
    ]
    
    for query in queries:
        result = rag.query(query)
        print(f"\n{'='*80}")
        print(f"Q: {result['question']}")
        print(f"A: {result['answer'][:500]}...")  # Show first 500 chars
        print(f"Sources: {len(result['sources'])} documents retrieved")
    
    # Print statistics
    stats = rag.get_statistics()
    print(f"\n{'='*80}")
    print("Dataset Statistics:")
    print(f"Total Records: {stats['total_records']}")
    print(f"Average Price: ${stats['avg_price']:.2f}")
    print(f"Price Range: ${stats['min_price']:.2f} - ${stats['max_price']:.2f}")


if __name__ == "__main__":
    main()
