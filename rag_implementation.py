#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain RAG Implementation
Scrape coriander information from Wikipedia, build vector database, implement Q&A system
"""

import requests
from bs4 import BeautifulSoup
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def scrape_wikipedia_content():
    """Scrape coriander information from Wikipedia"""
    print("Scraping coriander information from Wikipedia...")

    url = "https://en.wikipedia.org/wiki/Coriander"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Create list to store extracted content
    extracted_content = []

    # Extract main content
    content_div = soup.find("div", {"id": "mw-content-text"})
    if content_div:
        # Extract all paragraphs
        paragraphs = content_div.find_all("p")
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and len(text) > 50:  # Only include substantial paragraphs
                extracted_content.append(text)

        # Extract section headers
        headers = content_div.find_all(["h2", "h3", "h4"])
        for header in headers:
            text = header.get_text(strip=True)
            if text:
                extracted_content.append(f"\n## {text}")

    # Write to TXT file
    with open("./coriander.txt", "w", encoding="utf-8") as f:
        for item in extracted_content:
            f.write(item + "\n")

    print("Content successfully extracted to coriander.txt file")
    return True


def load_and_split_documents():
    """Load and split documents"""
    print("Loading and splitting documents...")

    # Load document
    loader = TextLoader("./coriander.txt")
    documents = loader.load()

    # Document splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128,  # Adjust based on embedding model (e.g., text-embedding-ada-002 supports 8191)
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", "//"],  # Optimized separators
    )

    texts = text_splitter.create_documents(
        [documents[0].page_content], metadatas=[documents[0].metadata]
    )

    print(f"Document split into {len(texts)} chunks")
    return texts


def create_vector_database(documents):
    """Create vector database"""
    print("Creating vector database...")

    # Use local model path
    model_path = "./AI-ModelScope/m3e-base"

    # Ensure model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Configure embedding model parameters
    model_kwargs = {"device": "cpu"}  # Use CPU
    encode_kwargs = {"normalize_embeddings": True}  # Normalize embeddings

    # Create embedding model instance
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_path,  # Use local path instead of HuggingFace ID
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Generate vector representation for text retrieval",  # M3E model doesn't need specific instruction, but keeping it doesn't affect
    )

    # Load data to Chroma database
    db = Chroma.from_documents(documents, embedding)

    print("Vector database creation completed")
    return db


def create_qa_chain(db):
    """Create Q&A chain with source attribution"""
    print("Creating Q&A chain with source attribution...")

    llm = ChatOpenAI(
        model="deepseek-chat",  # Use DeepSeek official deepseek-chat model
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # DeepSeek official API key
        base_url="https://api.deepseek.com/v1",  # DeepSeek official API endpoint
    )

    # Configure retriever to return more documents for better attribution
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},  # Retrieve top 5 most relevant documents
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)

    print("Q&A chain with source attribution creation completed")
    return qa, retriever


def calculate_source_confidence(source_docs, question):
    """Calculate confidence metrics for source documents"""
    confidence_metrics = []

    for i, doc in enumerate(source_docs):
        # Simple relevance scoring based on word overlap
        question_words = set(question.lower().split())
        doc_words = set(doc.page_content.lower().split())

        # Calculate word overlap ratio
        overlap = len(question_words.intersection(doc_words))
        total_unique = len(question_words.union(doc_words))
        relevance_score = overlap / total_unique if total_unique > 0 else 0

        # Content length factor (longer content might be more informative)
        length_factor = min(len(doc.page_content) / 500, 1.0)  # Normalize to 0-1

        # Combined confidence score
        confidence = relevance_score * 0.7 + length_factor * 0.3

        confidence_metrics.append(
            {
                "index": i,
                "relevance_score": relevance_score,
                "length_factor": length_factor,
                "confidence": confidence,
                "content_preview": doc.page_content[:150] + "..."
                if len(doc.page_content) > 150
                else doc.page_content,
            }
        )

    return confidence_metrics


def interactive_qa_with_attribution(qa_chain, retriever):
    """Interactive Q&A with source attribution"""
    print("\n=== Coriander Knowledge Q&A System with Source Attribution ===")
    print("Type 'quit' to exit the system")
    print("Type 'sources' to see detailed source information")
    print("-" * 300)

    while True:
        question = input("\nPlease enter your question: ").strip()

        if question.lower() in ["quit", "exit"]:
            print("Thank you for using!")
            break

        if not question:
            continue

        try:
            # Get answer from QA chain
            result = qa_chain({"question": question})
            answer = result["answer"]

            # Get source documents
            source_docs = retriever.get_relevant_documents(question)

            # Calculate confidence metrics
            confidence_metrics = calculate_source_confidence(source_docs, question)

            # Display answer
            print(f"\n{'=' * 50}")
            print(f"Answer: {answer}")
            print(f"{'=' * 50}")

            # Display source attribution with confidence scores
            print(f"\n📚 Source Attribution:")
            print(f"Based on {len(source_docs)} relevant document chunks:")
            print("-" * 50)

            # Sort by confidence score (highest first)
            confidence_metrics.sort(key=lambda x: x["confidence"], reverse=True)

            for metric in confidence_metrics:
                i = metric["index"] + 1
                doc = source_docs[metric["index"]]

                # Truncate long content for display
                content = doc.page_content
                if len(content) > 200:
                    content = content[:200] + "..."

                print(f"\nSource {i} (Confidence: {metric['confidence']:.2f}):")
                print(f"Relevance Score: {metric['relevance_score']:.2f}")
                print(f"Content: {content}")
                print(f"Metadata: {doc.metadata}")
                print("-" * 30)

            # Ask if user wants to see full sources
            show_full = (
                input("\nWould you like to see the full source documents? (y/n): ")
                .strip()
                .lower()
            )
            if show_full == "y":
                print(f"\n📖 Full Source Documents:")
                for i, doc in enumerate(source_docs, 1):
                    print(f"\n{'=' * 60}")
                    print(f"Source Document {i}:")
                    print(f"{'=' * 60}")
                    print(doc.page_content)
                    print(f"\nMetadata: {doc.metadata}")
                    print(f"{'=' * 60}")

                # Additional analysis
                print(f"\n🔍 Source Analysis:")
                total_confidence = sum(
                    metric["confidence"] for metric in confidence_metrics
                )
                avg_confidence = (
                    total_confidence / len(confidence_metrics)
                    if confidence_metrics
                    else 0
                )
                print(f"Average Confidence: {avg_confidence:.2f}")
                print(
                    f"Highest Confidence Source: {confidence_metrics[0]['confidence']:.2f}"
                    if confidence_metrics
                    else "N/A"
                )
                print(
                    f"Lowest Confidence Source: {confidence_metrics[-1]['confidence']:.2f}"
                    if confidence_metrics
                    else "N/A"
                )

        except Exception as e:
            print(f"Error occurred while answering the question: {e}")


def main():
    """Main function"""
    try:
        # 1. Scrape Wikipedia content
        if not os.path.exists("./coriander.txt"):
            scrape_wikipedia_content()
        else:
            print("coriander.txt file already exists, skipping scraping step")

        # 2. Load and split documents
        documents = load_and_split_documents()

        # 3. Create vector database
        db = create_vector_database(documents)

        # 4. Create Q&A chain
        qa_chain, retriever = create_qa_chain(db)

        # 5. Interactive Q&A with attribution
        interactive_qa_with_attribution(qa_chain, retriever)

    except Exception as e:
        print(f"Program execution error: {e}")


if __name__ == "__main__":
    main()
