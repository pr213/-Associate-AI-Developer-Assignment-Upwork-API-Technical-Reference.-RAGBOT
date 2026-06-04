"""
debug_retrieval.py — Debug script to test retrieval for specific test questions
============================================================================
This script tests the retrieval for the failing test cases to understand
what chunks are being returned and why the answers might not be found.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from rag import get_retriever

# Test questions from the assignment
test_questions = [
    "What happens after a subscription is created?",
    "What three fields does a subscription event payload contain?",
    "What entity types can I subscribe to?",
    "What actions can I subscribe to for job postings?",
    "What permissions do I need to query the list of countries?",
    "What fields are returned in the languages query?",
    "What is the reasons query used for?",
]

print("=" * 80)
print("DEBUGGING RETRIEVAL FOR TEST QUESTIONS")
print("=" * 80)

# Load retriever
print("\nLoading retriever...")
retriever = get_retriever()
print("Retriever loaded successfully\n")

# Test each question
for i, question in enumerate(test_questions, 1):
    print(f"\n{'=' * 80}")
    print(f"QUESTION {i}: {question}")
    print(f"{'=' * 80}")
    
    # Retrieve chunks
    docs = retriever.invoke(question)
    
    print(f"\nRetrieved {len(docs)} chunks:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"--- Chunk {j} (Page {doc.metadata.get('page', '?')}) ---")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Full content length: {len(doc.page_content)} characters")
        print()
    
    # Check if any chunk contains keywords from the expected answers
    print(f"Keyword search in retrieved chunks:")
    
    # Question-specific keywords
    if "subscription" in question.lower():
        keywords = ["REVIEW", "approved", "entity", "action", "id", "JP", "OFFER", "MILESTONE"]
    elif "countries" in question.lower():
        keywords = ["Common Functionality", "Read And Write Access"]
    elif "languages" in question.lower():
        keywords = ["iso639Code", "active", "englishName"]
    elif "reasons" in question.lower():
        keywords = ["declining", "invitations", "ending contracts", "withdrawing offers"]
    else:
        keywords = []
    
    for keyword in keywords:
        found = False
        for doc in docs:
            if keyword.lower() in doc.page_content.lower():
                found = True
                break
        status = "✓ FOUND" if found else "✗ NOT FOUND"
        print(f"  {status}: '{keyword}'")

print("\n" + "=" * 80)
print("DEBUGGING COMPLETE")
print("=" * 80)
