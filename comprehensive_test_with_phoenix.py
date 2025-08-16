#!/usr/bin/env python3
"""
Comprehensive test of the Agentic RAG Pipeline with Phoenix Evaluation
Tests real questions and gets actual Phoenix scores
"""

import os
import pandas as pd
from agentic_rag_pipeline import AgenticRAGPipeline
from phoenix.evals import (
    RelevanceEvaluator,
    QAEvaluator, 
    HallucinationEvaluator,
    OpenAIModel,
    run_evals
)

def test_agentic_rag_with_phoenix():
    """Run comprehensive test with real questions and Phoenix evaluation"""
    
    print("🚀 Comprehensive Agentic RAG + Phoenix Test")
    print("=" * 60)
    
    # Test questions (mix of simple and complex)
    test_questions = [
        "How has Apple's total net sales changed over time?",
        "What are the major factors contributing to Apple's gross margin changes?", 
        "Compare Microsoft's revenue growth across different segments",
        "What significant changes occurred in Apple's operating expenses?",
        "How does Amazon's revenue compare to Apple's revenue in recent quarters?"
    ]
    
    # Initialize the agentic pipeline
    try:
        print("🔧 Initializing Agentic RAG Pipeline...")
        pipeline = AgenticRAGPipeline(
            lightrag_api_url="https://lightrag-h7h2g9atabdjg3af.canadacentral-01.azurewebsites.net",
            lightrag_api_key="YOUR_LIGHTRAG_API_KEY"
        )
        print("✅ Agentic pipeline initialized")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Initialize Phoenix evaluators with GPT-4o-mini (cost-effective)
    try:
        print("🔧 Initializing Phoenix evaluators...")
        model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.0)
        
        relevance_evaluator = RelevanceEvaluator(model)
        qa_evaluator = QAEvaluator(model)
        hallucination_evaluator = HallucinationEvaluator(model)
        print("✅ Phoenix evaluators ready")
    except Exception as e:
        print(f"❌ Failed to initialize evaluators: {e}")
        return
    
    # Run questions through pipeline and collect results
    results = []
    print(f"\n🔍 Processing {len(test_questions)} questions...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}/{len(test_questions)}: {question}")
        print("-" * 50)
        
        try:
            # Get response from agentic pipeline
            print("🔄 Getting response from agentic pipeline...")
            response = pipeline.query(question)
            
            # Extract the answer and context
            answer_text = response.get('answer', '')
            retrieval_context = response.get('context', 'No context available')
            
            print(f"✅ Response received ({len(answer_text)} chars)")
            print(f"📄 Answer preview: {answer_text[:100]}...")
            
            # Store for Phoenix evaluation
            results.append({
                'input': question,
                'output': answer_text,
                'reference': retrieval_context,
                'retrieval_mode': response.get('retrieval_mode', 'hybrid'),
                'query_complexity': response.get('query_classification', {}).get('complexity', 'unknown')
            })
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            # Add error case for evaluation
            results.append({
                'input': question,
                'output': f"Error: {str(e)}",
                'reference': "No context due to error",
                'retrieval_mode': 'error',
                'query_complexity': 'error'
            })
    
    if not results:
        print("❌ No results to evaluate")
        return
    
    # Create DataFrame for Phoenix evaluation
    print(f"\n📊 Preparing Phoenix evaluation DataFrame...")
    eval_df = pd.DataFrame(results)
    print(f"✅ Created DataFrame with {len(eval_df)} entries")
    
    # Run Phoenix evaluations
    try:
        print(f"\n🔍 Running Phoenix evaluations...")
        print("⏳ This may take a few minutes...")
        
        evaluation_results = run_evals(
            dataframe=eval_df,
            evaluators=[relevance_evaluator, qa_evaluator, hallucination_evaluator],
            provide_explanation=True,
            verbose=False,  # Reduce output noise
            concurrency=3   # Moderate concurrency to avoid rate limits
        )
        
        print("✅ Phoenix evaluations completed!")
        
        # Process and display results
        relevance_df, qa_df, hallucination_df = evaluation_results
        
        print(f"\n🎉 AGENTIC RAG + PHOENIX RESULTS")
        print("=" * 60)
        
        # Combine results for analysis
        for i in range(len(eval_df)):
            question = eval_df.iloc[i]['input']
            answer = eval_df.iloc[i]['output']
            mode = eval_df.iloc[i]['retrieval_mode']
            complexity = eval_df.iloc[i]['query_complexity']
            
            # Extract Phoenix scores
            relevance_score = relevance_df.iloc[i].get('label', 'N/A')
            qa_score = qa_df.iloc[i].get('label', 'N/A') 
            hallucination_score = hallucination_df.iloc[i].get('label', 'N/A')
            
            # Convert to numeric if possible
            rel_num = 1.0 if str(relevance_score).lower() in ['relevant', 'true', '1'] else 0.0 if str(relevance_score).lower() in ['unrelated', 'false', '0'] else relevance_score
            qa_num = 1.0 if str(qa_score).lower() in ['correct', 'true', '1'] else 0.0 if str(qa_score).lower() in ['incorrect', 'false', '0'] else qa_score
            hall_num = 1.0 if str(hallucination_score).lower() in ['hallucinated', 'true', '1'] else 0.0 if str(hallucination_score).lower() in ['factual', 'false', '0'] else hallucination_score
            
            # Calculate overall score
            try:
                overall = (float(rel_num) * 0.3 + float(qa_num) * 0.4 + (1 - float(hall_num)) * 0.3)
            except:
                overall = "N/A"
            
            print(f"\n🔍 Question {i+1}: {question[:50]}...")
            print(f"  🎛️  Mode: {mode} | Complexity: {complexity}")
            print(f"  📊 Relevance: {relevance_score} ({rel_num})")
            print(f"  📝 QA Quality: {qa_score} ({qa_num})")
            print(f"  🚫 Hallucination: {hallucination_score} ({hall_num})")
            print(f"  ⭐ Overall Score: {overall}")
            print(f"  💬 Answer: {answer[:80]}...")
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS")
        print("=" * 40)
        
        try:
            rel_scores = [1.0 if str(s).lower() in ['relevant', 'true', '1'] else 0.0 for s in relevance_df['label']]
            qa_scores = [1.0 if str(s).lower() in ['correct', 'true', '1'] else 0.0 for s in qa_df['label']]
            hall_scores = [1.0 if str(s).lower() in ['hallucinated', 'true', '1'] else 0.0 for s in hallucination_df['label']]
            
            avg_relevance = sum(rel_scores) / len(rel_scores)
            avg_qa = sum(qa_scores) / len(qa_scores) 
            avg_hallucination = sum(hall_scores) / len(hall_scores)
            avg_overall = (avg_relevance * 0.3 + avg_qa * 0.4 + (1 - avg_hallucination) * 0.3)
            
            print(f"📊 Average Relevance: {avg_relevance:.3f}")
            print(f"📝 Average QA Quality: {avg_qa:.3f}")
            print(f"🚫 Average Hallucination: {avg_hallucination:.3f}")
            print(f"⭐ Average Overall: {avg_overall:.3f}")
            
            # Performance assessment
            if avg_overall >= 0.9:
                grade = "🏆 Excellent"
            elif avg_overall >= 0.8:
                grade = "🥇 Very Good"
            elif avg_overall >= 0.7:
                grade = "🥈 Good"
            elif avg_overall >= 0.6:
                grade = "🥉 Fair"
            else:
                grade = "❌ Needs Improvement"
                
            print(f"🎯 Overall Grade: {grade}")
            
        except Exception as e:
            print(f"⚠️  Could not calculate summary statistics: {e}")
        
        # Show detailed Phoenix results
        print(f"\n📋 DETAILED PHOENIX RESULTS")
        print("=" * 40)
        print("\n📊 Relevance Evaluations:")
        print(relevance_df.to_string(index=False, max_colwidth=50))
        print("\n📝 QA Evaluations:")
        print(qa_df.to_string(index=False, max_colwidth=50))
        print("\n🚫 Hallucination Evaluations:")
        print(hallucination_df.to_string(index=False, max_colwidth=50))
        
    except Exception as e:
        print(f"❌ Phoenix evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Test Complete!")
    print("💡 This shows real performance of your hybrid-mode agentic pipeline!")

if __name__ == "__main__":
    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found. Please set it first:")
        print("export OPENAI_API_KEY=your_key_here")
        exit(1)
    
    test_agentic_rag_with_phoenix()
