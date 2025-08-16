#!/usr/bin/env python3
"""
Phoenix Evaluator Script
Loads saved evaluation results and calculates Phoenix metrics
"""

import pandas as pd
import logging
from typing import Dict, Any
import asyncio
import nest_asyncio
from datetime import datetime
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Phoenix imports
import phoenix as px
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    OpenAIModel,
    run_evals,
    llm_classify,
    QA_PROMPT_TEMPLATE,
    QA_PROMPT_RAILS_MAP,
    HALLUCINATION_PROMPT_TEMPLATE,
    HALLUCINATION_PROMPT_RAILS_MAP,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply nest_asyncio for async operations
nest_asyncio.apply()

class PhoenixEvaluator:
    """Phoenix evaluator for saved results"""
    
    def __init__(self):
        self.results_df = None
        self.evaluators = {}
        
    def load_results(self, csv_path: str = "evaluation_results_detailed.csv") -> pd.DataFrame:
        """Load saved evaluation results"""
        try:
            self.results_df = pd.read_csv(csv_path)
            logger.info(f"✅ Loaded {len(self.results_df)} evaluation results")
            return self.results_df
        except Exception as e:
            logger.error(f"❌ Failed to load results: {e}")
            raise
    
    def setup_evaluators(self):
        """Initialize Phoenix evaluators"""
        try:
            # Check for OpenAI API key
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")
            
            # Use GPT-4 for evaluations
            eval_model = OpenAIModel(
                model="gpt-4",
                temperature=0.0,
                api_key=openai_api_key
            )
            
            # Initialize evaluators
            self.evaluators = {
                "hallucination": HallucinationEvaluator(eval_model),
                "qa": QAEvaluator(eval_model)
            }
            
            logger.info("✅ Phoenix evaluators initialized")
            
        except Exception as e:
            logger.error(f"❌ Evaluator initialization failed: {e}")
            raise
    
    def calculate_rouge_metrics(self) -> Dict[str, float]:
        """Calculate ROUGE-1 metrics (Precision, Recall, F1)"""
        try:
            from rouge import Rouge
            
            rouge = Rouge()
            
            # Prepare data for ROUGE calculation
            hypotheses = self.results_df['generated_answer'].fillna('').tolist()
            references = self.results_df['ground_truth'].fillna('').tolist()
            
            # Calculate ROUGE-1 scores
            scores = rouge.get_scores(hypotheses, references, avg=True)
            rouge_1 = scores['rouge-1']
            
            metrics = {
                'precision': rouge_1['p'],
                'recall': rouge_1['r'],
                'f1': rouge_1['f']
            }
            
            logger.info(f"✅ ROUGE-1 Metrics: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ ROUGE calculation failed: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def calculate_phoenix_metrics(self) -> Dict[str, float]:
        """Calculate Phoenix metrics (Hallucination, Correctness)"""
        try:
            # Prepare data for Phoenix evaluators
            eval_data = []
            
            for _, row in self.results_df.iterrows():
                eval_data.append({
                    "question": row['question'],
                    "reference": row['ground_truth'],
                    "input": row['question'],
                    "output": row['generated_answer']
                })
            
            eval_df = pd.DataFrame(eval_data)
            
            # Run hallucination evaluation with conservative rate limiting
            logger.info("🔍 Running hallucination evaluation...")
            hallucination_results = run_evals(
                dataframe=eval_df,
                evaluators=[self.evaluators["hallucination"]],
                concurrency=1  # Very conservative concurrency
            )
            
            # Add longer delay between evaluations to respect rate limits
            logger.info("⏳ Waiting 60 seconds to respect rate limits...")
            time.sleep(60)
            
            # Run QA correctness evaluation with conservative rate limiting
            logger.info("🔍 Running QA correctness evaluation...")
            qa_results = run_evals(
                dataframe=eval_df,
                evaluators=[self.evaluators["qa"]],
                concurrency=1  # Very conservative concurrency
            )
            
            # Calculate metrics
            hallucination_rate = 1 - hallucination_results["score"].mean()
            qa_accuracy = qa_results["score"].mean()
            
            metrics = {
                'hallucination_rate': hallucination_rate,
                'qa_accuracy': qa_accuracy
            }
            
            logger.info(f"✅ Phoenix Metrics: Hallucination={hallucination_rate:.3f}, QA Accuracy={qa_accuracy:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Phoenix metrics calculation failed: {e}")
            return {'hallucination_rate': 1.0, 'qa_accuracy': 0.0}
    
    def calculate_llm_accuracy(self) -> float:
        """Calculate LLM-based accuracy"""
        try:
            # Prepare data for LLM accuracy
            accuracy_data = []
            
            for _, row in self.results_df.iterrows():
                accuracy_data.append({
                    "question": row['question'],
                    "reference": row['ground_truth'],
                    "input": row['question'],
                    "output": row['generated_answer']
                })
            
            accuracy_df = pd.DataFrame(accuracy_data)
            
            # Use GPT-4 for accuracy evaluation
            openai_api_key = os.getenv("OPENAI_API_KEY")
            eval_model = OpenAIModel(
                model="gpt-4",
                temperature=0.0,
                api_key=openai_api_key
            )
            
            # Run accuracy evaluation with conservative rate limiting
            logger.info("🔍 Running LLM accuracy evaluation...")
            accuracy_eval_df = llm_classify(
                dataframe=accuracy_df,
                model=eval_model,
                template=QA_PROMPT_TEMPLATE,
                rails=list(QA_PROMPT_RAILS_MAP.values()),
                provide_explanation=True,
                concurrency=1  # Very conservative concurrency
            )
            
            accuracy_score = accuracy_eval_df["score"].mean()
            
            logger.info(f"✅ LLM-based Accuracy: {accuracy_score:.3f}")
            return accuracy_score
            
        except Exception as e:
            logger.error(f"❌ LLM accuracy calculation failed: {e}")
            return 0.0
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation on saved results"""
        logger.info("🚀 Starting Phoenix evaluation on saved results...")
        logger.info("📊 Rate limiting enabled: 1 concurrent request, 60s delays between evaluations")
        
        # Load results
        self.load_results()
        
        # Setup evaluators
        self.setup_evaluators()
        
        # Calculate all metrics
        rouge_metrics = self.calculate_rouge_metrics()
        phoenix_metrics = self.calculate_phoenix_metrics()
        llm_accuracy = self.calculate_llm_accuracy()
        
        # Combine all metrics
        final_metrics = {
            **rouge_metrics,
            **phoenix_metrics,
            'accuracy': llm_accuracy,
            'total_questions': len(self.results_df),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self._save_metrics(final_metrics)
        
        # Print summary
        self._print_summary(final_metrics)
        
        return final_metrics
    
    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save evaluation metrics"""
        try:
            # Save metrics summary
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv("phoenix_evaluation_metrics.csv", index=False)
            
            logger.info("✅ Phoenix metrics saved to phoenix_evaluation_metrics.csv")
            
        except Exception as e:
            logger.error(f"❌ Failed to save metrics: {e}")
    
    def _print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("🎯 PHOENIX EVALUATION RESULTS")
        print("="*60)
        
        print(f"📊 Total Questions Evaluated: {metrics['total_questions']}")
        print(f"⏰ Evaluation Timestamp: {metrics['evaluation_timestamp']}")
        print()
        
        print("📈 CORE METRICS:")
        print(f"   • Precision: {metrics.get('precision', 0):.3f}")
        print(f"   • Recall: {metrics.get('recall', 0):.3f}")
        print(f"   • F1 Score: {metrics.get('f1', 0):.3f}")
        print(f"   • Accuracy: {metrics.get('accuracy', 0):.3f}")
        print()
        
        print("🔍 PHOENIX METRICS:")
        print(f"   • Hallucination Rate: {metrics.get('hallucination_rate', 0):.3f}")
        print(f"   • QA Correctness: {metrics.get('qa_accuracy', 0):.3f}")
        print()
        
        print("🎯 OVERALL PERFORMANCE:")
        overall_score = (
            metrics.get('f1', 0) * 0.4 +
            metrics.get('accuracy', 0) * 0.3 +
            (1 - metrics.get('hallucination_rate', 0)) * 0.3
        )
        print(f"   • Overall Score: {overall_score:.3f}")
        print()
        
        print("="*60)
        print("🎉 Phoenix evaluation completed successfully!")
        print("📁 Results saved to: phoenix_evaluation_metrics.csv")
        print("="*60)

def main():
    """Main function"""
    try:
        # Initialize evaluator
        evaluator = PhoenixEvaluator()
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
