"""
Dataset Generation Pipeline for Cache Classification
Loads and combines multiple datasets to create training data for binary cache classification.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import datasets
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

class DatasetGenerator:
    """Generate and combine datasets for cache classification training."""

    def __init__(self, output_dir: str = "cache_classifier_data", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.seed = seed
        random.seed(seed)

    def load_ms_marco(self, max_samples: int = 10000) -> List[Dict[str, Any]]:
        """
        Load MS MARCO queries (diverse, realistic search queries).
        Mix of cacheable and non-cacheable queries.
        """
        print("Loading MS MARCO dataset...")
        try:
            # Load MS MARCO v2.1 (queries)
            dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)

            queries = []
            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break

                query = example.get("query", "")
                if query and len(query) > 10:  # Filter short/empty queries
                    queries.append({
                        "text": query,
                        "source": "ms_marco",
                        "label": None  # To be labeled by LLM
                    })

            print(f"Loaded {len(queries)} queries from MS MARCO")
            return queries

        except Exception as e:
            print(f"Error loading MS MARCO: {e}")
            print("Skipping MS MARCO - you may need to accept terms at https://huggingface.co/datasets/microsoft/ms_marco")
            return []

    def load_trivia_qa(self, max_samples: int = 5000) -> List[Dict[str, Any]]:
        """
        Load TriviaQA - factual trivia questions (mostly cacheable).
        """
        print("Loading TriviaQA dataset...")
        try:
            dataset = load_dataset("trivia_qa", "rc.nocontext", split="train", streaming=True)

            queries = []
            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break

                question = example.get("question", "")
                if question:
                    queries.append({
                        "text": question,
                        "source": "trivia_qa",
                        "label": 1,  # Trivia is almost always cacheable
                        "ground_truth_reason": "Factual trivia question with static answer"
                    })

            print(f"Loaded {len(queries)} questions from TriviaQA")
            return queries

        except Exception as e:
            print(f"Error loading TriviaQA: {e}")
            return []

    def load_natural_questions(self, max_samples: int = 5000) -> List[Dict[str, Any]]:
        """
        Load Natural Questions - Google search queries (mostly factual/cacheable).
        """
        print("Loading Natural Questions dataset...")
        try:
            dataset = load_dataset("google-research-datasets/natural_questions", split="train", streaming=True)

            queries = []
            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break

                question = example.get("question", {}).get("text", "")
                if question and len(question) > 10:
                    queries.append({
                        "text": question,
                        "source": "natural_questions",
                        "label": 1,  # Mostly factual/cacheable
                        "ground_truth_reason": "Factual question from Google search"
                    })

            print(f"Loaded {len(queries)} questions from Natural Questions")
            return queries

        except Exception as e:
            print(f"Error loading Natural Questions: {e}")
            return []

    def load_squad(self, max_samples: int = 3000) -> List[Dict[str, Any]]:
        """
        Load SQuAD - Wikipedia-based reading comprehension (cacheable).
        """
        print("Loading SQuAD dataset...")
        try:
            dataset = load_dataset("rajpurkar/squad", split="train")

            queries = []
            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break

                question = example.get("question", "")
                if question:
                    queries.append({
                        "text": question,
                        "source": "squad",
                        "label": 1,  # Wikipedia facts are cacheable
                        "ground_truth_reason": "Wikipedia-based factual question"
                    })

            print(f"Loaded {len(queries)} questions from SQuAD")
            return queries

        except Exception as e:
            print(f"Error loading SQuAD: {e}")
            return []

    def generate_time_sensitive_queries(self, count: int = 3000) -> List[Dict[str, Any]]:
        """
        Generate synthetic time-sensitive queries (NOT cacheable).
        """
        print(f"Generating {count} time-sensitive queries...")

        templates = [
            # Current prices/stocks
            "What is the current price of {} in USD?",
            "What is {}'s stock price right now?",
            "How much does {} cost today?",
            "What is the latest price of {}?",

            # Weather
            "What's the weather like in {} today?",
            "What is the current temperature in {}?",
            "Will it rain in {} tomorrow?",
            "What's the weather forecast for {} this week?",

            # Time
            "What time is it right now in {}?",
            "What is the current time in {}?",

            # Live events/scores
            "Who won the last {} game?",
            "What is the score of the {} match right now?",
            "Who is winning the {} championship?",

            # News/Current events
            "What are today's top headlines on {}?",
            "What is the latest news about {}?",
            "What happened in {} today?",

            # Exchange rates
            "What is the current exchange rate {} to {}?",
            "How much is {} worth in {} today?",

            # Blockchain/crypto
            "What is the current block height of {}?",
            "How many transactions are pending on {} right now?",
            "What is the current {} hashrate?",

            # Software versions (frequently changing)
            "What is the latest version of {} released today?",
            "What was the most recent commit on {}?",

            # Age (changes yearly)
            "How old is {} right now?",
            "What is the current age of {}?",
        ]

        entities = [
            # Crypto/stocks
            "Bitcoin", "Ethereum", "Tesla", "Apple", "Google", "NVIDIA", "Microsoft",
            "gold", "silver", "oil", "EUR", "JPY", "GBP", "Amazon stock",

            # Cities
            "London", "Tokyo", "New York", "Paris", "Sydney", "Hong Kong", "Berlin",
            "Singapore", "Dubai", "Moscow", "Beijing",

            # Sports
            "NBA", "NFL", "Premier League", "Formula 1", "Champions League", "World Cup",

            # News sources
            "CNN", "BBC", "Hacker News", "Reddit", "Twitter",

            # Tech
            "Python", "Node.js", "React", "Kubernetes", "Docker", "PostgreSQL",
            "the Linux kernel", "TensorFlow", "PyTorch",

            # Blockchain
            "Ethereum", "Bitcoin", "Polygon", "Solana",

            # People (living, age changes)
            "Elon Musk", "Bill Gates", "Jeff Bezos", "Mark Zuckerberg", "Sam Altman",
        ]

        queries = []
        for _ in range(count):
            template = random.choice(templates)

            # Count placeholders in template
            num_placeholders = template.count("{}")

            if num_placeholders == 1:
                entity = random.choice(entities)
                query = template.format(entity)
            elif num_placeholders == 2:
                entity1 = random.choice(entities[:7])  # currencies/stocks
                entity2 = random.choice(entities[:7])
                query = template.format(entity1, entity2)
            else:
                continue

            queries.append({
                "text": query,
                "source": "synthetic_time_sensitive",
                "label": 0,  # NOT cacheable
                "ground_truth_reason": "Time-sensitive query with dynamic answer"
            })

        print(f"Generated {len(queries)} time-sensitive queries")
        return queries

    def generate_creative_queries(self, count: int = 2000) -> List[Dict[str, Any]]:
        """
        Generate synthetic creative/open-ended queries (NOT cacheable).
        """
        print(f"Generating {count} creative queries...")

        templates = [
            "Write a {} about {}.",
            "Generate {} ideas for {}.",
            "Create a {} for {}.",
            "Come up with a {} for {}.",
            "Help me write a {} about {}.",
            "Suggest a {} for {}.",
            "Role-play as {} and {}.",
            "Tell me a {} about {}.",
        ]

        creative_types = [
            "poem", "story", "joke", "essay", "script", "song", "haiku",
            "creative name", "slogan", "tagline", "title", "bedtime story",
            "horror story", "funny story", "workout plan", "meal plan",
            "travel itinerary", "business plan"
        ]

        topics = [
            "cats", "technology", "space", "cooking", "programming", "AI",
            "a coffee shop", "a startup", "a robot", "the future", "Mars",
            "dinosaurs", "dragons", "a detective", "a programmer", "climate change"
        ]

        queries = []
        for _ in range(count):
            template = random.choice(templates)
            creative_type = random.choice(creative_types)
            topic = random.choice(topics)

            query = template.format(creative_type, topic)

            queries.append({
                "text": query,
                "source": "synthetic_creative",
                "label": 0,  # NOT cacheable
                "ground_truth_reason": "Creative/generative query with variable output"
            })

        print(f"Generated {len(queries)} creative queries")
        return queries

    def generate_computation_queries(self, count: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate computational queries (mostly NOT cacheable unless pure facts).
        """
        print(f"Generating {count} computation queries...")

        templates = [
            "What is {} × {}?",
            "Calculate {} + {}.",
            "Solve {} - {} for x.",
            "What is {} divided by {}?",
            "Compute {} to the power of {}.",
            "Find the {} of {}.",
            "What is the {} derivative of {}?",
        ]

        queries = []
        for _ in range(count // 2):
            num1 = random.randint(10, 999)
            num2 = random.randint(10, 999)

            template = random.choice(templates[:5])
            query = template.format(num1, num2)

            queries.append({
                "text": query,
                "source": "synthetic_computation",
                "label": 0,  # Computation typically not cached
                "ground_truth_reason": "Requires computation, not lookup"
            })

        # Add some pure math facts (cacheable)
        math_facts = [
            "What is sin(π/3)?",
            "What is the value of π to 5 decimal places?",
            "What is Euler's identity?",
            "What is the square root of 144?",
            "What is e (Euler's number)?",
            "What is the Pythagorean theorem?",
            "What is the quadratic formula?",
            "What does i² equal in complex numbers?",
        ]

        for fact in math_facts * (count // (2 * len(math_facts))):
            queries.append({
                "text": fact,
                "source": "math_facts",
                "label": 1,  # Math facts are cacheable
                "ground_truth_reason": "Pure mathematical fact/constant"
            })

        print(f"Generated {len(queries)} computation queries")
        return queries

    def generate_code_queries(self, count: int = 1500) -> List[Dict[str, Any]]:
        """
        Generate code-related queries (mix of facts and generation).
        """
        print(f"Generating {count} code-related queries...")

        # Code generation (NOT cacheable)
        generation_templates = [
            "Write a {} function in {} that {}.",
            "Implement {} in {}.",
            "Create a {} program that {}.",
            "Build a {} in {} for {}.",
        ]

        languages = ["Python", "JavaScript", "Rust", "Go", "Java", "C++"]
        tasks = [
            "sorts an array", "reverses a string", "finds prime numbers",
            "validates email", "parses JSON", "makes API calls",
            "handles errors", "uses async/await"
        ]

        queries = []

        # Generation queries (NOT cacheable)
        for _ in range(count // 2):
            template = random.choice(generation_templates)
            lang = random.choice(languages)
            task = random.choice(tasks)

            query = template.format(task, lang, random.choice(tasks))
            queries.append({
                "text": query,
                "source": "synthetic_code_gen",
                "label": 0,
                "ground_truth_reason": "Code generation has high variance"
            })

        # Code facts (cacheable)
        fact_templates = [
            "What is the syntax for {} in {}?",
            "How do you {} in {}?",
            "What does {} do in {}?",
            "What is the default {} in {}?",
        ]

        concepts = [
            "list comprehension", "virtual environment", "async/await",
            "arrow functions", "destructuring", "generators", "decorators",
        ]

        for _ in range(count // 2):
            template = random.choice(fact_templates)
            lang = random.choice(languages)
            concept = random.choice(concepts)

            query = template.format(concept, lang)
            queries.append({
                "text": query,
                "source": "code_facts",
                "label": 1,  # Facts are cacheable
                "ground_truth_reason": "Factual question about syntax/concept"
            })

        print(f"Generated {len(queries)} code queries")
        return queries

    def combine_and_balance(self, all_queries: List[Dict[str, Any]],
                           target_size: int = 50000) -> List[Dict[str, Any]]:
        """
        Combine all queries, balance classes, and sample to target size.
        """
        print("\nCombining and balancing dataset...")

        # Separate by label status
        labeled_positive = [q for q in all_queries if q.get("label") == 1]
        labeled_negative = [q for q in all_queries if q.get("label") == 0]
        unlabeled = [q for q in all_queries if q.get("label") is None]

        print(f"Labeled positive (cacheable): {len(labeled_positive)}")
        print(f"Labeled negative (non-cacheable): {len(labeled_negative)}")
        print(f"Unlabeled (for LLM labeling): {len(unlabeled)}")

        # Balance labeled examples
        min_labeled = min(len(labeled_positive), len(labeled_negative))
        balanced_positive = random.sample(labeled_positive, min_labeled)
        balanced_negative = random.sample(labeled_negative, min_labeled)

        # Combine
        labeled_balanced = balanced_positive + balanced_negative
        random.shuffle(labeled_balanced)

        # Sample unlabeled
        unlabeled_sample = random.sample(unlabeled, min(len(unlabeled), target_size // 3))

        # Combine all
        final_dataset = labeled_balanced + unlabeled_sample
        random.shuffle(final_dataset)

        # Trim to target size
        if len(final_dataset) > target_size:
            final_dataset = final_dataset[:target_size]

        print(f"\nFinal dataset size: {len(final_dataset)}")
        print(f"  - Labeled: {len(labeled_balanced)}")
        print(f"  - Unlabeled: {len(unlabeled_sample)}")

        return final_dataset

    def save_dataset(self, queries: List[Dict[str, Any]], filename: str = "raw_dataset.jsonl"):
        """Save dataset to JSONL file."""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            for query in queries:
                f.write(json.dumps(query, ensure_ascii=False) + '\n')

        print(f"\nSaved {len(queries)} queries to {output_path}")

        # Also save statistics
        stats = {
            "total": len(queries),
            "labeled": len([q for q in queries if q.get("label") is not None]),
            "unlabeled": len([q for q in queries if q.get("label") is None]),
            "cacheable": len([q for q in queries if q.get("label") == 1]),
            "non_cacheable": len([q for q in queries if q.get("label") == 0]),
            "sources": {}
        }

        for query in queries:
            source = query.get("source", "unknown")
            stats["sources"][source] = stats["sources"].get(source, 0) + 1

        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Saved statistics to {stats_path}")
        return output_path

    def generate_full_pipeline(self,
                              use_ms_marco: bool = True,
                              use_trivia_qa: bool = True,
                              use_natural_questions: bool = False,  # Large download
                              use_squad: bool = True,
                              target_size: int = 50000) -> Path:
        """
        Run the full dataset generation pipeline.
        """
        print("="*60)
        print("Starting Dataset Generation Pipeline")
        print("="*60)

        all_queries = []

        # Load public datasets
        if use_ms_marco:
            all_queries.extend(self.load_ms_marco(max_samples=10000))

        if use_trivia_qa:
            all_queries.extend(self.load_trivia_qa(max_samples=5000))

        if use_natural_questions:
            all_queries.extend(self.load_natural_questions(max_samples=5000))

        if use_squad:
            all_queries.extend(self.load_squad(max_samples=3000))

        # Generate synthetic queries
        all_queries.extend(self.generate_time_sensitive_queries(count=3000))
        all_queries.extend(self.generate_creative_queries(count=2000))
        all_queries.extend(self.generate_computation_queries(count=1000))
        all_queries.extend(self.generate_code_queries(count=1500))

        # Combine and balance
        final_dataset = self.combine_and_balance(all_queries, target_size=target_size)

        # Save
        output_path = self.save_dataset(final_dataset)

        print("\n" + "="*60)
        print("Dataset Generation Complete!")
        print("="*60)

        return output_path


def main():
    """Main execution."""
    generator = DatasetGenerator(output_dir="cache_classifier_data")

    # Generate dataset
    dataset_path = generator.generate_full_pipeline(
        use_ms_marco=True,
        use_trivia_qa=True,
        use_natural_questions=False,  # Set to True if you want it (large download)
        use_squad=True,
        target_size=50000
    )

    print(f"\nNext step: Run llm_labeling.py to label unlabeled queries")
    print(f"Dataset saved to: {dataset_path}")


if __name__ == "__main__":
    main()
