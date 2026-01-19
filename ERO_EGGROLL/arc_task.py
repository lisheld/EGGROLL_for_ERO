"""
ARC-1 Visual Reasoning Task Adapter for EGGROLL

This module adapts ERO's ARC-1 dataset format to EGGROLL's BanditTask interface.
It handles multimodal vision-language inputs and implements the fitness calculation
using character-level similarity matching.

ARC-1 (Abstraction and Reasoning Corpus) tasks:
- Visual pattern recognition
- Abstract reasoning
- Grid transformations
- 15 tasks total
"""

import sys
import os
import json
import numpy as np
from typing import List, Tuple, Dict, Any

# Add ERO to path to import utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ERO'))

from recorder import Recorder
from utils_fn import filter_text, fill_matrix, calculate_similarity_between
from config import DATA_PATH, ARC_TASKS, NUM_TRAIN_EXAMPLES


class ARCVisualTask:
    """
    ARC-1 visual reasoning task adapter.

    Unlike EGGROLL's text-only BanditTask which returns token arrays,
    this task works with vision-language models that need:
    - Multimodal prompts (text + grid visualizations)
    - Structured output parsing
    - Grid-based fitness evaluation

    The task interface is simplified for single-GPU sequential evaluation
    instead of batched parallel evaluation.
    """

    def __init__(self,
                 processor,
                 data_path: str = DATA_PATH,
                 task_ids: List[str] = ARC_TASKS,
                 max_answer_length: int = 500):
        """
        Initialize ARC visual reasoning task.

        Args:
            processor: Qwen2.5-VL AutoProcessor for text processing
            data_path: Path to ARC JSON files
            task_ids: List of ARC task IDs to use
            max_answer_length: Maximum generation length
        """
        self.processor = processor
        self.data_path = data_path
        self.task_ids = task_ids
        self.max_answer_length = max_answer_length

        # Load task data using ERO's Recorder
        self.recorder = Recorder(output_path=".", datapath=data_path)

        # Pre-build prompts for all tasks
        self.prompts = {}
        self.ground_truth = {}

        for task_id in task_ids:
            self.prompts[task_id] = self._build_arc_prompt(task_id)
            self.ground_truth[task_id] = self.recorder.get(task_id, 'answer')

        print(f"Loaded {len(task_ids)} ARC tasks from {data_path}")

    def __len__(self):
        """Return number of tasks."""
        return len(self.task_ids)

    def _build_arc_prompt(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Build multimodal prompt for ARC task following ERO format.

        The prompt structure matches ERO's implementation (ERO/models_fn.py:125-144):
        1. System message: Expert in ARC visual reasoning
        2. User message:
           - Training examples (input/output grids)
           - Test input grid
           - Instruction to output only the 2D array

        Args:
            task_id: ARC task identifier

        Returns:
            List of message dicts for processor.apply_chat_template()
        """
        # System prompt (from ERO)
        system_content = (
            "You are an expert in visual abstract reasoning and pattern recognition. "
            "You excel at identifying transformation rules from examples and applying "
            "them to new inputs. Your task is to analyze grid patterns and predict outputs."
        )

        # Build user prompt with training examples
        input1 = str(self.recorder.get(task_id, 'input1'))
        output1 = str(self.recorder.get(task_id, 'output1'))
        input2 = str(self.recorder.get(task_id, 'input2'))
        output2 = str(self.recorder.get(task_id, 'output2'))
        question = str(self.recorder.get(task_id, 'question'))

        user_content = f"""Below is an ARC reasoning task with {NUM_TRAIN_EXAMPLES} training examples.

Example 1:
Input grid: {input1}
Output grid: {output1}

Example 2:
Input grid: {input2}
Output grid: {output2}

Test input grid: {question}

Analyze the pattern and transformation rule from the training examples.
Only output the final **Predicted Output** as a 2D array of integers.
Do not include any explanation or reasoning."""

        # Format as chat messages
        messages = [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        return messages

    def get_input(self, task_idx: int) -> str:
        """
        Get prompt text for a specific task.

        Args:
            task_idx: Index into self.task_ids

        Returns:
            Formatted text prompt ready for processing
        """
        task_id = self.task_ids[task_idx]
        messages = self.prompts[task_id]

        # Apply chat template to convert messages to text
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return text

    def get_processed_inputs(self, task_idx: int, device: str = "cuda:0") -> Dict:
        """
        Get fully processed inputs ready for model.generate().

        Args:
            task_idx: Index into self.task_ids
            device: Target device for tensors

        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        import torch

        text = self.get_input(task_idx)

        # Process with Qwen2.5-VL processor
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        return inputs

    def calculate_fitness(self,
                          task_idx: int,
                          generated_text: str) -> float:
        """
        Calculate fitness for a generated output.

        Uses ERO's character-level similarity metric:
        - Filter to extract grid structure
        - Fill incomplete matrices
        - Character-by-character comparison

        Args:
            task_idx: Index into self.task_ids
            generated_text: Model's generated output

        Returns:
            Fitness score [0.0, 1.0] where 1.0 is perfect match
        """
        task_id = self.task_ids[task_idx]

        # Get ground truth
        answer = str(self.ground_truth[task_id])

        # Post-process generated output (ERO's pipeline)
        # 1. Filter to keep only grid structure characters
        filtered_output = filter_text(generated_text)

        # 2. Fill incomplete matrices with zeros
        filled_output = fill_matrix(filtered_output)

        # 3. Re-filter to ensure clean format
        result = filter_text(filled_output)

        # 4. Filter ground truth for fair comparison
        answer_filtered = filter_text(answer)

        # 5. Calculate character-level similarity
        fitness = calculate_similarity_between(result, answer_filtered)

        return fitness

    def get_batch_fitness(self,
                          task_indices: List[int],
                          generated_texts: List[str]) -> np.ndarray:
        """
        Calculate fitness for a batch of generated outputs.

        Args:
            task_indices: List of task indices
            generated_texts: List of generated outputs (same length as task_indices)

        Returns:
            NumPy array of fitness scores
        """
        fitnesses = []

        for task_idx, gen_text in zip(task_indices, generated_texts):
            fitness = self.calculate_fitness(task_idx, gen_text)
            fitnesses.append(fitness)

        return np.array(fitnesses, dtype=np.float32)

    def evaluate_all_tasks(self, generated_texts: List[str]) -> Tuple[float, np.ndarray]:
        """
        Evaluate model on all ARC tasks.

        Args:
            generated_texts: List of generated outputs (one per task)

        Returns:
            Tuple of (mean_fitness, fitness_array)
        """
        assert len(generated_texts) == len(self.task_ids), \
            f"Expected {len(self.task_ids)} outputs, got {len(generated_texts)}"

        task_indices = list(range(len(self.task_ids)))
        fitnesses = self.get_batch_fitness(task_indices, generated_texts)

        mean_fitness = np.mean(fitnesses)

        return mean_fitness, fitnesses

    def get_task_info(self, task_idx: int) -> Dict[str, Any]:
        """Get detailed information about a task."""
        task_id = self.task_ids[task_idx]

        return {
            'task_id': task_id,
            'input1': self.recorder.get(task_id, 'input1'),
            'output1': self.recorder.get(task_id, 'output1'),
            'input2': self.recorder.get(task_id, 'input2'),
            'output2': self.recorder.get(task_id, 'output2'),
            'question': self.recorder.get(task_id, 'question'),
            'answer': self.recorder.get(task_id, 'answer'),
        }


def test_arc_task():
    """Test ARC task loading and prompt generation."""
    print("Testing ARC Task Adapter")
    print("=" * 80)

    try:
        # Mock processor (for testing without loading full model)
        class MockProcessor:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                # Simple text concatenation
                text = ""
                for msg in messages:
                    text += f"{msg['role']}: {msg['content']}\n\n"
                return text

        processor = MockProcessor()

        # Load first 3 tasks for testing
        test_task_ids = ARC_TASKS[:3]
        task = ARCVisualTask(processor, DATA_PATH, test_task_ids)

        print(f"\nLoaded {len(task)} tasks")
        print(f"Task IDs: {task.task_ids}")

        # Test prompt generation
        print("\nTesting prompt generation for task 0:")
        print("-" * 80)
        prompt = task.get_input(0)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

        # Test ground truth access
        print("\nGround truth for task 0:")
        print(task.ground_truth[task.task_ids[0]][:200] + "...")

        # Test fitness calculation
        print("\nTesting fitness calculation:")
        # Perfect match
        task_id = task.task_ids[0]
        perfect_output = str(task.ground_truth[task_id])
        fitness_perfect = task.calculate_fitness(0, perfect_output)
        print(f"  Perfect match fitness: {fitness_perfect:.4f} (should be ~1.0)")

        # Random output
        random_output = "[[1, 2, 3], [4, 5, 6]]"
        fitness_random = task.calculate_fitness(0, random_output)
        print(f"  Random output fitness: {fitness_random:.4f} (should be <1.0)")

        # Empty output
        fitness_empty = task.calculate_fitness(0, "")
        print(f"  Empty output fitness: {fitness_empty:.4f} (should be 0.0)")

        print("\n✓ ARC task adapter test passed")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_arc_task()
