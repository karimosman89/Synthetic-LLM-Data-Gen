#!/usr/bin/env python3
"""
Synthetic LLM Data Generator
Generate high-quality synthetic data for training language models.
"""

import os
import json
import random
import openai
from typing import List, Dict, Any, Optional
from datetime import datetime
import csv

class SyntheticDataGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the synthetic data generator."""
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.data_templates = {
            "conversation": {
                "system_prompt": "Generate a natural conversation between two people about {topic}.",
                "format": "dialogue"
            },
            "instruction_following": {
                "system_prompt": "Create an instruction-response pair for {topic}.",
                "format": "instruction_response"
            },
            "question_answering": {
                "system_prompt": "Generate a question and detailed answer about {topic}.",
                "format": "qa"
            },
            "creative_writing": {
                "system_prompt": "Write a creative piece about {topic} in {style} style.",
                "format": "text"
            },
            "code_generation": {
                "system_prompt": "Create a coding problem and solution for {topic}.",
                "format": "code"
            }
        }
    
    def generate_synthetic_data(self, 
                              data_type: str, 
                              topic: str, 
                              count: int = 10,
                              style: str = "neutral") -> List[Dict[str, Any]]:
        """
        Generate synthetic data of specified type and topic.
        
        Args:
            data_type: Type of data to generate
            topic: Topic for the data
            count: Number of examples to generate
            style: Style of the generated data
            
        Returns:
            List of generated data examples
        """
        if data_type not in self.data_templates:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        template = self.data_templates[data_type]
        generated_data = []
        
        print(f"Generating {count} examples of {data_type} data about {topic}...")
        
        for i in range(count):
            try:
                # Create the prompt
                prompt = template["system_prompt"].format(topic=topic, style=style)
                
                # Generate the data
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a high-quality synthetic data generator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8
                )
                
                content = response.choices[0].message.content
                
                # Format the data based on type
                formatted_data = self._format_data(content, template["format"], topic, style)
                formatted_data["id"] = f"{data_type}_{topic}_{i+1}"
                formatted_data["generated_at"] = datetime.now().isoformat()
                
                generated_data.append(formatted_data)
                
                print(f"Generated example {i+1}/{count}")
                
            except Exception as e:
                print(f"Error generating example {i+1}: {str(e)}")
                continue
        
        return generated_data
    
    def _format_data(self, content: str, format_type: str, topic: str, style: str) -> Dict[str, Any]:
        """Format the generated content based on the specified format."""
        base_data = {
            "topic": topic,
            "style": style,
            "format": format_type,
            "content": content
        }
        
        if format_type == "dialogue":
            # Try to parse dialogue format
            lines = content.split('\n')
            dialogue = []
            for line in lines:
                if ':' in line:
                    speaker, text = line.split(':', 1)
                    dialogue.append({"speaker": speaker.strip(), "text": text.strip()})
            base_data["dialogue"] = dialogue
            
        elif format_type == "instruction_response":
            # Try to parse instruction-response format
            parts = content.split('\n\n', 1)
            if len(parts) == 2:
                base_data["instruction"] = parts[0].strip()
                base_data["response"] = parts[1].strip()
            
        elif format_type == "qa":
            # Try to parse question-answer format
            parts = content.split('\n\n', 1)
            if len(parts) == 2:
                base_data["question"] = parts[0].strip()
                base_data["answer"] = parts[1].strip()
        
        return base_data
    
    def augment_dataset(self, original_data: List[Dict[str, Any]], augmentation_factor: int = 2) -> List[Dict[str, Any]]:
        """
        Augment an existing dataset by generating variations.
        
        Args:
            original_data: Original dataset
            augmentation_factor: How many variations to generate per original example
            
        Returns:
            Augmented dataset
        """
        augmented_data = []
        
        for original in original_data:
            # Add original data
            augmented_data.append(original)
            
            # Generate variations
            for i in range(augmentation_factor):
                try:
                    prompt = f"Create a variation of the following content while maintaining the same meaning and format:\n{original.get('content', '')}"
                    
                    response = self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a data augmentation specialist."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7
                    )
                    
                    variation = original.copy()
                    variation["content"] = response.choices[0].message.content
                    variation["id"] = f"{original.get('id', 'unknown')}_var_{i+1}"
                    variation["is_variation"] = True
                    variation["original_id"] = original.get("id")
                    
                    augmented_data.append(variation)
                    
                except Exception as e:
                    print(f"Error creating variation: {str(e)}")
                    continue
        
        return augmented_data
    
    def save_data(self, data: List[Dict[str, Any]], filename: str, format_type: str = "json"):
        """
        Save generated data to file.
        
        Args:
            data: Data to save
            filename: Output filename
            format_type: Output format (json, csv, jsonl)
        """
        if format_type == "json":
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        elif format_type == "jsonl":
            with open(filename, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
        elif format_type == "csv":
            if data:
                fieldnames = data[0].keys()
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
        
        print(f"Data saved to {filename}")
    
    def validate_data_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the quality of generated data.
        
        Args:
            data: Data to validate
            
        Returns:
            Quality metrics
        """
        metrics = {
            "total_examples": len(data),
            "average_content_length": 0,
            "unique_topics": set(),
            "format_distribution": {},
            "style_distribution": {}
        }
        
        total_length = 0
        for item in data:
            content = item.get("content", "")
            total_length += len(content)
            
            metrics["unique_topics"].add(item.get("topic", "unknown"))
            
            format_type = item.get("format", "unknown")
            metrics["format_distribution"][format_type] = metrics["format_distribution"].get(format_type, 0) + 1
            
            style = item.get("style", "unknown")
            metrics["style_distribution"][style] = metrics["style_distribution"].get(style, 0) + 1
        
        if data:
            metrics["average_content_length"] = total_length / len(data)
        
        metrics["unique_topics"] = len(metrics["unique_topics"])
        
        return metrics

def main():
    """Main function to run the synthetic data generator."""
    generator = SyntheticDataGenerator()
    
    print("Synthetic LLM Data Generator")
    print("=" * 35)
    
    while True:
        print("\nOptions:")
        print("1. Generate new synthetic data")
        print("2. Augment existing dataset")
        print("3. Validate data quality")
        print("4. Quit")
        
        choice = input("\nSelect an option (1-4): ")
        
        if choice == '1':
            print("\nAvailable data types:")
            for i, data_type in enumerate(generator.data_templates.keys(), 1):
                print(f"{i}. {data_type}")
            
            type_choice = input("Select data type (number): ")
            try:
                data_type = list(generator.data_templates.keys())[int(type_choice) - 1]
            except (ValueError, IndexError):
                print("Invalid choice.")
                continue
            
            topic = input("Enter topic: ")
            count = int(input("Number of examples (default 5): ") or "5")
            style = input("Style (default 'neutral'): ") or "neutral"
            
            data = generator.generate_synthetic_data(data_type, topic, count, style)
            
            if data:
                filename = input("Save filename (without extension): ")
                format_type = input("Format (json/csv/jsonl, default json): ") or "json"
                generator.save_data(data, f"{filename}.{format_type}", format_type)
                
                # Show quality metrics
                metrics = generator.validate_data_quality(data)
                print("\nQuality Metrics:")
                for key, value in metrics.items():
                    print(f"{key}: {value}")
        
        elif choice == '2':
            filename = input("Enter path to existing dataset (JSON): ")
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                
                factor = int(input("Augmentation factor (default 2): ") or "2")
                augmented_data = generator.augment_dataset(original_data, factor)
                
                output_filename = input("Output filename (without extension): ")
                generator.save_data(augmented_data, f"{output_filename}.json")
                
            except Exception as e:
                print(f"Error: {str(e)}")
        
        elif choice == '3':
            filename = input("Enter path to dataset (JSON): ")
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                metrics = generator.validate_data_quality(data)
                print("\nQuality Metrics:")
                for key, value in metrics.items():
                    print(f"{key}: {value}")
                    
            except Exception as e:
                print(f"Error: {str(e)}")
        
        elif choice == '4':
            break
        
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()

