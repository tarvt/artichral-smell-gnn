import json
from openai import OpenAI
import os
import random
import re
import json
import numpy as np

# Set the API key for OpenAI
client = OpenAI(
  api_key=os.environ['OPENAI_KEY'],  # this is also the default, it can be omitted
)

def get_gpt_response_w_system(user_prompt):
    """Generate a response from GPT model given a prompt with a system message prepended."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    print(response.choices[0].message.content)
    try:
        response_json = json.loads(response.choices[0].message.content)
        result = json.dumps(response_json)
        return result
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if json_match:
            # Extract the matched JSON string
            json_str = json_match.group(0)
            
            # Attempt to parse the JSON string
            response_json = json.loads(json_str)
            result = json.dumps(response_json)
            return result
        else:
            print("No JSON structure found")


# Read the system prompt for conversation initialization
system_prompt = ""
with open('./dataset/graph_prompt.txt', 'r') as f:
    system_prompt = f.read()


# Process unprocessed IDs
def generate_with_openai():
    possible_smells = ["ESB Usage", "Cyclic Dependency", "Inappropriate Service Intimacy", "Microservice Greedy"]
    for i in range(5000):
        n = random.randint(3, 30)
        # Randomly select a number of smells from the list (between 0 and the length of possible_smells)
        selected_smells = random.sample(possible_smells, k=random.randint(0, len(possible_smells)))

        user_prompt = f"""
Create a detailed graphical representation of a microservices architecture with a specified number of microservices (n = {n}). This graph should:
- Accurately reflect the interconnections and roles of each microservice.
- Highlight potential design flaws (anti-patterns) based on the identified architectural smells.
- Include specific architectural smells as required. 

The architectural smells to be included in the graph are: {', '.join(selected_smells)}. Each smell should be clearly indicated, with visual cues or labels to identify where and how they manifest within the architecture. If no specific smells are indicated, the graph should still provide a clear overview of the architecture with potential areas for such smells to occur highlighted for further analysis.
"""
        graph_data = get_gpt_response_w_system(user_prompt)
        with open("graph_dataset.jsonl", 'a') as f:
            f.write(graph_data + '\n')
            
    

generate_with_openai()

