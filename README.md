# MoGA (Mixture-of-Graph-Agents)

MoGA is a powerful question-answering system that utilizes multiple language models and a graph-based approach to generate high-quality responses. It combines the strengths of various models, performs web searches for up-to-date information, and uses an iterative refinement process to produce comprehensive answers.

## Features

- Uses multiple LLMs as proposer models (llama3-8b-8192, gemma-7b-it, mixtral-8x7b-32768)Groq Models
- Aggregates responses using llama3-70b-8192
- Performs web searches for current information
- Iterative refinement process with reflection
- Rate limiting to manage API usage

## Installation

1. Clone the repository: git clone https://github.com/mikeoller82/moga.git
cd moga
2. Create a virtual environment:python -m venv venv
source venv/bin/activate
# On Windows, use venv\Scripts\activate

3. Install the required packages: pip install -r requirements.txt

4. 4. Set up your environment variables:
Create a `.env` file in the project root and add your API keys:

OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
TAVILY_API_KEY=your_tavily_api_key

## Usage

Run the main script: python moga.py

Follow the prompts to enter your query and specify the maximum number of iterations.

## License

This project is licensed under the MIT License - see the [LICENSE] file for details.
