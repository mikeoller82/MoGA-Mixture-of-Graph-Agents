import json
import operator
import os
import time
from typing import Dict, TypedDict, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

load_dotenv()

console = Console()

welcome_message = Group(
    Text("Welcome to MoGA (Mixture-of-Graph-Agents)!", style="bold cyan"),
    Text("\nThis script uses the following LLMs as proposer models, then passes the results to the aggregate model for the final response:"),
    Text("- llama3-8b-8192", style="green"),
    Text("- gemma-7b-it", style="green"),
    Text("- mixtral-8x7b-32768", style="green"),
    Text("\nThe aggregator and reflector models use ", end=""),
    Text("llama3-70b-8192", style="magenta"),
    Text(".\n"),
    Text("The agents can perform web searches for up-to-date information!", style="bold green")
)

class GlobalRateLimiter:
    def __init__(self, max_tokens_per_minute):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.tokens_used = 0
        self.last_reset_time = time.time()

    def reset_if_needed(self):
        current_time = time.time()
        if current_time - self.last_reset_time >= 60:
            self.tokens_used = 0
            self.last_reset_time = current_time

    def add_tokens(self, num_tokens):
        self.reset_if_needed()
        self.tokens_used += num_tokens
        if self.tokens_used > self.max_tokens_per_minute:
            sleep_time = 60 - (time.time() - self.last_reset_time)
            if sleep_time > 0:
                console.print(Panel("[yellow]The models are collaborating...[/yellow]", 
                                    border_style="bold magenta", 
                                    expand=False))
                time.sleep(sleep_time)
            self.tokens_used = num_tokens
            self.last_reset_time = time.time()

global_rate_limiter = GlobalRateLimiter(max_tokens_per_minute=6000)

class State(TypedDict):
    input: str
    responses: str
    feedback: str
    aggregated_response: Annotated[list, operator.add]
    final_answer: str
    iteration: int
    max_iterations: int
    reflection: str

# Initialize models and search tool
proposer_models = [
    ChatGroq(temperature=0, model_name="llama3-8b-8192",),
    ChatGroq(temperature=0, model_name="gemma-7b-it",),
    ChatGroq(temperature=0, model_name="mixtral-8x7b-32768"),
]

#aggregator_model = ChatOpenAI(
#        openai_api_base="https://api.groq.com/openai/v1",
#        openai_api_key=os.getenv("GROQ_API_KEY"),
#        model_name="llama3-70b-8192"
#    )

aggregator_model = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
)

reflector_model = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192"
    )

class Proposer:
    def __init__(self, model: ChatOpenAI, id: str):
        self.model = model
        self.id = id

    def __call__(self, state: State) -> Dict[str, Dict[str, str]]:
        prompt = ChatPromptTemplate.from_messages([
            ("human", "Please provide a response to the following prompt: {input}\n\n"
             "The previous response was {response}\n\n"
             "The provided feedback was {feedback} \n\n"
             "If you need to search the web for current information, you can use the command: [SEARCH: your query]"
             "Always answer with as much detail as possible.")
        ])
        messages = prompt.format_messages(
            input=state["input"],
            response=state['responses'],
            feedback=state['feedback']
        )
        time.sleep(2)
        time.sleep(2)
    
        
        # Estimate token count (you may need to implement a more accurate method)
        estimated_tokens = len(str(messages)) // 4
        global_rate_limiter.add_tokens(estimated_tokens)
        
        response = self.model.invoke(messages)
        
        # Store the proposer's output
        with open('prop_output.md', 'a', encoding='utf-8') as f:
            f.write(f"## {self.id} Output\n\n{response.content}\n\n")
        
        # Check if a web search is requested
        if "[SEARCH:" in response.content:
            search_query = response.content.split("[SEARCH:", 1)[1].split("]", 1)[0].strip()
            search_results = perform_web_search(search_query)
            response.content += f"\n\nWeb search results for '{search_query}':\n{search_results}"
        
        # Estimate tokens and add to global counter
        tokens_used = len(str(response.content)) // 4
        global_rate_limiter.add_tokens(tokens_used)
        time.sleep(2)
        
        return {"aggregated_response": [json.dumps({self.id: response.content})]}
    
def aggregator(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert aggregator. Your task is to synthesize multiple responses into a single, high-quality answer."),
        ("human", '''
        You have been provided with a set of responses from various open-source models to the user query '{input}'.
        Your task is to synthesize these responses into a single, high-quality response.
        It is crucial to critically evaluate the information provided in these responses, 
        recognizing that some of it may be biased or incorrect. 
        Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the user query. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
 
        Responses from models:
        {responses}
        ''')
    ])
    messages = prompt.format_messages(
        input=state["input"],
        responses="\n\n".join(state["aggregated_response"])
    )
    
    time.sleep(2)
    
    # Estimate token count
    estimated_tokens = len(str(messages)) // 4
    global_rate_limiter.add_tokens(estimated_tokens)
    
    response = aggregator_model.invoke(messages)
    
    # Export aggregator's output
    write_to_file('agg.md', f"# Aggregator Output\n\n{response.content}")
    
    # Estimate tokens and add to global counter
    tokens_used = len(str(response.content)) // 4
    global_rate_limiter.add_tokens(tokens_used)
    
    state["responses"] = response.content
    return state


class GradeGeneration(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    score: str = Field(
        description="Is this the correct answer to the question, 'yes' or 'no'"
    )
    feedback: str = Field(
        description="Provided specific feedback for improvement"
    )
        
def reflector(state: State) -> State:
    parser = PydanticOutputParser(pydantic_object=GradeGeneration)

    prompt = ChatPromptTemplate.from_messages([
        ("human", "Given the following answer to the question: '{input}'\n\n"
         "Answer: {aggregated_response}\n\n"
         "If the answer is satisfactory and complete, grade it as yes. \n"
         "Provide json object with feedback string and binary score 'yes' or 'no' score to indicate whether the answer is correct"
        )
    ])
    chain = prompt | reflector_model | parser
    
    time.sleep(2)

    # Estimate token count
    estimated_tokens = len(state["input"]) // 4 + len(state["responses"]) // 4
    global_rate_limiter.add_tokens(estimated_tokens)

    response = chain.invoke({
        'input':state["input"],
        'aggregated_response':state["responses"]
    })
    
    # Export reflector's output
    write_to_file('reflect.md', f"# Reflector Output\n\nScore: {response.score}\nFeedback: {response.feedback}")
    
    # Estimate tokens and add to global counter
    tokens_used = len(str(response)) // 4
    global_rate_limiter.add_tokens(tokens_used)
    
    state["reflection"] = response.score
    state['feedback'] = response.feedback
    state["iteration"] += 1
    return state

def should_continue(state: State) -> str:
    if state["iteration"] >= state["max_iterations"] or "YES" in state["reflection"].upper():
        state["final_answer"] = state["responses"]
        return "end"
    return "refine"

# Create the graph
workflow = StateGraph(State)

# Add nodes
for i, model in enumerate(proposer_models):
    workflow.add_node(f"proposer_{i}", Proposer(model, f"proposer_{i}"))
workflow.add_node("aggregator", aggregator)
workflow.add_node("reflector", reflector)

# Define edges
workflow.set_entry_point("proposer_0")
for i in range(1, len(proposer_models)):
    workflow.add_edge(f"proposer_0", f"proposer_{i}")
    workflow.add_edge(f"proposer_{i}", "aggregator")
workflow.add_edge("aggregator", "reflector")
workflow.add_conditional_edges(
    "reflector",
    should_continue,
    {
        "refine": "proposer_0",
        "end": END
    }
)

app = workflow.compile()

def query_moa(question, max_iterations=3):
    initial_state = {
        "input": question,
        "responses": '',
        'feedback': "",
        "aggregated_response": [],
        "final_answer": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "reflection": ""
    }
    
    current_state = initial_state
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Processing...", total=max_iterations)
        for output in app.stream(initial_state):
            for key, value in output.items():
                if key.startswith("proposer"):
                    console.print(Panel(f"[bold green]{key.capitalize()} completed[/bold green]", 
                                        border_style="green", 
                                        expand=False))
                elif key == "aggregator":
                    console.print(Panel(f"[bold orange1]Aggregator completed[/bold orange1]", 
                                        border_style="orange1", 
                                        expand=False))
                elif key == "reflector":
                    console.print(Panel(f"[bold red]Reflector completed[/bold red]", 
                                        border_style="red", 
                                        expand=False))
                else:
                    console.print(Panel(f"[bold blue]{key.capitalize()} completed[/bold blue]", 
                                        border_style="blue", 
                                        expand=False))
                
                if isinstance(value, dict) and 'responses' in value:
                    current_state = value  # Update the current state
            
            # Estimate tokens for this iteration
            iteration_tokens = len(str(current_state)) // 4
            global_rate_limiter.add_tokens(iteration_tokens)
            
            progress.update(task, advance=1)
            time.sleep(2)
            
            # Check if we've reached the maximum iterations
            if current_state['iteration'] >= max_iterations:
                console.print(Panel("[yellow]Maximum iterations reached. Finalizing response.[/yellow]", 
                                    border_style="bold yellow", 
                                    expand=False))
                break
    
    return current_state

def safe_str(obj):
    """Safely convert any object to a string."""
    if obj is None:
        return "None"
    try:
        return str(obj)
    except:
        return "Error: Unable to convert to string"
    
def write_to_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
        
def display_file_contents():
    files = ['prop_output.md', 'agg.md', 'reflect.md']
    for file in files:
        if os.path.exists(file):
            console.print(f"\n[bold blue]Contents of {file}:[/bold blue]")
            with open(file, 'r', encoding='utf-8') as f:
                console.print(Panel(f.read(), expand=False, border_style="green"))
        else:
            console.print(f"[yellow]File {file} not found.[/yellow]")
    
def perform_web_search(query: str) -> str:
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error performing web search: {str(e)}"

def main():
    console.print(Panel(
        welcome_message,
        title="MoGA: Mixture of Graph Agents w/Tools",
        title_align="center",
        border_style="bold blue",
        box=box.DOUBLE,
        expand=False,
        padding=(1, 1)
    ))
    
    console.print("\n[bold cyan]Enter your query below:[/bold cyan]")
    console.print("[italic]The agents will search the web if they need current information.[/italic]")
    
    while True:
        question = Prompt.ask("\n[bold magenta]Query >>[/bold magenta]")
        
        if question.lower() in ["exit", "quit"]:
            console.print(Panel("[yellow]Thank you for using the MoGA. Goodbye![/yellow]", 
                                border_style="bold yellow", 
                                expand=False))
            break
        
        max_iterations = int(Prompt.ask(
            "[bold cyan]Max iterations[/bold cyan]", default="3", show_default=True
        ))
        
        with console.status("[bold green]Processing your query...[/bold green]") as status:
            try:
                result = query_moa(question, max_iterations=max_iterations)
            except Exception as e:
                result = {"responses": f"Error occurred: {str(e)}"}

        # Safely extract and convert the response to a string
        response = safe_str(result.get('responses', "No response generated"))

        # Create the panel content
        panel_content = Group(
            Text("Question:", style="bold blue"),
            Text(question, style="cyan"),
            Text("\nFinal Answer:", style="bold blue"),
            Text(response, style="green")
        )
        
        display_file_contents()
        # Create and print the panel
        answer_panel = Panel(
            panel_content,
            border_style="bold blue",
            expand=False,
            padding=(1, 1)
        )

        try:
            console.print(answer_panel)
        except Exception as e:
            console.print(f"[bold red]Error displaying result: {str(e)}[/bold red]")
            console.print(f"Raw response: {response}")

        console.print("\n[italic cyan]Enter another question or type 'exit' to quit.[/italic cyan]")
        
        

if __name__ == "__main__":
    main()
    

    
 