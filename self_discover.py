import time
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rich.panel import Panel
from rich.console import Console

console = Console()

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
                console.print(Panel("[yellow]Self Discovery in Process...[/yellow]", 
                                    border_style="bold magenta", 
                                    expand=False))
                time.sleep(sleep_time)
            self.tokens_used = num_tokens
            self.last_reset_time = time.time()

global_rate_limiter = GlobalRateLimiter(max_tokens_per_minute=6000)

class SelfDiscover:
    def __init__(self, model):
        self.model = model
        self.select_prompt = hub.pull("hwchase17/self-discovery-select")
        self.adapt_prompt = hub.pull("hwchase17/self-discovery-adapt")
        self.structured_prompt = hub.pull("hwchase17/self-discovery-structure")
        self.reasoning_prompt = hub.pull("hwchase17/self-discovery-reasoning")

        self.select_chain = self.select_prompt | self.rate_limited_model | StrOutputParser()
        self.adapt_chain = self.adapt_prompt | self.rate_limited_model | StrOutputParser()
        self.structure_chain = self.structured_prompt | self.rate_limited_model | StrOutputParser()
        self.reasoning_chain = self.reasoning_prompt | self.rate_limited_model | StrOutputParser()

        self.overall_chain = (
            RunnablePassthrough.assign(selected_modules=self.select_chain)
            .assign(adapted_modules=self.adapt_chain)
            .assign(reasoning_structure=self.structure_chain)
            .assign(answer=self.reasoning_chain)
        )

    async def rate_limited_model(self, messages):
        estimated_tokens = len(str(messages)) // 4
        global_rate_limiter.add_tokens(estimated_tokens)
        
        response = await self.model.ainvoke(messages)
        
        tokens_used = len(str(response.content)) // 4
        global_rate_limiter.add_tokens(tokens_used)
        
        return response

    async def process(self, task_description, reasoning_modules):
        result = await self.overall_chain.ainvoke({
            "task_description": task_description,
            "reasoning_modules": reasoning_modules
        })
        return result['answer']

# Define reasoning modules
reasoning_modules = [
    "1. Experimental Design: How could I devise a controlled experiment to test hypotheses related to this problem? Consider variables, control groups, and measurable outcomes.",
    
    "2. Systematic Problem-Solving: Create a comprehensive list of potential solutions, then methodically apply each one, documenting outcomes and lessons learned from each attempt.",
    
    "3. Root Cause Analysis: Employ techniques like the '5 Whys' or Ishikawa diagrams to identify the underlying causes of the problem, not just its symptoms.",
    
    "4. Systems Thinking: Analyze the problem within its larger context, identifying interconnections, feedback loops, and potential unintended consequences of proposed solutions.",
    
    "5. Stakeholder Analysis: Identify all parties affected by or influencing the problem, and consider their perspectives, needs, and potential contributions to the solution.",
    
    "6. Cost-Benefit Analysis: Evaluate potential solutions based on their expected costs (financial, time, resources) versus their anticipated benefits and long-term impact.",
    
    "7. Risk Assessment and Mitigation: Identify potential risks associated with each solution, their likelihood and impact, and develop strategies to mitigate these risks.",
    
    "8. Scenario Planning: Develop multiple future scenarios based on different assumptions and potential outcomes, then strategize for each possibility.",
    
    "9. Analogical Reasoning: Identify similar problems in other fields or contexts, and analyze how those solutions might be adapted to the current problem.",
    
    "10. Constraint Analysis: Identify all constraints (time, budget, resources, regulations) affecting the problem, then explore solutions within these boundaries or ways to alter the constraints.",
    
    "11. Data-Driven Decision Making: Gather and analyze relevant data to inform the problem-solving process, using statistical methods where appropriate.",
    
    "12. Ethical Considerations: Evaluate the moral and ethical implications of potential solutions, considering long-term consequences and societal impact.",
    
    "13. Interdisciplinary Approach: Combine insights and methodologies from multiple relevant disciplines to gain a more comprehensive understanding of the problem.",
    
    "14. Reverse Engineering: If a desired end state is known, work backwards to determine the steps needed to reach that state from the current situation.",
    
    "15. Prototyping and Iteration: Develop quick, low-cost prototypes of potential solutions, test them, gather feedback, and refine iteratively.",
    
    "16. Network Analysis: Map out the relationships and interactions between different elements of the problem, identifying key nodes and potential leverage points for intervention.",
    
    "17. Historical Analysis: Research how similar problems have been addressed throughout history, learning from both successes and failures.",
    
    "18. Counterfactual Thinking: Imagine how the situation might be different if key factors were changed, helping to identify critical variables and potential intervention points.",
    
    "19. Scalability Assessment: Evaluate how potential solutions might work at different scales, from small pilot projects to full-scale implementation.",
    
    "20. Sustainability Analysis: Consider the long-term viability of potential solutions, including their environmental, social, and economic sustainability.",
    
    "21. Cultural Sensitivity: Analyze how cultural factors might influence the problem and its potential solutions, ensuring that proposed interventions are culturally appropriate and effective.",
    
    "22. Trend Analysis and Forecasting: Examine relevant trends that might impact the problem or its solutions in the future, using techniques like trend extrapolation or predictive modeling.",
    
    "23. Resource Optimization: Analyze how to most efficiently allocate available resources (time, money, personnel, etc.) to maximize the effectiveness of the solution.",
    
    "24. Adaptive Management: Develop a flexible approach that allows for ongoing learning and adjustment of strategies as new information becomes available or circumstances change.",
    
    "25. Synergy Identification: Look for ways to combine multiple partial solutions to create a more comprehensive and effective overall solution.",
    
    "26. Constraint Relaxation: Temporarily ignore certain constraints to generate innovative ideas, then work backwards to see how these ideas might be adapted to fit within real-world limitations.",
    
    "27. Paradigm Shift Analysis: Consider how fundamentally changing the way we think about or approach the problem might lead to breakthrough solutions.",
    
    "28. Resilience Planning: Design solutions that can withstand and adapt to potential future shocks or changes in the problem's context.",
    
    "29. Opportunity Cost Evaluation: Consider not just the benefits of potential solutions, but also what opportunities might be missed by pursuing one course of action over another.",
    
    "30. Collaborative Problem-Solving: Engage diverse groups in participatory problem-solving processes, leveraging collective intelligence and building buy-in for implemented solutions."
]

reasoning_modules_str = "\n".join(reasoning_modules)