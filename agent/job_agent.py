import os
from typing import Literal
import asyncio
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from deepagents import create_deep_agent, SubAgent
 
async def main():
    # It's best practice to initialize the client once and reuse it.
    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    # Search tool to use for finding job postings and career information
    def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ):
        """Run a web search"""
        search_docs = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        return search_docs


    # Initialize LinkedIn MCP client
    mcp_client = MultiServerMCPClient({
        "linkedin_scraper": {
            "url": "http://127.0.0.1:8000/mcp",
            "transport": "streamable_http"
        }
    })


    # Get LinkedIn tools from MCP client
    linkedin_tools = await mcp_client.get_tools()
    
    # Extract tool names for subagents (they expect tool names, not tool objects)
    linkedin_tool_names = [tool.name for tool in linkedin_tools]

    sub_job_search_prompt = """You are a dedicated job search specialist. Your job is to find and analyze job opportunities based on the user's criteria.

    Conduct thorough job searches using the available tools and then reply to the user with detailed information about relevant job opportunities.

    You have access to multiple job search tools:
    - linkedin_scraper: For specific LinkedIn job searching with advanced filtering

    When searching for jobs, consider:
    - Job title and role requirements
    - Location preferences (remote, hybrid, on-site)
    - Experience level required
    - Company information and culture
    - Salary ranges and benefits
    - Required skills and technologies

    Use LinkedIn tools for more targeted, professional job searches and company research. Use internet search for broader job market analysis and additional job boards.

    Only your FINAL answer will be passed on to the user. They will have NO knowledge of anything except your final message, so your final job search results should be comprehensive and actionable!"""

    job_search_sub_agent = {
        "name": "job-search-agent", 
        "description": "Used to search for specific job opportunities. Give this agent clear job search criteria including role, location, experience level, and any specific requirements. Focus on one type of role at a time for best results.",
        "prompt": sub_job_search_prompt,
        "tools": linkedin_tool_names
    }

    sub_career_advisor_prompt = """You are a dedicated career advisor. Your job is to review and enhance job search strategies and results.

    You can find the job search results at `job_search_results.md`.

    You can find the user's job search criteria at `search_criteria.txt`.

    The user may ask for specific areas to improve their job search strategy. Respond with detailed advice on:

    You can use the search tool to find information about career trends, salary data, or industry insights that will help improve the job search.

    Do not write to the `job_search_results.md` yourself.

    Things to evaluate and advise on:
    - Check if the job results match the user's criteria and career goals
    - Assess if the user should expand or narrow their search parameters
    - Evaluate if the user's skills align with the job requirements found
    - Suggest improvements to the user's job search strategy
    - Recommend additional job boards, networking opportunities, or application approaches
    - Analyze market trends and salary expectations for the roles
    - Identify skill gaps that might need to be addressed
    - Suggest ways to make the candidate more competitive
    - Recommend timing strategies for applications
    """

    career_advisor_sub_agent = {
        "name": "career-advisor-agent",
        "description": "Used to provide career advice and improve job search strategies. Give this agent information about the user's career goals, current search results, and areas where they need guidance.",
        "prompt": sub_career_advisor_prompt,
        "tools": ["internet_search"]
    }


    # Prompt prefix to steer the agent to be an expert job search assistant
    job_search_instructions = """You are an expert job search assistant. Your job is to help users find relevant job opportunities and provide comprehensive career guidance.

    The first thing you should do is to write the user's job search criteria to `search_criteria.txt` so you have a clear record of their requirements.

    Use the job-search-agent to find specific job opportunities. It will respond with detailed information about relevant positions based on the criteria you provide.

    When you have enough job opportunities gathered, compile them into `job_search_results.md` with detailed information about each position.

    You can call the career-advisor-agent to get advice on the job search strategy, market analysis, and recommendations for improvement. After that (if needed) you can do more job searching and update the `job_search_results.md`
    You can do this however many times you want until you are satisfied with the comprehensiveness of the results.

    Only edit the file once at a time (if you call this tool in parallel, there may be conflicts).

    Here are instructions for compiling the final job search results:

    <job_search_results_instructions>

    CRITICAL: Make sure the results are written in the same language as the human messages! If you make a todo plan - you should note in the plan what language the results should be in so you dont forget!
    Note: the language the results should be in is the language the SEARCH CRITERIA is in, not the language/country that the job is located in.

    Please create a comprehensive job search results document that:
    1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
    2. Includes specific job details such as company name, position title, location, salary range, and requirements
    3. References job posting sources using [Company - Position](URL) format
    4. Provides thorough analysis of each opportunity, including company background, role responsibilities, and growth potential
    5. Includes application instructions and deadlines where available
    6. Includes a "Job Sources" section at the end with all referenced job posting links

    You can structure your job search results in a number of different ways. Here are some examples:

    For a general job search request, you might structure your results like this:
    1/ Executive Summary of Search Results
    2/ High Priority Opportunities
    3/ Additional Relevant Positions
    4/ Market Analysis and Trends
    5/ Application Strategy Recommendations

    For a request comparing different types of roles, you might structure it like this:
    1/ intro
    2/ Role Type A Opportunities
    3/ Role Type B Opportunities  
    4/ Comparison Analysis
    5/ Recommendations

    For a request focused on specific companies, you might structure it like this:
    1/ Company A Opportunities
    2/ Company B Opportunities
    3/ Company C Opportunities
    4/ Comparative Analysis

    For entry-level job searches, you might focus on:
    1/ Entry-Level Positions by Industry
    2/ Internship and Graduate Program Opportunities
    3/ Skills Development Recommendations
    4/ Application Tips for New Graduates

    For senior-level searches, you might focus on:
    1/ Executive and Senior Positions
    2/ Leadership Opportunities
    3/ Compensation Analysis
    4/ Strategic Career Movement Advice

    REMEMBER: Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!
    Make sure that your sections are cohesive, and make sense for the reader.

    For each section of the job search results, do the following:
    - Use simple, clear language that job seekers can easily understand
    - Use ## for section title (Markdown format) for each section of the results
    - Do NOT ever refer to yourself as the writer of the results. This should be a professional job search document without any self-referential language.
    - Do not explain what you are doing in the document. Just present the job opportunities and analysis without commentary.
    - Each section should provide comprehensive information about the jobs found. Include details like job responsibilities, requirements, company culture, salary ranges, and application processes.
    - Use bullet points to clearly organize job requirements, benefits, and key details, but also include paragraph descriptions for company backgrounds and role analysis.
    - Always include actionable next steps for each opportunity (how to apply, who to contact, deadlines)

    REMEMBER:
    The job search criteria may be in English, but you need to present the results in the right language when writing the final document.
    Make sure the final job search results are in the SAME language as the human messages in the message history.

    Format the results in clear markdown with proper structure and include job posting source references where appropriate.

    <Job Source Citation Rules>
    - Assign each unique job posting URL a single citation number in your text
    - End with ### Job Sources that lists each source with corresponding numbers
    - IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
    - Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
    - Example format:
    [1] Company Name - Position Title: Job Posting URL
    [2] Company Name - Position Title: Job Posting URL
    - Job source citations are extremely important for applicants. Make sure to include these, and pay attention to getting these right. Users will use these citations to access the actual job postings and apply.
    </Job Source Citation Rules>
    </job_search_results_instructions>

    You have access to tools for comprehensive job searching.

    ## `linkedin_scraper`

    Use this to search specifically for jobs on LinkedIn. This tool provides more targeted, professional job search results with LinkedIn's advanced filtering capabilities. You can specify query, location, company, and experience level parameters.

    """

    # Ensure OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Create the job search agent
    agent = create_deep_agent(
        tools = [internet_search] + linkedin_tools,
        instructions = job_search_instructions,
        subagents=[career_advisor_sub_agent, job_search_sub_agent],
        model="gpt-4o-mini",
    ).with_config({"recursion_limit": 1000})
    
    return agent

# Create the agent instance for LangGraph
agent = asyncio.run(main())

if __name__ == "__main__":
    # This will only run if the script is executed directly
    asyncio.run(main())