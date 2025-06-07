import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_mcp_adapters.tools import load_mcp_tools

template = """
You are an AI agent designed to assist with the following tools:

{tools}

When you receive a user input, decide if you need to use one of these tools:

Available tools: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: your thought process about what to do next
Action: the tool to use, must be one of [{tool_names}]
Action Input: JSON formatted input for the tool
Observation: the result of the tool action
... (this Thought/Action/Action Input/Observation can repeat as needed)
Thought: I now know the final answer
Final Answer: your final answer to the user

{agent_scratchpad}
"""

sentiment_prompt = PromptTemplate(
    input_variables=["tool_names", "tools", "agent_scratchpad"],
    template=template,
)
server_params = StdioServerParameters(
    command='uv',
    args=['run', 'C:\\Users\\jerry\\Documents\\Cursor Projects\\Stock\\MCP\\server.py'],
    env=None
)

async def main():

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            llm = ChatOllama(model='llama3.2', temperature=0)
            agent = create_react_agent(llm=llm, tools=tools, prompt=sentiment_prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
            response = await agent_executor.ainvoke({"input": "What is the current price of TSLA?"})
            print(response["output"])

if __name__ == "__main__":
    asyncio.run(main())