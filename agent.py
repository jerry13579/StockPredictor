import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_ollama import ChatOllama

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

server_params = StdioServerParameters(
    command="python",
    args=["server.py"],
)
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            llm = ChatOllama(model='llama3.2', temperature=0)

            agent = create_react_agent(llm, tools)
            agent_response = await agent.ainvoke({"messages": "what's the sentiment of TSLA"})
            print(agent_response['messages'][3].content)


if __name__ == "__main__":
    asyncio.run(main())