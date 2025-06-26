import asyncio
from fastmcp import Client

client = Client("http://127.0.0.1:8000/mcp")

async def call_tool(tool_name: str, args: dict):
    async with client:
        response = await client.call_tool(tool_name, args)
        print(response)

async def list_available_tools():
    """List available tools"""
    async with client:
        tools = await client.list_tools()
        print("Available tools:")
        print(tools)
        return tools


if __name__ == "__main__":
    print("Available tools:")
    print(asyncio.run(list_available_tools()))
    print("Getting database stats...")
    response = asyncio.run(call_tool("get_database_stats", {}))
    print(response)
    print("Getting trial by NCT ID...")
    response = asyncio.run(call_tool("get_trial_by_nct_id", {"nct_id": "NCT00024908"}))
    print(response)
    print("Getting trial by Condition...")
    response = asyncio.run(call_tool("search_trials_by_condition_graph", {"condition": "Asthma", "top_k": 2}))
    print(response)
    print("Getting trial by Intervention...")
    response = asyncio.run(call_tool("search_trials_by_intervention_graph", {"intervention": "Procaterol", "top_k": 2}))
    print(response)
