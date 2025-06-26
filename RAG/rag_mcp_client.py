import asyncio
from fastmcp import Client

client = Client("http://127.0.0.1:8000/mcp")

async def call_tool(tool_name: str, args: dict):
    async with client:
        response = await client.call_tool(tool_name, args)
        print(response)


if __name__ == "__main__":
    asyncio.run(call_tool("get_trial_by_nct_id", {"nct_id": "NCT00024908"}))