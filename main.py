"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the project instructions and rubric for guidance.
Work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
import botocore.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
# Create a BedrockAgentCoreApp instance.
# This registers the ASGI server for AgentCore deployment.
# There must be exactly one instance per deployment.

app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Replace the placeholder strings with your actual AWS resource values.
# You collected these in the infrastructure setup section of the project instructions.

GATEWAY_URL = "https://customersupportgateway-fymre0zzv5.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "LJO0JPSJVN"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-DS1O0lDYI0"


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
# Create:
#   1. A BedrockModel using model_id "global.amazon.nova-2-lite-v1:0"
#   2. A MemoryClient with region_name=REGION
#   3. A boto3 client for the "bedrock-agent-runtime" service in REGION

model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
# Implement get_namespaces() to return a dict mapping strategy type to
# namespace template string.

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict[str, str]:
    """Return a dict mapping strategy type → namespace template string."""
    try:
        strategies_response = mem_client.get_memory_strategies(memory_id=memory_id)
        namespaces = {}
        for strategy in strategies_response:
            strat_type = strategy.get("type", "UNKNOWN")
            templates = strategy.get("namespaceTemplates")
            if templates and len(templates) > 0:
                namespaces[strat_type] = templates[0]
            else:
                fallback = strategy.get("namespaces")
                if fallback and len(fallback) > 0:
                    namespaces[strat_type] = fallback[0]
        return namespaces
    except Exception as e:
        logger.warning(f"Error fetching memory strategies: {e}")
        return {
            "SEMANTIC": "/summaries/{actorId}/{sessionId}/",
            "USER_PREFERENCE": "/preferences/{actorId}/"
        }


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
# Implement MemoryHook, a HookProvider subclass that adds long-term memory.

class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        if not event.agent.messages:
            return

        last_message = event.agent.messages[-1]
        if last_message.get("role") != "user":
            return

        content = last_message.get("content")
        user_query = ""

        if isinstance(content, str):
            user_query = content
        elif isinstance(content, list):
            # Guard against tool result messages
            for block in content:
                if isinstance(block, dict) and "toolResult" in block:
                    return
                if isinstance(block, dict) and "text" in block:
                    user_query += block["text"] + " "
            user_query = user_query.strip()

        if not user_query:
            return

        retrieved_texts = []
        for strat_type, template in self.namespaces.items():
            namespace = template.format(actorId=self.actor_id, sessionId=self.session_id)
            try:
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=user_query,
                    top_k=5
                )
                for mem in memories:
                    mem_content = mem.get("content", {})
                    text = mem_content.get("text") or mem_content.get("statement") or str(mem_content)
                    if text:
                        retrieved_texts.append(f"[{strat_type}] {text}")
            except Exception as e:
                logger.warning(f"Error retrieving memory from {namespace}: {e}")

        if retrieved_texts:
            context_block = "Customer Context:\n" + "\n".join(retrieved_texts) + "\n\n"
            if isinstance(content, str):
                last_message["content"] = context_block + content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        block["text"] = context_block + block["text"]
                        break

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = getattr(event.agent, "messages", [])
        if not messages:
            return

        last_user_query = None
        last_assistant_response = None

        for msg in reversed(messages):
            role = msg.get("role")
            content = msg.get("content")

            # Extract user query
            if role == "user" and last_user_query is None:
                if isinstance(content, str):
                    last_user_query = content
                elif isinstance(content, list):
                    is_tool_result = any(isinstance(b, dict) and "toolResult" in b for b in content)
                    if not is_tool_result:
                        text_parts = [b["text"] for b in content if isinstance(b, dict) and "text" in b]
                        if text_parts:
                            last_user_query = " ".join(text_parts)

            # Extract assistant response
            if role == "assistant" and last_assistant_response is None:
                if isinstance(content, str):
                    last_assistant_response = content
                elif isinstance(content, list):
                    text_parts = [b["text"] for b in content if isinstance(b, dict) and "text" in b]
                    if text_parts:
                        last_assistant_response = " ".join(text_parts)

            if last_user_query and last_assistant_response:
                break

        if last_user_query and last_assistant_response:
            try:
                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[
                        (last_user_query, "USER"),
                        (last_assistant_response, "ASSISTANT"),
                    ],
                )
            except Exception as e:
                logger.warning(f"Error saving interaction to memory: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
# Implement search_knowledge_base(query) using the @tool decorator.

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."

    try:
        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query}
        )
        results = response.get("retrievalResults", [])
        if not results:
            return "No relevant information found in the knowledge base."

        chunks = [r["content"]["text"] for r in results if "content" in r and "text" in r["content"]]
        return "\n---\n".join(chunks)
    except Exception as e:
        return f"Error querying Knowledge Base: {str(e)}"


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
# Implement calculate_loyalty_discount() using the @tool decorator.

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json

loyalty_points = {loyalty_points}
tier = "{tier}"
order_total = {order_total}
product_category = "{product_category.lower()}"

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

# 100 points = $1.00 USD value
# Max redemption cap: 50% of the order value
max_point_discount = order_total * 0.50
max_usable_points = int(max_point_discount * 100)

usable_points = min(loyalty_points, max_usable_points)
points_redeemed = (usable_points // 500) * 500
point_discount = points_redeemed / 100.0

subtotal_after_points = order_total - point_discount
tier_rate = tier_rates.get(tier, 0.0)
tier_discount = round(subtotal_after_points * tier_rate, 2)

final_total = round(subtotal_after_points - tier_discount, 2)
total_savings = round(point_discount + tier_discount, 2)

rate = earn_rates.get(product_category, 1)
points_earned = int(final_total * rate)
remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "order_total": order_total,
    "points_redeemed": points_redeemed,
    "point_discount": point_discount,
    "tier": tier,
    "tier_discount": tier_discount,
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points
}}
print(json.dumps(result))
"""
    try:
        session = code_session(REGION)
        events = session.invoke(
            "executeCode",
            {"code": code, "language": "python", "clearContext": True}
        )
        for event in events:
            if "result" in event:
                return json.dumps(event["result"])
            if "stdout" in event:
                return event["stdout"].strip()
        return str(events)
    except Exception as e:
        logger.warning(f"Code Interpreter execution failed, applying fallback: {e}")
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_rate = tier_rates.get(tier, 0.0)
        tier_discount = round(order_total * tier_rate, 2)
        final_total = round(order_total - tier_discount, 2)
        fallback = {
            "order_total": order_total,
            "points_redeemed": 0,
            "point_discount": 0.0,
            "tier": tier,
            "tier_discount": tier_discount,
            "final_total": final_total,
            "total_savings": tier_discount,
            "note": "Fallback calculation: Code Interpreter unavailable."
        }
        return json.dumps(fallback)


def get_sigv4_headers(url: str, region: str) -> dict:
    """Generate AWS SigV4 authorization headers for AgentCore MCP Gateway."""
    session = boto3.Session()
    creds = session.get_credentials().get_frozen_credentials()
    request = AWSRequest(method="POST", url=url, data="")
    SigV4Auth(creds, "bedrock-agentcore", region).add_auth(request)
    return dict(request.headers)

# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
# Implement the invoke() function decorated with @app.entrypoint.

@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.
    """
    try:
        if isinstance(payload, str):
            payload = json.loads(payload)

        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "anonymous_customer")
        session_id = payload.get("session_id", str(uuid.uuid4()))

        # Instantiate Memory Hook and Browser tool
        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID
        )
        agent_core_browser = AgentCoreBrowser(region=REGION)

        # Baseline internal tools
        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser
        ]

        system_prompt = (
            "You are a helpful and professional customer support assistant for our e-commerce platform. "
            "You have access to order tracking tools, return/refund operations, product catalog knowledge base, "
            "loyalty discount calculator, and long-term customer context. "
            "Always prioritize accurate information, polite guidance, and concise problem resolution."
        )

        # Connect to MCP Gateway with SigV4 headers
        try:
            auth_headers = get_sigv4_headers(GATEWAY_URL, REGION)
            async with streamable_http_client(GATEWAY_URL, headers=auth_headers) as (read_stream, write_stream):
                async with MCPClient(read_stream, write_stream) as mcp_client:
                    gateway_tools = await mcp_client.list_tools()
                    tools.extend(gateway_tools)

                    agent = Agent(
                        model=model,
                        tools=tools,
                        hooks=[memory_hook],
                        system_prompt=system_prompt
                    )
                    response = agent(user_input)
        except Exception as mcp_err:
            logger.warning(f"MCP Gateway connection failed: {mcp_err}. Falling back to baseline tools.")
            # Fallback without crashing if MCP gateway is unreachable
            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=system_prompt
            )
            response = agent(user_input)

        # Extract response text
        if hasattr(response, "messages") and response.messages:
            last_msg = response.messages[-1]
            content = last_msg.get("content")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        return block["text"]

        return str(response)

    except Exception as e:
        logger.error(f"Execution error in agent invoke: {e}", exc_info=True)
        return f"An error occurred while processing your request: {str(e)}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()
