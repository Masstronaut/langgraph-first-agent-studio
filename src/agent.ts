import { ChatOpenAI, wrapOpenAIClientError } from "@langchain/openai";
import { AIMessage, BaseMessageLike } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { MemorySaver } from "@langchain/langgraph";
// read the environment variables from .env
//import "dotenv/config";

const tools = [new TavilySearchResults({ maxResults: 3 })];
// Create a model and give it access to the tools
const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
}).bindTools(tools);

// Define the function that calls the model
async function callModel(state: typeof MessagesAnnotation.State) {
  const messages = state.messages;

  const response = await model.invoke(messages);

  return { messages: response };
}

function shouldUseTool(state: typeof MessagesAnnotation.State) {
  const lastMessage = state.messages[state.messages.length - 1];

  // If the LLM makes a tool call, then we route to the "tools" node
  if (lastMessage.additional_kwargs.tool_calls) {
    return "tools";
  }
  // Otherwise, we stop (reply to the user) using the special "__end__" node
  return "__end__";
}

// Define the graph and compile it into a runnable
const graphBuilder = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge("__start__", "agent")
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("agent", shouldUseTool)
  .addEdge("tools", "agent");

export const graph = graphBuilder.compile({
  // checkpointer: new MemorySaver(),
  // interruptBefore: ["tools"],
});

/*
// Create a command line interface to interact with the chat bot

// We'll use these helpers to read from the standard input in the command line
import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

const lineReader = readline.createInterface({ input, output });

console.log("Type 'exit' or 'quit' to quit");

while (true) {
  const answer = await lineReader.question("User: ");
  if (["exit", "quit", "q"].includes(answer.toLowerCase())) {
    console.log("Goodbye!");
    lineReader.close();
    break;
  }

  // Run the chatbot and add its response to the conversation history
  const output = await graph.invoke(
    {
      messages: [{ content: answer, type: "user" }],
    },
    { configurable: { thread_id: "42" } },
  );

  // Check if the AI is trying to use a tool
  const lastMessage = output.messages[output.messages.length - 1];
  if (!lastMessage.tool_calls) {
    console.log("Agent: ", output.messages[output.messages.length - 1]);
    continue;
  }

  console.log(
    "Agent: I would like to make the following tool calls: ",
    lastMessage.tool_calls,
  );

  // Let the human decide whether to continue or not
  const humanFeedback = await lineReader.question(
    "Type 'y' to continue, or anything else to exit: ",
  );
  if (humanFeedback.toLowerCase() !== "y") {
    console.log("Goodbye!");
    lineReader.close();
    break;
  }

  // No new state is needed for the agent to use the tool, so pass `null`
  const outputWithTool = await graph.invoke(null, {
    configurable: { thread_id: "42" },
  });
  console.log(
    "Agent: ",
    outputWithTool.messages[outputWithTool.messages.length - 1].content,
  );
}
*/
