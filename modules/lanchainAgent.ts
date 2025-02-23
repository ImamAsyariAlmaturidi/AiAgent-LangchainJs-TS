import dotenv from "dotenv";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { tool } from "@langchain/core/tools";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createToolCallingAgent } from "langchain/agents";
import { AgentExecutor } from "langchain/agents";
import { z } from "zod";
import { AzureOpenAIEmbeddings } from "@langchain/openai";
import { connectDB, getDB } from "../mongodb/mongo";

dotenv.config();

export class LangchainAgent {
  private model: ChatAnthropic;
  private embeddings: AzureOpenAIEmbeddings;

  constructor() {
    this.model = new ChatAnthropic({
      temperature: 0.5,
      model: "claude-3-haiku-20240307",
      streaming: true,
      callbacks: [
        {
          handleLLMNewToken(token) {
            process.stdout.write(token);
          },
        },
      ],
    });

    this.embeddings = new AzureOpenAIEmbeddings();
  }

  async main() {
    await connectDB();
    const db = getDB();
    const collection = db.collection("embeddings");

    // 🟢 **Cek apakah database sudah ada embeddings**
    const existingEmbeddings = await collection.find({}).toArray();

    if (existingEmbeddings.length === 0) {
      console.log(
        "❌ No embeddings found in database. Run the embedding process first."
      );
      return;
    }

    console.log(`✅ Total embeddings found: ${existingEmbeddings.length}`);

    // 🟢 **Buat tool untuk pencarian berbasis vektor langsung dari MongoDB**
    const retrieverTool = tool(
      async ({ input }) => {
        console.log(`🔎 Searching for: ${input}`);

        // Ambil semua data (karena kita tidak menggunakan vektor untuk pencocokan di MongoDB)
        const docs = await collection.find({}).toArray();

        console.log(`📄 Found documents: ${docs.length}`);

        // Ambil metadata dari hasil pencarian
        return docs
          .map((doc) => {
            return `🛒 Produk: ${doc.metadata.slug}
                💲 Harga: ${doc.metadata.price}
                🏷️ Tags: ${doc.metadata.tags.join(", ")}
                📷 Gambar: ${doc.metadata.thumbnail}`;
          })
          .join("\n\n");
      },
      {
        name: "get_all_products",
        description:
          "Search for product information stored in MongoDB. Questions about product name, price, and tags must use this tool!",
        schema: z.object({ input: z.string() }),
      }
    );

    // 🟢 **Buat agent & jalankan**
    const tools = [retrieverTool];

    const prompt = ChatPromptTemplate.fromMessages([
      ["system", "You are a helpful assistant"],
      ["placeholder", "{chat_history}"],
      ["human", "{input}"],
      ["placeholder", "{agent_scratchpad}"],
    ]);

    const agent = createToolCallingAgent({ llm: this.model, tools, prompt });

    const agentExecutor = new AgentExecutor({
      agent,
      tools,
      verbose: false,
    });

    const result = await agentExecutor.invoke({
      input: "berikan 10 product yang anda punya",
    });

    console.log(result);
  }
}
