import dotenv from "dotenv";
import { PromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableLambda } from "@langchain/core/runnables";
import { ChatAnthropic } from "@langchain/anthropic";
import { Tool } from "@langchain/core/tools";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { AzureOpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { createToolCallingAgent } from "langchain/agents";
import { AgentExecutor } from "langchain/agents";
dotenv.config();
export class LanchainIntro {
  private model: ChatAnthropic;

  constructor() {
    this.model = new ChatAnthropic({
      temperature: 0.5,
      model: "claude-3-haiku-20240307",
    });
  }

  async main(prompt: string) {
    const res = await this.model.invoke(prompt);
    console.log(res.content);
  }

  async propmtTempalteMultiChain(topic: string) {
    //memberikan prompt dengan variable topic
    const prompt = ChatPromptTemplate.fromTemplate(
      "berikan saya jokes tentang {topic}"
    );

    //membuat pipeline dan menjalankan chain nya, sesuai dengan topic yang diinginkan
    const chain = prompt.pipe(this.model).pipe(new StringOutputParser());

    // melanjutkan dengen koreksi, apakah joke tersebut lucu atau tidak?
    const analysisPrompt = ChatPromptTemplate.fromTemplate(
      "apakah joke ini lucu? tolong rangkum yang menurutmu lucu {joke}"
    );

    // menjalankan runnable, dalam artian menjalankan 2 chain dalam 1 topic yang sama dengan sebuah koreksi
    const composedChain = new RunnableLambda({
      func: async (input: { topic: string }) => {
        const result = await chain.invoke(input);
        console.log(result);
        return { joke: result };
      },
    })
      .pipe(analysisPrompt)
      .pipe(this.model)
      .pipe(new StringOutputParser());

    const result = await composedChain.invoke({ topic });
    console.log(result);
  }

  async promptAgent() {
    try {
      // Memuat dokumen dari URL LangSmith menggunakan CheerioWebBaseLoader
      const loader = new CheerioWebBaseLoader(
        process.env.URL_SCRAP_TARGET || ""
      );
      const docs = await loader.load();

      // Membagi dokumen menjadi beberapa chunk agar mudah diolah oleh AI
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });
      const documents = await splitter.splitDocuments(docs);

      // Membuat penyimpanan vektor (vector store) dengan embedding menggunakan Azure OpenAI
      const vectorStore = await MemoryVectorStore.fromDocuments(
        documents,
        new AzureOpenAIEmbeddings()
      );
      const retriever = vectorStore.asRetriever();

      // Membuat tool untuk melakukan pencarian informasi LangSmith
      const retrieverTool = tool(
        async ({ input }) => {
          const docs = await retriever.invoke(input);
          const resultDocs = docs.map((doc) => doc.pageContent).join("\n\n");
          console.log("Retriever Tool Output:", resultDocs);
          return resultDocs;
        },
        {
          name: "detik_berita_dortmund_vs_union_berlin",
          description:
            "Search for information about match Dortmun VS Union Berlin from detik.com. For any questions about total goals, date match play, you must use this tool!",
          schema: z.object({ input: z.string() }),
        }
      );

      // Menggabungkan tools (retriever) ke dalam agent
      const tools = [retrieverTool];

      // Menggunakan agent untuk mendapatkan jawaban atas pertanyaan

      // Membuat template prompt untuk interaksi dengan agent
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
      });

      const result = await agentExecutor.invoke({
        input:
          "Siapa saja sih yang memasukan bola ke dalam gawang, tolong sebutkan namanya?",
      });
      console.log(result);
    } catch (error) {
      console.error("Error in promptAgent:", error);
    }
  }
}
