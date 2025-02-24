import dotenv from "dotenv";
import { PromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";
import {
  JsonOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";
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
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import {
  callbackHandlerPrefersStreaming,
  HandleLLMNewTokenCallbackFields,
} from "@langchain/core/dist/callbacks/base";
dotenv.config();

interface Aset {
  kas: string;
  penempatan_pada_bank_indonesia: string;
  penempatan_pada_bank_lain: string;
}
interface Liabilitas {
  Giro: string;
  Tabungan: string;
  Deposito: string;
}
interface Report {
  explanation: string;
  liablitas: Liabilitas;
  aset: Aset;
}
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
          // console.log("Retriever Tool Output:", resultDocs);
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
        verbose: false,
      });

      const result = await agentExecutor.invoke({
        input:
          "Siapa saja sih yang memasukan bola ke dalam gawang, tolong sebutkan namanya?",
      });
    } catch (error) {
      console.error("Error in promptAgent:", error);
    }
  }

  async langchainPDF(message: string) {
    if (!message || typeof message !== "string") {
      throw new Error("Invalid message: must be a non-empty string");
    }

    try {
      const formatInstructions = `Extract key financial data from multiple PDF reports and return an array of structured JSON objects.

### **Extraction Guidelines:**
- Each **PDF report should be a separate object** in the JSON array.
- Dynamically extract **key financial categories**, including:  
  - **Aset (Assets)**
  - **Liabilitas (Liabilities)**
  - **Ekuitas (Equity)**
- Use **extracted section headers as JSON keys**.
- **Prefix all numerical values with "RP."**  
- Provide a **brief summary** of the report in the "explanation" field.  
- **If a category is missing, omit it instead of returning null.**  

---

### **Expected Output Format:**
json
[
  {
    "explanation": "<Extracted Explanation>",
    "Aset": {
      "<Extracted Aset Field>": "RP. <value>",
      "<Extracted Aset Field>": "RP. <value>"
    },
    "Liabilitas": {
      "<Extracted Liabilitas Field>": "RP. <value>",
      "<Extracted Liabilitas Field>": "RP. <value>"
    },
    "Ekuitas": {
      "<Extracted Ekuitas Field>": "RP. <value>",
      "<Extracted Ekuitas Field>": "RP. <value>"
    }
  }
]`;

      const parser = new JsonOutputParser<Report>();
      const files = ["document_loaders/example_data/9.pdf"];

      const embeddings = new AzureOpenAIEmbeddings();
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 400,
      });

      // ✅ Load semua PDF sekaligus
      const loaders = files.map((file) =>
        new PDFLoader(file, { splitPages: false }).load()
      );
      const docsPDFArray = await Promise.all(loaders);
      const allDocs = docsPDFArray.flat();

      // ✅ Split dokumen jadi potongan kecil
      const splitDocsPDF = await splitter.splitDocuments(allDocs);

      // ✅ Simpan embedding ke dalam Vector Store
      const vectorStore = await MemoryVectorStore.fromDocuments(
        splitDocsPDF,
        embeddings
      );

      // ✅ Buat retriever tool
      const retrieverTool = tool(
        async ({ input }) => {
          const docs = await vectorStore.asRetriever().invoke(input);
          return docs.map((doc) => doc.pageContent).join("\n\n");
        },
        {
          name: "all_bca_report_finance",
          description:
            "Retrieve the official Bank Central Asia (BCA) financial report. This tool should only be used when the prompt explicitly requests BCA's financial data.",
          schema: z.object({
            input: z
              .string()
              .describe(
                "Specific query about BCA's financial report, e.g., 'BCA Januari 2025 report'"
              ),
          }),
        }
      );

      // ✅ Siapkan prompt dengan format_instructions
      const prompt = ChatPromptTemplate.fromMessages([
        ["system", "You are a helpful assistant. {format_instructions}"],
        ["placeholder", "{chat_history}"],
        ["human", "{input}"],
        ["placeholder", "{agent_scratchpad}"],
      ]);

      // ✅ Inject format instructions ke dalam prompt
      const formattedPrompt = await prompt.partial({
        chat_history: "",
        format_instructions: formatInstructions,
      });

      // ✅ Pastikan `this.model` sudah diinisialisasi
      if (!this.model) {
        throw new Error("Model is not initialized");
      }

      // ✅ Siapkan agent
      const agent = createToolCallingAgent({
        llm: this.model,
        tools: [retrieverTool],
        prompt: formattedPrompt,
      });

      // ✅ Jalankan AgentExecutor
      const agentExecutor = new AgentExecutor({
        agent,
        tools: [retrieverTool],
        verbose: false,
      });
      const result = await agentExecutor.invoke({
        input: message,
        chat_history: "",
      });
      const text = result?.output?.[0]?.text;
      let extractedJson = null;

      if (text) {
        try {
          // Coba langsung parse JSON dari teks
          extractedJson = JSON.parse(
            text.slice(text.indexOf("["), text.lastIndexOf("]") + 1)
          );
        } catch (error) {
          console.error("Error parsing JSON:", error);
        }
      }

      console.log(extractedJson, "Extracted JSON");

      if (extractedJson) {
        return extractedJson;
      }
    } catch (error) {
      console.error("LangchainPDF Error:", error);
      throw new Error("Failed to process the PDF data.");
    }
  }
}
