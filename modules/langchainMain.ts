import dotenv from "dotenv";
import { ChatOpenAI } from "@langchain/openai";

dotenv.config();

export class LanchainMain {
  private model: ChatOpenAI;

  constructor() {
    this.model = new ChatOpenAI({
      temperature: 0.5,
      modelName: "gpt-3.5-turbo",
    });
  }

  async main() {
    //main function code
    console.log("Learn Langchain...!");
  }

  async sampleFunction() {
    //sub function code
  }
}
