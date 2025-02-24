// import { LanchainMain } from "./modules/langchainMain";
import { LanchainIntro } from "./modules/langchainIntro";
import cors from "cors";
// langchainIntro.propmtTempalteMultiChain("beruang");
// langchainIntro.main("Halo nama saya imam");
// const langchainMain = new LanchainMain();
// langchainMain.main();

import Express from "express";
import { Request, Response } from "express";

const app = Express();
app.use(cors());
app.use(Express.json());

app.post("/chat", async (req: Request, res: Response): Promise<void> => {
  try {
    const { message } = req.body;

    if (!message) {
      res.json({ message: "message not is required" });
    } else {
      const langchainIntro = new LanchainIntro();
      const response = await langchainIntro.langchainPDF(message);
      console.log(response);
      res.json({ reply: response });
    }
  } catch (error) {
    console.error("Chatbot Error:", error);
    res.status(500).json({ error: "Failed to get response from AI" });
  }
});

app.listen(3000, () => {
  console.log("running");
});
