import { MongoClient, Db } from "mongodb";
import dotenv from "dotenv";

dotenv.config();

const client = new MongoClient(process.env.MONGODB_URI as string);

let db: Db;

const connectDB = async (): Promise<void> => {
  try {
    await client.connect();
    db = client.db(process.env.DB_NAME);
    console.log("✅ MongoDB Connected Successfully");
  } catch (error) {
    console.error("❌ MongoDB Connection Failed:", (error as Error).message);
    process.exit(1);
  }
};

const getDB = (): Db => {
  if (!db) {
    throw new Error("❌ Database not connected!");
  }
  return db;
};

export { connectDB, getDB };
