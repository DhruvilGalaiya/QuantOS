/**
 * Make the owner sign-in settings for gateway/.env (never committed):
 *
 *   npm run hash-password -w gateway        # then type the password and press Enter (it is not shown)
 *
 * Prints OWNER_PASSWORD_HASH (scrypt) and a fresh random SESSION_SECRET.
 */
import { randomBytes } from "node:crypto";
import { createInterface } from "node:readline";
import { hashPassword } from "../src/auth.ts";

const PROMPT = "Owner password (12+ characters): ";
const rl = createInterface({ input: process.stdin, output: process.stderr, terminal: true });
// echo the prompt only, never the typed characters
(rl as unknown as { _writeToOutput: (s: string) => void })._writeToOutput = (s: string) => {
  if (s.includes(PROMPT)) process.stderr.write(PROMPT);
};
rl.question(PROMPT, async (pw) => {
  rl.close();
  process.stderr.write("\n");
  if (pw.length < 12) {
    console.error("Use at least 12 characters.");
    process.exit(1);
  }
  console.log(`OWNER_PASSWORD_HASH=${await hashPassword(pw)}`);
  console.log(`SESSION_SECRET=${randomBytes(32).toString("base64url")}`);
});
