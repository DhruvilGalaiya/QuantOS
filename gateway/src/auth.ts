import { createHmac, randomBytes, scrypt as scryptCb, timingSafeEqual } from "node:crypto";
import { promisify } from "node:util";

const scrypt = promisify(scryptCb) as (pw: string, salt: Buffer, len: number, opts: object) => Promise<Buffer>;

/**
 * Owner sign-in without a user database: one password, stored only as a scrypt hash in the environment
 * (OWNER_PASSWORD_HASH), and a session cookie signed with SESSION_SECRET (HMAC-SHA256). Nothing secret is committed.
 */
export const COOKIE = "quantos_owner";

const N = 16384;
const R = 8;
const P = 1;
const LEN = 32;

export async function hashPassword(password: string): Promise<string> {
  const salt = randomBytes(16);
  const hash = await scrypt(password, salt, LEN, { N, r: R, p: P });
  return `scrypt$${N}$${R}$${P}$${salt.toString("base64url")}$${hash.toString("base64url")}`;
}

export async function verifyPassword(password: string, stored: string): Promise<boolean> {
  const [kind, n, r, p, salt, hash] = stored.split("$");
  if (kind !== "scrypt" || !salt || !hash) return false;
  const want = Buffer.from(hash, "base64url");
  const got = await scrypt(password, Buffer.from(salt, "base64url"), want.length, {
    N: Number(n),
    r: Number(r),
    p: Number(p),
    maxmem: 64 * 1024 * 1024,
  });
  return got.length === want.length && timingSafeEqual(got, want);
}

const sign = (secret: string, payload: string) => createHmac("sha256", secret).update(`owner.${payload}`).digest("base64url");

/** "<expiry ms>.<signature>" */
export function issueSession(secret: string, ttlMs: number, now = Date.now()): string {
  const exp = String(now + ttlMs);
  return `${exp}.${sign(secret, exp)}`;
}

export function verifySession(secret: string, token: string | undefined, now = Date.now()): boolean {
  if (!token) return false;
  const [exp, sig] = token.split(".");
  if (!exp || !sig || !/^\d+$/.test(exp) || Number(exp) <= now) return false;
  const want = Buffer.from(sign(secret, exp));
  const got = Buffer.from(sig);
  return got.length === want.length && timingSafeEqual(got, want);
}

export function readCookie(header: string | undefined, name: string): string | undefined {
  for (const part of (header ?? "").split(";")) {
    const i = part.indexOf("=");
    if (i > 0 && part.slice(0, i).trim() === name) return decodeURIComponent(part.slice(i + 1).trim());
  }
  return undefined;
}

export function sessionCookie(value: string, maxAgeS: number, secure: boolean): string {
  return [
    `${COOKIE}=${encodeURIComponent(value)}`,
    "Path=/api",
    "HttpOnly",
    "SameSite=Strict",
    `Max-Age=${maxAgeS}`,
    ...(secure ? ["Secure"] : []),
  ].join("; ");
}
