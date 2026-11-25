import { describe, expect, it } from "vitest";
import { impliedCone } from "./cone";

const imp = { expiry: "2026-12-01", atm_iv: 0.17, move_points: 575, move_pct: 0.022, spot: 25716, lower: 25141, upper: 26291, snapshot_ts: "2026-11-25T13:46:49+05:30" };

describe("impliedCone", () => {
  it("opens from spot and reaches ±move at expiry, skipping weekends", () => {
    const c = impliedCone(imp, "2026-11-25")!;
    expect(c.upper[0]).toEqual({ time: "2026-11-25", value: 25716 });
    expect(c.upper.at(-1)).toEqual({ time: "2026-12-01", value: 25716 + 575 });
    expect(c.lower.at(-1)?.value).toBe(25716 - 575);
    expect(c.upper.map((p) => p.time)).toEqual(["2026-11-25", "2026-11-26", "2026-11-27", "2026-11-30", "2026-12-01"]);
  });
  it("is empty when expiry is not after the start", () => {
    expect(impliedCone({ ...imp, expiry: "2026-11-20" }, "2026-11-25")).toBeNull();
  });
});
