import { describe, expect, it } from "vitest";
import { day, level, pct, pts, rupees, share } from "./format";

describe("format", () => {
  it("uses Indian digit grouping", () => {
    expect(level(1234567.891)).toBe("12,34,567.89");
    expect(rupees(2878.57)).toBe("₹2,879");
  });
  it("signs percents and points with a true minus", () => {
    expect(pct(-0.01747, 2, true)).toBe("−1.75%");
    expect(pct(0.0123, 1, true)).toBe("+1.2%");
    expect(pts(0.02426)).toBe("+2.4 pts");
    expect(share(0.809)).toBe("81%");
  });
  it("shows a dash for missing values", () => {
    expect(level(null)).toBe("—");
    expect(pct(Number.NaN)).toBe("—");
  });
  it("formats session dates in IST", () => {
    expect(day("2026-11-23")).toBe("Mon 23 Nov");
  });
});
