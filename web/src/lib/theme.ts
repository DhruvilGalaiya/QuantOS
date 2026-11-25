import { create } from "zustand";

export type Theme = "dark" | "light";

function initial(): Theme {
  return typeof document !== "undefined" && !document.documentElement.classList.contains("dark") ? "light" : "dark";
}

export const useTheme = create<{ theme: Theme; toggle: () => void }>((set, get) => ({
  theme: initial(),
  toggle: () => {
    const next: Theme = get().theme === "dark" ? "light" : "dark";
    document.documentElement.classList.toggle("dark", next === "dark");
    try {
      localStorage.setItem("quantos-theme", next);
    } catch {
      /* private mode: theme just won't persist */
    }
    set({ theme: next });
  },
}));

/** Read a CSS custom property (chart libraries need concrete colours, not var()). */
export function cssVar(name: string, el: Element = document.documentElement): string {
  return getComputedStyle(el).getPropertyValue(name).trim();
}
