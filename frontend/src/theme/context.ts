import { createContext, useContext } from "react";

export type ThemePreference = "light" | "dark";

export interface ThemeContextValue {
  preference: ThemePreference;
  toggle: () => void;
  setPreference: (preference: ThemePreference) => void;
}

export const ThemeContext = createContext<ThemeContextValue | null>(null);

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
