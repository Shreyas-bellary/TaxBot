import { useCallback, useEffect, useState, type ReactNode } from "react";

import { ThemeContext, type ThemePreference } from "./context";

const THEME_KEY = "taxbot.theme.v1";

function readPreference(): ThemePreference {
  const stored = localStorage.getItem(THEME_KEY);
  return stored === "light" || stored === "dark" ? stored : "dark";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [preference, setPreferenceState] = useState<ThemePreference>(readPreference);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", preference === "dark");
  }, [preference]);

  const setPreference = useCallback((next: ThemePreference) => {
    localStorage.setItem(THEME_KEY, next);
    setPreferenceState(next);
  }, []);

  const toggle = useCallback(() => {
    setPreferenceState((current) => {
      const next: ThemePreference = current === "dark" ? "light" : "dark";
      localStorage.setItem(THEME_KEY, next);
      return next;
    });
  }, []);

  return (
    <ThemeContext.Provider value={{ preference, setPreference, toggle }}>
      {children}
    </ThemeContext.Provider>
  );
}
