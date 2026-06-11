import { useCallback, useEffect, useState, type ReactNode } from "react";

import { ThemeContext, type ThemePreference } from "./context";

const THEME_KEY = "taxbot.theme.v1";

function readPreference(): ThemePreference {
  const stored = localStorage.getItem(THEME_KEY);
  return stored === "light" || stored === "dark" ? stored : "system";
}

function systemIsDark(): boolean {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [preference, setPreferenceState] =
    useState<ThemePreference>(readPreference);
  const [systemDark, setSystemDark] = useState<boolean>(systemIsDark);

  const resolved: "light" | "dark" =
    preference === "system" ? (systemDark ? "dark" : "light") : preference;

  // Sync the resolved theme to the <html> class (external system).
  useEffect(() => {
    document.documentElement.classList.toggle("dark", resolved === "dark");
  }, [resolved]);

  // Subscribe to OS theme changes.
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setSystemDark(media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  const setPreference = useCallback((next: ThemePreference) => {
    localStorage.setItem(THEME_KEY, next);
    setPreferenceState(next);
  }, []);

  const cycle = useCallback(() => {
    const order: ThemePreference[] = ["light", "dark", "system"];
    setPreferenceState((current) => {
      const next = order[(order.indexOf(current) + 1) % order.length];
      localStorage.setItem(THEME_KEY, next);
      return next;
    });
  }, []);

  return (
    <ThemeContext.Provider value={{ preference, resolved, setPreference, cycle }}>
      {children}
    </ThemeContext.Provider>
  );
}
