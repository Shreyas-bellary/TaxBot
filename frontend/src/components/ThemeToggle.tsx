import { Moon, Sun } from "lucide-react";

import { useTheme } from "../theme/context";

export function ThemeToggle() {
  const { preference, toggle } = useTheme();
  const isDark = preference === "dark";

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      className="theme-toggle group relative flex h-9 w-9 items-center justify-center rounded-full text-ink-muted transition-colors duration-300 hover:bg-surface-2 hover:text-ink"
    >
      <Sun
        size={17}
        className={`absolute transition-all duration-300 ease-out ${
          isDark
            ? "rotate-90 scale-0 opacity-0"
            : "rotate-0 scale-100 opacity-100"
        }`}
      />
      <Moon
        size={17}
        className={`absolute transition-all duration-300 ease-out ${
          isDark
            ? "rotate-0 scale-100 opacity-100"
            : "-rotate-90 scale-0 opacity-0"
        }`}
      />
    </button>
  );
}
