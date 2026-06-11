import {
  Landmark,
  Monitor,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Sun,
  Trash2,
} from "lucide-react";

import type { ChatGroup } from "../hooks/useChats";
import { useTheme, type ThemePreference } from "../theme/context";

interface SidebarProps {
  groups: ChatGroup[];
  activeChatId: string | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  onDeleteChat: (id: string) => void;
}

const THEME_ICONS: Record<ThemePreference, typeof Sun> = {
  light: Sun,
  dark: Moon,
  system: Monitor,
};

const THEME_LABELS: Record<ThemePreference, string> = {
  light: "Light theme",
  dark: "Dark theme",
  system: "System theme",
};

export function Sidebar({
  groups,
  activeChatId,
  collapsed,
  onToggleCollapsed,
  onNewChat,
  onSelectChat,
  onDeleteChat,
}: SidebarProps) {
  const { preference, cycle } = useTheme();
  const ThemeIcon = THEME_ICONS[preference];

  if (collapsed) {
    return (
      <aside className="flex h-full w-12 shrink-0 flex-col items-center gap-2 border-r border-border bg-surface py-3">
        <button
          type="button"
          onClick={onToggleCollapsed}
          aria-label="Open sidebar"
          className="rounded-lg p-2 text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
        >
          <PanelLeftOpen size={18} />
        </button>
        <button
          type="button"
          onClick={onNewChat}
          aria-label="New chat"
          className="rounded-lg p-2 text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
        >
          <Plus size={18} />
        </button>
      </aside>
    );
  }

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-border bg-surface">
      <div className="flex items-center justify-between px-4 py-3.5">
        <div className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-soft text-accent">
            <Landmark size={15} />
          </span>
          <span className="text-[15px] font-semibold tracking-tight">TaxBot</span>
        </div>
        <button
          type="button"
          onClick={onToggleCollapsed}
          aria-label="Collapse sidebar"
          className="rounded-lg p-1.5 text-ink-faint transition-colors hover:bg-surface-2 hover:text-ink"
        >
          <PanelLeftClose size={17} />
        </button>
      </div>

      <div className="px-3 pb-2">
        <button
          type="button"
          onClick={onNewChat}
          className="flex w-full items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm font-medium text-ink transition-colors hover:border-border-strong hover:bg-surface-2"
        >
          <Plus size={16} className="text-ink-muted" />
          New chat
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto px-2 pb-2" aria-label="Chat history">
        {groups.length === 0 ? (
          <p className="px-3 pt-6 text-center text-[13px] leading-relaxed text-ink-faint">
            No conversations yet.
            <br />
            Your chats stay in this browser only.
          </p>
        ) : (
          groups.map((group) => (
            <section key={group.label} className="mt-3">
              <h2 className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-wider text-ink-faint">
                {group.label}
              </h2>
              <ul>
                {group.chats.map((chat) => {
                  const active = chat.id === activeChatId;
                  return (
                    <li key={chat.id} className="group relative">
                      <button
                        type="button"
                        onClick={() => onSelectChat(chat.id)}
                        className={`relative flex w-full items-center rounded-lg py-2 pl-3 pr-8 text-left text-[13.5px] transition-colors ${
                          active
                            ? "bg-surface-2 font-medium text-ink"
                            : "text-ink-muted hover:bg-surface-2 hover:text-ink"
                        }`}
                      >
                        {active && (
                          <span
                            aria-hidden
                            className="absolute left-0 top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-full bg-accent"
                          />
                        )}
                        <span className="truncate">{chat.title}</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => onDeleteChat(chat.id)}
                        aria-label={`Delete chat: ${chat.title}`}
                        className="absolute right-1.5 top-1/2 hidden -translate-y-1/2 rounded-md p-1 text-ink-faint transition-colors hover:text-danger group-hover:block"
                      >
                        <Trash2 size={14} />
                      </button>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))
        )}
      </nav>

      <div className="border-t border-border px-3 py-2.5">
        <button
          type="button"
          onClick={cycle}
          className="flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-[13px] text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
        >
          <ThemeIcon size={15} />
          {THEME_LABELS[preference]}
        </button>
        <p className="mt-1.5 px-2.5 text-[11px] leading-snug text-ink-faint">
          History is stored locally — nothing is saved on our servers.
        </p>
      </div>
    </aside>
  );
}
