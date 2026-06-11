import {
  Landmark,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Trash2,
} from "lucide-react";

import type { ChatGroup } from "../hooks/useChats";

interface SidebarProps {
  groups: ChatGroup[];
  activeChatId: string | null;
  isDraft: boolean;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  onDeleteChat: (id: string) => void;
}

export function Sidebar({
  groups,
  activeChatId,
  isDraft,
  collapsed,
  onToggleCollapsed,
  onNewChat,
  onSelectChat,
  onDeleteChat,
}: SidebarProps) {
  return (
    <aside
      className={`sidebar-panel flex h-full shrink-0 flex-col overflow-hidden border-r border-border bg-surface transition-[width] duration-300 ease-in-out ${
        collapsed ? "w-[52px]" : "w-64"
      }`}
    >
      <div
        className={`flex shrink-0 items-center py-4 transition-all duration-300 ease-in-out ${
          collapsed ? "justify-center px-0" : "justify-between px-4"
        }`}
      >
        <div
          className={`flex items-center gap-2.5 overflow-hidden transition-all duration-300 ease-in-out ${
            collapsed ? "w-0 opacity-0" : "w-auto opacity-100"
          }`}
        >
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-accent-soft text-accent">
            <Landmark size={17} strokeWidth={1.75} />
          </span>
          <span className="whitespace-nowrap text-[18px] font-semibold tracking-[-0.02em] text-ink">
            TaxBot
          </span>
        </div>
        <button
          type="button"
          onClick={(event) => {
            onToggleCollapsed();
            event.currentTarget.blur();
          }}
          aria-label={collapsed ? "Open sidebar" : "Collapse sidebar"}
          className="sidebar-btn shrink-0 rounded-lg p-1.5 text-ink-faint transition-colors duration-200 hover:bg-surface-2 hover:text-ink"
        >
          {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={17} />}
        </button>
      </div>

      <div
        className={`shrink-0 px-3 pb-2 transition-all duration-300 ease-in-out ${
          collapsed ? "px-2" : ""
        }`}
      >
        <button
          type="button"
          onClick={onNewChat}
          aria-label="New chat"
          aria-current={isDraft ? "page" : undefined}
          title={collapsed ? "New chat" : undefined}
          className={`sidebar-btn relative flex w-full items-center rounded-lg text-sm transition-colors duration-200 ${
            collapsed ? "justify-center p-2" : "gap-2 px-3 py-2"
          } ${
            isDraft
              ? "bg-surface-2 font-medium text-ink"
              : "font-medium text-ink hover:bg-surface-2"
          }`}
        >
          {isDraft && !collapsed && (
            <span
              aria-hidden
              className="absolute left-0 top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-full bg-accent"
            />
          )}
          <Plus size={16} className="shrink-0 text-ink-muted" />
          {!collapsed && <span className="whitespace-nowrap">New chat</span>}
        </button>
      </div>

      <nav
        className={`min-h-0 flex-1 overflow-y-auto overflow-x-hidden pb-2 transition-opacity duration-300 ${
          collapsed ? "pointer-events-none opacity-0" : "opacity-100"
        }`}
        aria-label="Chat history"
        aria-hidden={collapsed}
      >
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
                        className={`relative flex w-full items-center rounded-lg py-2 pl-3 pr-8 text-left text-[13.5px] transition-all duration-200 ${
                          active
                            ? "bg-surface-2 font-medium text-ink"
                            : "text-ink-muted hover:bg-surface-2 hover:text-ink"
                        }`}
                      >
                        {active && (
                          <span
                            aria-hidden
                            className="absolute left-0 top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-full bg-accent transition-all duration-200"
                          />
                        )}
                        <span className="truncate">{chat.title}</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => onDeleteChat(chat.id)}
                        aria-label={`Delete chat: ${chat.title}`}
                        className="absolute right-1.5 top-1/2 hidden -translate-y-1/2 rounded-md p-1 text-ink-faint transition-colors duration-200 hover:text-danger group-hover:block"
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
    </aside>
  );
}
