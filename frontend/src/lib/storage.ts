import type { Chat } from "./types";

const CHATS_KEY = "taxbot.chats.v1";

/** Read all chats from localStorage. Corrupt or missing data yields []. */
export function loadChats(): Chat[] {
  try {
    const raw = localStorage.getItem(CHATS_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isChat);
  } catch {
    return [];
  }
}

export function saveChats(chats: Chat[]): void {
  try {
    localStorage.setItem(CHATS_KEY, JSON.stringify(chats));
  } catch {
    // Quota exceeded or storage unavailable — chat continues in memory only.
  }
}

function isChat(value: unknown): value is Chat {
  if (typeof value !== "object" || value === null) return false;
  const chat = value as Record<string, unknown>;
  return (
    typeof chat.id === "string" &&
    typeof chat.title === "string" &&
    Array.isArray(chat.messages) &&
    typeof chat.createdAt === "number" &&
    typeof chat.updatedAt === "number"
  );
}
