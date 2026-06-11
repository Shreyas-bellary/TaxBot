import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { loadChats, saveChats } from "../lib/storage";
import type { Chat, Message } from "../lib/types";

const TITLE_MAX_LENGTH = 40;

export function titleFromQuery(query: string): string {
  const flattened = query.replace(/\s+/g, " ").trim();
  if (flattened.length <= TITLE_MAX_LENGTH) return flattened || "New chat";
  return `${flattened.slice(0, TITLE_MAX_LENGTH).trimEnd()}…`;
}

export type ChatGroupLabel = "Today" | "Yesterday" | "Older";

export interface ChatGroup {
  label: ChatGroupLabel;
  chats: Chat[];
}

function startOfDay(timestamp: number): number {
  const date = new Date(timestamp);
  date.setHours(0, 0, 0, 0);
  return date.getTime();
}

export function groupChatsByDate(chats: Chat[], now = Date.now()): ChatGroup[] {
  const today = startOfDay(now);
  const yesterday = today - 86_400_000;
  const sorted = [...chats].sort((a, b) => b.updatedAt - a.updatedAt);

  const buckets: Record<ChatGroupLabel, Chat[]> = {
    Today: [],
    Yesterday: [],
    Older: [],
  };
  for (const chat of sorted) {
    const day = startOfDay(chat.updatedAt);
    if (day >= today) buckets.Today.push(chat);
    else if (day >= yesterday) buckets.Yesterday.push(chat);
    else buckets.Older.push(chat);
  }

  return (Object.keys(buckets) as ChatGroupLabel[])
    .map((label) => ({ label, chats: buckets[label] }))
    .filter((group) => group.chats.length > 0);
}

export interface ChatsStore {
  chats: Chat[];
  groups: ChatGroup[];
  activeChatId: string | null;
  activeChat: Chat | null;
  /** Start a fresh draft conversation (chat is created on first message). */
  newChat: () => void;
  selectChat: (id: string) => void;
  deleteChat: (id: string) => void;
  /** Create a chat from its first user message; returns the new chat id. */
  createChat: (firstUserMessage: Message) => string;
  appendMessage: (chatId: string, message: Message) => void;
}

export function useChats(): ChatsStore {
  const [chats, setChats] = useState<Chat[]>(loadChats);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const persistTimer = useRef<number | null>(null);

  // Debounced persistence keeps rapid message appends cheap.
  useEffect(() => {
    if (persistTimer.current !== null) window.clearTimeout(persistTimer.current);
    persistTimer.current = window.setTimeout(() => saveChats(chats), 150);
    return () => {
      if (persistTimer.current !== null) window.clearTimeout(persistTimer.current);
    };
  }, [chats]);

  const newChat = useCallback(() => setActiveChatId(null), []);

  const selectChat = useCallback((id: string) => setActiveChatId(id), []);

  const deleteChat = useCallback(
    (id: string) => {
      setChats((prev) => prev.filter((chat) => chat.id !== id));
      setActiveChatId((current) => (current === id ? null : current));
    },
    [],
  );

  const createChat = useCallback((firstUserMessage: Message): string => {
    const now = Date.now();
    const chat: Chat = {
      id: crypto.randomUUID(),
      title: titleFromQuery(firstUserMessage.content),
      messages: [firstUserMessage],
      createdAt: now,
      updatedAt: now,
    };
    setChats((prev) => [chat, ...prev]);
    setActiveChatId(chat.id);
    return chat.id;
  }, []);

  const appendMessage = useCallback((chatId: string, message: Message) => {
    setChats((prev) =>
      prev.map((chat) =>
        chat.id === chatId
          ? {
              ...chat,
              messages: [...chat.messages, message],
              updatedAt: Date.now(),
            }
          : chat,
      ),
    );
  }, []);

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) ?? null,
    [chats, activeChatId],
  );

  const groups = useMemo(() => groupChatsByDate(chats), [chats]);

  return {
    chats,
    groups,
    activeChatId,
    activeChat,
    newChat,
    selectChat,
    deleteChat,
    createChat,
    appendMessage,
  };
}
