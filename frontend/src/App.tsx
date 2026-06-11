import { useCallback, useState } from "react";

import { ChatView } from "./components/ChatView";
import { InputBar } from "./components/InputBar";
import { Sidebar } from "./components/Sidebar";
import { SourceDrawer, type DrawerState } from "./components/SourceDrawer";
import { useChats } from "./hooks/useChats";
import { ask, AskError } from "./lib/api";
import type { CitedParent, Message } from "./lib/types";

function makeMessage(partial: Omit<Message, "id" | "createdAt">): Message {
  return { ...partial, id: crypto.randomUUID(), createdAt: Date.now() };
}

export default function App() {
  const store = useChats();
  const [pendingChatId, setPendingChatId] = useState<string | null>(null);
  const [drawer, setDrawer] = useState<DrawerState | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const sendMessage = useCallback(
    async (query: string) => {
      if (pendingChatId !== null) return;

      const userMessage = makeMessage({
        role: "user",
        content: query,
        status: "complete",
      });

      const chatId =
        store.activeChatId === null
          ? store.createChat(userMessage)
          : store.activeChatId;
      if (store.activeChatId !== null) {
        store.appendMessage(chatId, userMessage);
      }
      setPendingChatId(chatId);

      try {
        const response = await ask(query);
        store.appendMessage(
          chatId,
          makeMessage({
            role: "assistant",
            content: response.answer,
            status: "complete",
            sources: response.parents,
            citations: response.citations,
          }),
        );
      } catch (error) {
        const content =
          error instanceof AskError
            ? error.message
            : "Something unexpected went wrong. Please try again.";
        store.appendMessage(
          chatId,
          makeMessage({ role: "assistant", content, status: "error" }),
        );
      } finally {
        setPendingChatId(null);
      }
    },
    [pendingChatId, store],
  );

  const openSource = useCallback(
    (sources: CitedParent[], anchorId: string) =>
      setDrawer({ sources, anchorId }),
    [],
  );

  const messages = store.activeChat?.messages ?? [];
  const viewingPendingChat =
    pendingChatId !== null && pendingChatId === store.activeChatId;

  return (
    <div className="flex h-full overflow-hidden">
      <Sidebar
        groups={store.groups}
        activeChatId={store.activeChatId}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((value) => !value)}
        onNewChat={store.newChat}
        onSelectChat={store.selectChat}
        onDeleteChat={store.deleteChat}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <ChatView
          messages={messages}
          pending={viewingPendingChat}
          onOpenSource={openSource}
          onAskSample={(query) => void sendMessage(query)}
        />
        <InputBar
          pending={pendingChatId !== null}
          onSend={(query) => void sendMessage(query)}
        />
      </main>

      {drawer && <SourceDrawer state={drawer} onClose={() => setDrawer(null)} />}
    </div>
  );
}
