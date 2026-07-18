import { useCallback, useEffect, useRef, useState } from "react";

import { ChatView } from "./components/ChatView";
import { InputBar } from "./components/InputBar";
import { Sidebar } from "./components/Sidebar";
import { SourceDrawer, type DrawerState } from "./components/SourceDrawer";
import { ThemeToggle } from "./components/ThemeToggle";
import { WelcomeHero } from "./components/WelcomeHero";
import { useChats } from "./hooks/useChats";
import { useComposerDock } from "./hooks/useComposerDock";
import { ask, AskError, fetchRateLimit } from "./lib/api";
import type { CitedParent, Message, RateLimitInfo } from "./lib/types";
import { toChatHistory } from "./lib/types";

function makeMessage(partial: Omit<Message, "id" | "createdAt">): Message {
  return { ...partial, id: crypto.randomUUID(), createdAt: Date.now() };
}

export default function App() {
  const store = useChats();
  const [pendingChatId, setPendingChatId] = useState<string | null>(null);
  const [drawer, setDrawer] = useState<DrawerState | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [rateLimit, setRateLimit] = useState<RateLimitInfo | null>(null);
  const mainRef = useRef<HTMLElement | null>(null);
  const composerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    void fetchRateLimit().then(setRateLimit);
  }, []);

  const sendMessage = useCallback(
    async (query: string) => {
      if (pendingChatId !== null) return;

      const priorMessages = store.activeChat?.messages ?? [];
      const history = toChatHistory(priorMessages);

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
        const response = await ask(query, history);
        if (response.rate_limit) setRateLimit(response.rate_limit);
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
        if (error instanceof AskError && error.rateLimit) {
          setRateLimit(error.rateLimit);
        }
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
  const isCentered = messages.length === 0 && !viewingPendingChat;
  const showSuggestions = isCentered;
  const quotaExhausted =
    rateLimit !== null && rateLimit.remaining <= 0;

  const { offsetY, ready, durationMs, easing } = useComposerDock(
    mainRef,
    composerRef,
    isCentered,
  );

  return (
    <div className="flex h-full overflow-hidden">
      <Sidebar
        groups={store.groups}
        activeChatId={store.activeChatId}
        isDraft={store.activeChatId === null}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((value) => !value)}
        onNewChat={store.newChat}
        onSelectChat={store.selectChat}
        onDeleteChat={store.deleteChat}
      />

      <main ref={mainRef} className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="flex shrink-0 items-center justify-end gap-3 px-4 pb-1 pt-3">
          {rateLimit && (
            <p
              className={`text-[12px] tabular-nums ${
                quotaExhausted ? "text-danger" : "text-ink-faint"
              }`}
              title={
                rateLimit.reset_at
                  ? `Resets ${rateLimit.reset_at}`
                  : undefined
              }
            >
              {rateLimit.remaining} / {rateLimit.limit} free today
            </p>
          )}
          <ThemeToggle />
        </header>

        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          <ChatView
            messages={messages}
            pending={viewingPendingChat}
            onOpenSource={openSource}
          />
        </div>

        <div
          ref={composerRef}
          className={`composer-dock shrink-0 px-4 pb-5 pt-6 ${
            isCentered ? "transition-transform" : ""
          } ${ready ? "opacity-100" : "opacity-0"}`}
          style={{
            transform: `translateY(${offsetY}px)`,
            transitionDuration: `${durationMs}ms`,
            transitionTimingFunction: easing,
          }}
        >
          <div
            className={`mx-auto flex w-full max-w-[820px] flex-col items-center transition-all duration-400 ease-out ${
              isCentered
                ? "mb-8 max-h-[280px] opacity-100"
                : "pointer-events-none mb-0 max-h-0 overflow-hidden opacity-0"
            }`}
            aria-hidden={!isCentered}
          >
            <WelcomeHero />
          </div>

          <InputBar
            key={store.activeChatId ?? "draft"}
            pending={pendingChatId !== null}
            disabled={quotaExhausted}
            showSuggestions={showSuggestions && !quotaExhausted}
            docked={!isCentered}
            onSend={(query) => void sendMessage(query)}
            onAskSample={(query) => void sendMessage(query)}
          />
        </div>
      </main>

      {drawer && <SourceDrawer state={drawer} onClose={() => setDrawer(null)} />}
    </div>
  );
}
