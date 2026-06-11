import { Landmark } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";

import type { CitedParent, Message } from "../lib/types";
import { MessageBubble } from "./MessageBubble";
import { ThinkingIndicator } from "./ThinkingIndicator";

interface ChatViewProps {
  messages: Message[];
  pending: boolean;
  onOpenSource: (sources: CitedParent[], anchorId: string) => void;
}

export function ChatView({ messages, pending, onOpenSource }: ChatViewProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [stickToBottom, setStickToBottom] = useState(true);

  useLayoutEffect(() => {
    const container = scrollRef.current;
    if (container && stickToBottom) {
      container.scrollTop = container.scrollHeight;
    }
  }, [messages, pending, stickToBottom]);

  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;
    const onScroll = () => {
      const distance =
        container.scrollHeight - container.scrollTop - container.clientHeight;
      setStickToBottom(distance < 80);
    };
    container.addEventListener("scroll", onScroll, { passive: true });
    return () => container.removeEventListener("scroll", onScroll);
  }, []);

  if (messages.length === 0 && !pending) {
    return null;
  }

  return (
    <div ref={scrollRef} className="h-full min-h-0 overflow-y-auto">
      <div className="mx-auto w-full max-w-[820px] px-4 pb-8 pt-2">
        <div className="flex flex-col gap-7">
          {messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              onOpenSource={onOpenSource}
            />
          ))}
          {pending && (
            <div className="flex gap-3">
              <span
                className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-accent-soft text-accent"
                aria-hidden
              >
                <Landmark size={14} />
              </span>
              <ThinkingIndicator />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
