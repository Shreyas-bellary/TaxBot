import { Landmark } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";

import type { CitedParent, Message } from "../lib/types";
import { MessageBubble } from "./MessageBubble";
import { ThinkingIndicator } from "./ThinkingIndicator";

const SAMPLE_QUESTIONS = [
  "What is the standard deduction for tax year 2024?",
  "How do I file Form 1040 if I'm self-employed?",
  "Who qualifies for the Earned Income Tax Credit?",
  "What are the estimated tax payment deadlines?",
];

interface ChatViewProps {
  messages: Message[];
  pending: boolean;
  onOpenSource: (sources: CitedParent[], anchorId: string) => void;
  onAskSample: (query: string) => void;
}

export function ChatView({
  messages,
  pending,
  onOpenSource,
  onAskSample,
}: ChatViewProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [stickToBottom, setStickToBottom] = useState(true);

  // Auto-scroll on new content unless the user has scrolled up to read.
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
    return (
      <div className="flex flex-1 items-center justify-center overflow-y-auto px-4">
        <div className="w-full max-w-[560px] pb-24 text-center">
          <span className="mx-auto mb-5 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-soft text-accent">
            <Landmark size={22} />
          </span>
          <h1 className="text-[22px] font-semibold tracking-tight">
            What can I help you with?
          </h1>
          <p className="mx-auto mt-2 max-w-md text-[14px] leading-relaxed text-ink-muted">
            I answer U.S. tax questions using official IRS forms, instructions,
            and publications — every answer cites its sources.
          </p>
          <div className="mt-7 grid gap-2 sm:grid-cols-2">
            {SAMPLE_QUESTIONS.map((question) => (
              <button
                key={question}
                type="button"
                onClick={() => onAskSample(question)}
                className="rounded-xl border border-border bg-surface px-4 py-3 text-left text-[13.5px] leading-snug text-ink-muted transition-colors hover:border-border-strong hover:bg-surface-2 hover:text-ink"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-[820px] px-4 py-8">
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
