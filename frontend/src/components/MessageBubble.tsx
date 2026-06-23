import { AlertTriangle, Landmark } from "lucide-react";

import { getCitedSources } from "../lib/citations";
import type { CitedParent, Message } from "../lib/types";
import { CitationMarkdown } from "./CitationMarkdown";

interface MessageBubbleProps {
  message: Message;
  onOpenSource: (sources: CitedParent[], anchorId: string) => void;
}

export function MessageBubble({ message, onOpenSource }: MessageBubbleProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-user-bubble px-4 py-2.5 text-[15px] leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  const isError = message.status === "error";
  const sources = message.sources ?? [];
  const citedSources = getCitedSources(message.content, sources);

  return (
    <div className="flex gap-3">
      <span
        className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg ${
          isError ? "bg-surface-2 text-danger" : "bg-accent-soft text-accent"
        }`}
        aria-hidden
      >
        {isError ? <AlertTriangle size={14} /> : <Landmark size={14} />}
      </span>
      <div className="min-w-0 flex-1">
        {isError ? (
          <p className="text-[15px] leading-relaxed text-ink-muted">
            {message.content}
          </p>
        ) : (
          <div className="markdown text-[15px]">
            <CitationMarkdown
              content={message.content}
              sources={sources}
              onOpenSource={(anchorId) =>
                onOpenSource(citedSources, anchorId)
              }
            />
          </div>
        )}
      </div>
    </div>
  );
}
