import { AlertTriangle, Landmark } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { CitedParent, Message } from "../lib/types";
import { SourcesPanel } from "./SourcesPanel";

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
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}
        {sources.length > 0 && (
          <SourcesPanel
            sources={sources}
            onOpenSource={(anchorId) => onOpenSource(sources, anchorId)}
          />
        )}
      </div>
    </div>
  );
}
