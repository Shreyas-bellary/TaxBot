import { BookOpen, ChevronRight } from "lucide-react";
import { useState } from "react";

import { sourceBadges, sourceTitle } from "../lib/sources";
import type { CitedParent } from "../lib/types";

interface SourcesPanelProps {
  sources: CitedParent[];
  onOpenSource: (anchorId: string) => void;
}

export function SourcesPanel({ sources, onOpenSource }: SourcesPanelProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-3 overflow-hidden rounded-xl border border-border bg-surface">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3.5 py-2.5 text-left text-[13px] font-medium text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
      >
        <ChevronRight
          size={14}
          className={`transition-transform ${open ? "rotate-90" : ""}`}
        />
        <BookOpen size={14} />
        Sources ({sources.length})
      </button>

      {open && (
        <ul className="border-t border-border">
          {sources.map((source) => (
            <li key={source.parent_id} className="border-b border-border last:border-b-0">
              <button
                type="button"
                onClick={() => onOpenSource(source.parent_id)}
                className="block w-full px-3.5 py-3 text-left transition-colors hover:bg-surface-2"
              >
                <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="text-[13.5px] font-medium text-ink">
                    {sourceTitle(source)}
                  </span>
                  {sourceBadges(source).map((badge) => (
                    <span
                      key={badge}
                      className="rounded-md border border-border bg-surface-2 px-1.5 py-0.5 text-[11px] text-ink-muted"
                    >
                      {badge}
                    </span>
                  ))}
                </span>
                <span className="mt-1 line-clamp-2 block text-[12.5px] leading-relaxed text-ink-muted">
                  {source.text_content}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
