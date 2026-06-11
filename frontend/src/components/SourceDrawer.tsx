import { ExternalLink, X } from "lucide-react";
import { useEffect, useRef } from "react";

import { sourceBadges, sourceTitle } from "../lib/sources";
import { metaString, type CitedParent } from "../lib/types";

export interface DrawerState {
  sources: CitedParent[];
  anchorId: string;
}

interface SourceDrawerProps {
  state: DrawerState;
  onClose: () => void;
}

export function SourceDrawer({ state, onClose }: SourceDrawerProps) {
  const anchorRef = useRef<HTMLElement | null>(null);

  // Scroll the anchored source into view and flash-highlight it.
  useEffect(() => {
    const target = anchorRef.current;
    if (!target) return;
    target.scrollIntoView({ block: "start", behavior: "instant" });
    target.classList.remove("anchor-flash");
    // Force a reflow so the animation restarts when re-anchoring.
    void target.offsetWidth;
    target.classList.add("anchor-flash");
  }, [state.anchorId]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-40" role="dialog" aria-label="Source details">
      <button
        type="button"
        aria-label="Close source panel"
        onClick={onClose}
        className="absolute inset-0 cursor-default bg-black/25"
      />
      <div className="absolute bottom-0 right-0 top-0 flex w-full max-w-xl flex-col border-l border-border bg-surface shadow-xl">
        <header className="flex items-center justify-between border-b border-border px-5 py-3.5">
          <h2 className="text-[15px] font-semibold">
            Sources ({state.sources.length})
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg p-1.5 text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
          >
            <X size={17} />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {state.sources.map((source) => {
            const isAnchor = source.parent_id === state.anchorId;
            const sourceUrl = metaString(source.metadata, "source_url");
            return (
              <article
                key={source.parent_id}
                ref={isAnchor ? (node) => void (anchorRef.current = node) : undefined}
                className={`mb-4 scroll-mt-4 rounded-xl border p-4 ${
                  isAnchor ? "border-accent" : "border-border"
                }`}
              >
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <h3 className="text-[14px] font-semibold text-ink">
                    {sourceTitle(source)}
                  </h3>
                  {sourceBadges(source).map((badge) => (
                    <span
                      key={badge}
                      className="rounded-md border border-border bg-surface-2 px-1.5 py-0.5 text-[11px] text-ink-muted"
                    >
                      {badge}
                    </span>
                  ))}
                </div>

                {sourceUrl && (
                  <a
                    href={sourceUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-1.5 inline-flex items-center gap-1 text-[12.5px] text-accent hover:text-accent-hover"
                  >
                    View on irs.gov
                    <ExternalLink size={12} />
                  </a>
                )}

                <p className="mt-3 whitespace-pre-wrap border-t border-border pt-3 text-[13.5px] leading-relaxed text-ink">
                  {source.text_content}
                </p>
              </article>
            );
          })}
        </div>
      </div>
    </div>
  );
}
