import { ExternalLink, X } from "lucide-react";
import { useEffect, useRef } from "react";

import { sourceTitle } from "../lib/sources";
import { metaNumber, metaString, type CitedParent } from "../lib/types";

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
  const orderedSources = [
    ...state.sources.filter((source) => source.parent_id === state.anchorId),
    ...state.sources.filter((source) => source.parent_id !== state.anchorId),
  ];

  useEffect(() => {
    const target = anchorRef.current;
    if (!target) return;
    target.scrollIntoView({ block: "start", behavior: "instant" });
    target.classList.remove("anchor-flash");
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
        className="source-drawer-backdrop absolute inset-0 cursor-default bg-black/25"
      />
      <div className="source-drawer-panel absolute bottom-0 right-0 top-0 flex w-full max-w-md flex-col border-l border-border bg-surface shadow-xl">
        <header className="flex shrink-0 items-center justify-between border-b border-border px-4 py-3">
          <h2 className="text-[14px] font-semibold tracking-tight text-ink">
            Sources
            <span className="ml-1.5 font-normal text-ink-faint">
              ({state.sources.length})
            </span>
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg p-1.5 text-ink-faint transition-colors hover:bg-surface-2 hover:text-ink"
          >
            <X size={16} />
          </button>
        </header>

        <div className="flex flex-1 flex-col gap-1.5 overflow-y-auto px-3 py-2.5">
          {orderedSources.map((source) => {
            const isAnchor = source.parent_id === state.anchorId;
            const sourceUrl = metaString(source.metadata, "source_url");
            const docTitle = metaString(source.metadata, "doc_title");
            const docNumber = metaString(source.metadata, "doc_number");
            const taxYear = metaNumber(source.metadata, "tax_year");
            const title = sourceTitle(source);
            const showDocNumber = docNumber !== null && docNumber !== docTitle;

            const metaParts: string[] = [];
            if (showDocNumber && docNumber) metaParts.push(docNumber);
            if (taxYear !== null) metaParts.push(`Tax year ${taxYear}`);

            return (
              <article
                key={source.parent_id}
                ref={isAnchor ? (node) => void (anchorRef.current = node) : undefined}
                className={`source-card relative scroll-mt-2 rounded-lg py-2 pl-3.5 pr-3 transition-colors ${
                  isAnchor
                    ? "bg-surface-2"
                    : "hover:bg-surface-2/60"
                }`}
              >
                {isAnchor && (
                  <span
                    aria-hidden
                    className="absolute bottom-2 left-0 top-2 w-0.5 rounded-full bg-accent"
                  />
                )}

                <h3 className="pr-1 text-[13.5px] font-medium leading-snug text-ink">
                  {title}
                </h3>

                {metaParts.length > 0 && (
                  <p className="mt-0.5 text-[12px] leading-snug text-ink-muted">
                    {metaParts.join(" · ")}
                  </p>
                )}

                {sourceUrl && (
                  <a
                    href={sourceUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-1.5 inline-flex items-center gap-1 text-[11.5px] font-medium text-accent transition-colors hover:text-accent-hover"
                  >
                    irs.gov
                    <ExternalLink size={11} strokeWidth={2} />
                  </a>
                )}
              </article>
            );
          })}
        </div>
      </div>
    </div>
  );
}
