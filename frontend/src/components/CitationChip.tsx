import { Link } from "lucide-react";
import { sourceTitle } from "../lib/sources";
import type { CitedParent } from "../lib/types";

interface CitationChipProps {
  source: CitedParent;
  onOpen: () => void;
}

export function CitationChip({ source, onOpen }: CitationChipProps) {
  const title = sourceTitle(source);

  return (
    <button
      type="button"
      onClick={onOpen}
      title={title}
      aria-label={`Open source ${title}`}
      className="citation-chip mx-0.5 inline-flex size-[1.125rem] shrink-0 items-center justify-center rounded border border-border/60 bg-surface-2/70 align-middle text-accent transition-colors duration-200 hover:border-accent/35 hover:bg-accent-soft dark:text-accent"
    >
      <Link size={10} strokeWidth={2.25} aria-hidden="true" />
    </button>
  );
}
