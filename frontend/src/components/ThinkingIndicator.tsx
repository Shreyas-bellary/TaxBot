export function ThinkingIndicator() {
  return (
    <div
      className="flex items-center gap-2 py-1 text-ink-muted"
      role="status"
      aria-label="TaxBot is thinking"
    >
      <span className="flex items-center gap-1" aria-hidden>
        <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-ink-muted" />
        <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-ink-muted" />
        <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-ink-muted" />
      </span>
      <span className="text-[13px]">Consulting IRS documents…</span>
    </div>
  );
}
