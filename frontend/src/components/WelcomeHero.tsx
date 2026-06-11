import { Landmark } from "lucide-react";

export function WelcomeHero() {
  return (
    <div className="welcome-enter w-full max-w-[560px] text-center">
      <span className="mx-auto mb-5 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-soft text-accent">
        <Landmark size={22} />
      </span>
      <h1 className="text-[22px] font-semibold tracking-tight">
        What can I help you with?
      </h1>
      <p className="mx-auto mt-2 max-w-md text-[14px] leading-relaxed text-ink-muted">
        I answer U.S. tax questions using official IRS forms, instructions, and
        publications — every answer cites its sources.
      </p>
    </div>
  );
}
