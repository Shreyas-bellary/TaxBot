import { useLayoutEffect, useState, type RefObject } from "react";

const DOCK_EASING = "cubic-bezier(0.32, 0.72, 0, 1)";
const DOCK_DURATION_MS = 520;

export function useComposerDock(
  mainRef: RefObject<HTMLElement | null>,
  composerRef: RefObject<HTMLElement | null>,
  centered: boolean,
) {
  const [offsetY, setOffsetY] = useState(0);
  const [ready, setReady] = useState(!centered);

  useLayoutEffect(() => {
    const main = mainRef.current;
    const composer = composerRef.current;
    if (!main || !composer) return;

    const measure = () => {
      if (!centered) {
        setOffsetY(0);
        setReady(true);
        return;
      }

      const mainHeight = main.clientHeight;
      const composerHeight = composer.offsetHeight;
      const composerTop = composer.offsetTop;
      const composerCenter = composerTop + composerHeight / 2;
      const mainCenter = mainHeight / 2;
      setOffsetY(mainCenter - composerCenter);
      setReady(true);
    };

    measure();

    const observer = new ResizeObserver(measure);
    observer.observe(main);
    observer.observe(composer);
    window.addEventListener("resize", measure);

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [centered, mainRef, composerRef]);

  return { offsetY, ready, durationMs: DOCK_DURATION_MS, easing: DOCK_EASING };
}
