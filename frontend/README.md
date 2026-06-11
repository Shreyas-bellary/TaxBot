# TaxBot Chat Frontend

A Claude-inspired, financial-grade chat interface for the TaxBot IRS RAG API.

- **Stack**: Vite, React 19, TypeScript, Tailwind CSS v4
- **Persistence**: chat history lives only in the browser (`localStorage`) — nothing is stored server-side
- **Themes**: light / dark / system, persisted across sessions
- **Citations**: every grounded answer shows a collapsible Sources panel; clicking a source opens an anchored drawer with the full IRS chunk and a link to irs.gov

## Development

```bash
# from the repo root
make api            # start the FastAPI backend on :8000
make frontend-dev   # start the Vite dev server on :5173
```

Or directly:

```bash
cd frontend
npm install
npm run dev
```

The API base URL defaults to `http://localhost:8000`; override it by copying
`.env.example` to `.env` and setting `VITE_API_BASE_URL`.

> The backend must allow this origin via `TAXBOT_CORS_ALLOW_ORIGINS`
> (defaults to `http://localhost:5173`).

## Production build

```bash
npm run build    # typecheck + bundle into dist/
npm run preview  # serve the production build locally
```

## Structure

```
src/
  lib/        types mirrored from the FastAPI models, API client, localStorage
  hooks/      useChats — local-only chat store with date grouping
  components/ Sidebar, ChatView, MessageBubble, SourcesPanel, SourceDrawer, InputBar
  theme.tsx   light/dark/system ThemeProvider
  index.css   Tailwind v4 design tokens + markdown styles
```
