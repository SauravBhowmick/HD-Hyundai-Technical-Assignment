import { useEffect, useMemo, useState } from "react";
import { listPlots, plotUrl } from "../api";
import { PlotsManifest } from "../types";

type TabKey = "actual" | "per_model";

const TABS: { key: TabKey; label: string }[] = [
  { key: "actual",    label: "Comparison" },
  { key: "per_model", label: "Per-model" },
];

function prettyName(name: string): string {
  return name
    .replace(/\.png$/, "")
    .replace(/_/g, " ")
    .replace(/\bpr curve\b/i, "PR curve")
    .replace(/\broc curve\b/i, "ROC curve")
    .replace(/\bprob hist\b/i, "Probability histogram");
}

export function ModelPlotsCard() {
  const [manifest, setManifest] = useState<PlotsManifest | null>(null);
  const [tab, setTab] = useState<TabKey>("actual");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const data = await listPlots();
        if (mounted) setManifest(data);
      } catch (e: any) {
        if (mounted) setError(String(e?.message ?? e));
      }
    })();
    return () => { mounted = false; };
  }, []);

  const visible = useMemo(() => {
    if (!manifest) return [] as string[];
    if (tab === "actual") return manifest.groups.actual_comparison;
    return manifest.groups.actual_per_model;
  }, [manifest, tab]);

  return (
    <div className="card">
      <div className="plots-head">
        <div>
          <h2 style={{ margin: 0 }}>Model curves</h2>
          <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
            Scored on the held-out test split from the time-aware split.
            Training rows never enter these plots.
          </div>
        </div>
      </div>

      <div className="plots-tabs">
        {TABS.map((t) => {
          const count =
            t.key === "actual"
              ? manifest?.groups.actual_comparison.length ?? 0
              : manifest?.groups.actual_per_model.length ?? 0;
          return (
            <button
              key={t.key}
              className={`tab ${tab === t.key ? "tab--active" : ""}`}
              onClick={() => setTab(t.key)}
              disabled={count === 0}
            >
              {t.label} <span className="muted">({count})</span>
            </button>
          );
        })}
      </div>

      {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}

      {!manifest && !error && (
        <div className="muted">Loading plots…</div>
      )}

      {manifest && visible.length === 0 && !error && (
        <div className="muted plots-empty">No plots available.</div>
      )}

      {visible.length > 0 && (
        <div className="plots-grid">
          {visible.map((name) => (
            <figure key={name} className="plot">
              <img src={plotUrl(name)} alt={prettyName(name)} loading="lazy" />
              <figcaption className="muted">{prettyName(name)}</figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}
