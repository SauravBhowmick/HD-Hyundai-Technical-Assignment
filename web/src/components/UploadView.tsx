import { useState } from "react";
import { uploadAndRun } from "../api";
import { PipelineResult, SlotKey } from "../types";

const SLOTS: { key: SlotKey; label: string; hint: string }[] = [
  { key: "telemetry", label: "Telemetry",   hint: "hourly volt / rotate / pressure / vibration per machine" },
  { key: "errors",    label: "Errors",      hint: "non-fatal error codes with timestamp + machineID" },
  { key: "failures",  label: "Failures",    hint: "component failures (ground-truth labels)" },
  { key: "machines",  label: "Machines",    hint: "static machine metadata (model + age)" },
  { key: "maint",     label: "Maintenance", hint: "component replacements / scheduled maintenance" },
];

type FilesMap = Partial<Record<SlotKey, File>>;

type RunState =
  | { kind: "idle" }
  | { kind: "running"; elapsed: number }
  | { kind: "error"; message: string; result?: PipelineResult }
  | { kind: "done"; result: PipelineResult };

interface Props {
  onSuccess: () => void;
}

export function UploadView({ onSuccess }: Props) {
  const [files, setFiles] = useState<FilesMap>({});
  const [state, setState] = useState<RunState>({ kind: "idle" });

  const allPicked = SLOTS.every((s) => !!files[s.key]);

  function onPick(slot: SlotKey, f: File | null) {
    setFiles((prev) => {
      const next = { ...prev };
      if (f) next[slot] = f;
      else delete next[slot];
      return next;
    });
  }

  async function onRun() {
    if (!allPicked) return;
    setState({ kind: "running", elapsed: 0 });
    const started = Date.now();
    const ticker = window.setInterval(() => {
      setState((s) => s.kind === "running"
        ? { kind: "running", elapsed: Math.floor((Date.now() - started) / 1000) }
        : s);
    }, 500);
    try {
      const result = await uploadAndRun(files as Record<SlotKey, File>);
      window.clearInterval(ticker);
      if (!result.ok) {
        setState({ kind: "error", message: result.error ?? "pipeline failed", result });
        return;
      }
      setState({ kind: "done", result });
      onSuccess();
    } catch (e: any) {
      window.clearInterval(ticker);
      setState({ kind: "error", message: String(e?.message ?? e) });
    }
  }

  const running = state.kind === "running";
  return (
    <div className="upload-view">
      <div className="upload-card">
        <h1>Upload your dataset</h1>
        <p className="muted">
          Drop the five CSVs below to train a 24-hour failure-risk model on your
          own data. The pipeline runs validate → features → train (LR + LightGBM)
          → evaluate → drift, then unlocks the dashboard.
        </p>

        <div className="upload-grid">
          {SLOTS.map((s) => (
            <SlotInput
              key={s.key}
              label={s.label}
              hint={s.hint}
              file={files[s.key]}
              onChange={(f) => onPick(s.key, f)}
              disabled={running}
            />
          ))}
        </div>

        <div className="upload-actions">
          <button
            className="primary"
            onClick={onRun}
            disabled={!allPicked || running}
          >
            {running ? "Running pipeline..." : "Run analysis"}
          </button>
          {!allPicked && !running && (
            <span className="muted">
              Pick all five files to enable. ({Object.keys(files).length}/5)
            </span>
          )}
        </div>

        {state.kind === "running" && (
          <ProgressBlock elapsed={state.elapsed} />
        )}

        {state.kind === "done" && (
          <ResultBlock result={state.result} variant="ok" />
        )}

        {state.kind === "error" && (
          <ResultBlock result={state.result ?? null} variant="error" message={state.message} />
        )}
      </div>
    </div>
  );
}

interface SlotInputProps {
  label: string;
  hint: string;
  file: File | undefined;
  onChange: (f: File | null) => void;
  disabled?: boolean;
}

function SlotInput({ label, hint, file, onChange, disabled }: SlotInputProps) {
  const [drag, setDrag] = useState(false);
  return (
    <label
      className={`slot ${file ? "slot--filled" : ""} ${drag ? "slot--drag" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        if (disabled) return;
        const f = e.dataTransfer.files?.[0];
        if (f) onChange(f);
      }}
    >
      <div className="slot-head">
        <strong>{label}</strong>
        {file && (
          <button
            type="button"
            className="link"
            onClick={(e) => { e.preventDefault(); onChange(null); }}
            disabled={disabled}
          >
            clear
          </button>
        )}
      </div>
      <div className="slot-hint">{hint}</div>
      <input
        type="file"
        accept=".csv,text/csv"
        disabled={disabled}
        onChange={(e) => onChange(e.target.files?.[0] ?? null)}
      />
      {file && (
        <div className="slot-file">
          {file.name} <span className="muted">({(file.size / 1024).toFixed(0)} KB)</span>
        </div>
      )}
    </label>
  );
}

function ProgressBlock({ elapsed }: { elapsed: number }) {
  const STAGES = [
    { name: "Saving uploads",      at: 0 },
    { name: "Validating schemas",  at: 2 },
    { name: "Building features",   at: 5 },
    { name: "Training LR + LightGBM", at: 12 },
    { name: "Evaluating + drift",  at: 30 },
  ];
  return (
    <div className="progress">
      <div className="progress-bar"><div className="progress-bar-inner" /></div>
      <div className="muted">Pipeline running ({elapsed}s elapsed, typically 30-60s)</div>
      <ul className="progress-stages">
        {STAGES.map((s) => {
          const active = elapsed >= s.at;
          return (
            <li key={s.name} className={active ? "active" : ""}>
              <span className="dot" /> {s.name}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

interface ResultBlockProps {
  result: PipelineResult | null;
  variant: "ok" | "error";
  message?: string;
}

function ResultBlock({ result, variant, message }: ResultBlockProps) {
  return (
    <div className={`result result--${variant}`}>
      <strong>{variant === "ok" ? "Pipeline complete." : "Pipeline failed."}</strong>
      {message && <div className="error">{message}</div>}
      {result?.stages && (
        <ul className="stage-list">
          {result.stages.map((s) => (
            <li key={s.name} className={s.ok ? "ok" : "fail"}>
              <span>{s.ok ? "✓" : "✗"}</span>
              <span><strong>{s.name}</strong> ({s.seconds.toFixed(1)}s)</span>
              {s.info && <span className="muted"> – {s.info}</span>}
            </li>
          ))}
        </ul>
      )}
      {result?.metrics?.runs && (
        <div className="metrics-table">
          <h3>Model metrics</h3>
          <table>
            <thead>
              <tr>
                <th>run</th><th>PR-AUC</th><th>ROC-AUC</th>
                <th>precision</th><th>recall</th><th>F1</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.metrics.runs).map(([name, m]) => (
                <tr key={name} className={name === result.metrics?.best_run ? "best" : ""}>
                  <td>{name}{name === result.metrics?.best_run ? "  ★" : ""}</td>
                  <td>{m.pr_auc?.toFixed(3)}</td>
                  <td>{m.roc_auc?.toFixed(3)}</td>
                  <td>{m.precision?.toFixed(3)}</td>
                  <td>{m.recall?.toFixed(3)}</td>
                  <td>{m.f1?.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
