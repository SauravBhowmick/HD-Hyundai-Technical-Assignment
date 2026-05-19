import { useEffect, useMemo, useState } from "react";
import { getInfo, getSession, history, listMachines, predict, resetSession } from "./api";
import { DatasetInfo, DigitalTwin, HistoryPoint, MachineInfo, SessionStatus } from "./types";
import { MachineSelector } from "./components/MachineSelector";
import { TimestampPicker } from "./components/TimestampPicker";
import { DigitalTwinCard } from "./components/DigitalTwinCard";
import { TelemetryChart } from "./components/TelemetryChart";
import { EventTimeline } from "./components/EventTimeline";
import { UploadView } from "./components/UploadView";
import { ModelPlotsCard } from "./components/ModelPlotsCard";

function plusHours(iso: string, hours: number): string {
  const d = new Date(iso);
  d.setHours(d.getHours() + hours);
  return d.toISOString().slice(0, 19);
}

export default function App() {
  const [session, setSession] = useState<SessionStatus | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);

  async function refreshSession() {
    try {
      setSession(await getSession());
    } catch (e: any) {
      setBootError(String(e?.message ?? e));
    }
  }

  useEffect(() => { refreshSession(); }, []);

  if (bootError) {
    return <div className="boot-error">Cannot reach API: {bootError}</div>;
  }
  if (!session) {
    return <div className="boot-loading">Connecting to API...</div>;
  }
  if (!session.loaded) {
    return <UploadView onSuccess={refreshSession} />;
  }
  return <Dashboard session={session} onReset={async () => {
    await resetSession();
    await refreshSession();
  }} />;
}

interface DashboardProps {
  session: SessionStatus;
  onReset: () => Promise<void>;
}

function Dashboard({ session, onReset }: DashboardProps) {
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [machines, setMachines] = useState<MachineInfo[]>([]);
  const [machineID, setMachineID] = useState<number | null>(null);
  const [timestamp, setTimestamp] = useState<string>("");
  const [twin, setTwin] = useState<DigitalTwin | null>(null);
  const [hist, setHist] = useState<HistoryPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [i, m] = await Promise.all([getInfo(), listMachines()]);
        setInfo(i);
        setMachines(m);
        if (m.length && !machineID) setMachineID(m[0].machineID);
        if (!timestamp) {
          const d = new Date(i.min_datetime);
          d.setDate(d.getDate() + 90);
          setTimestamp(d.toISOString().slice(0, 19));
        }
      } catch (e: any) {
        setError(String(e?.message ?? e));
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const range = useMemo(() => {
    if (!info) return { min: "", max: "" };
    return { min: info.min_datetime.slice(0, 19), max: info.max_datetime.slice(0, 19) };
  }, [info]);

  async function onPredict() {
    if (!machineID || !timestamp) return;
    setLoading(true);
    setError(null);
    setTwin(null);
    try {
      const t = await predict(machineID, timestamp);
      setTwin(t);
      const start = plusHours(timestamp, -24);
      const end = plusHours(timestamp, 24);
      const h = await history(machineID, start, end);
      setHist(h);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          PdM Digital Twin
          <small>Predictive maintenance demo</small>
        </div>

        <MachineSelector
          machines={machines}
          value={machineID}
          onChange={setMachineID}
        />

        <TimestampPicker
          value={timestamp}
          min={range.min}
          max={range.max}
          onChange={setTimestamp}
        />

        <button
          className="primary"
          onClick={onPredict}
          disabled={loading || !machineID || !timestamp}
        >
          {loading ? "Predicting..." : "Run prediction"}
        </button>

        {error && <div className="error">{error}</div>}

        <div className="sidebar-foot">
          {info && (
            <div className="muted" style={{ fontSize: 12 }}>
              <div>model: {info.model_name}</div>
              <div>threshold: {info.threshold.toFixed(3)}</div>
              <div>{info.n_machines} machines</div>
              {session.uploaded_at && (
                <div>uploaded: {new Date(session.uploaded_at).toLocaleString()}</div>
              )}
            </div>
          )}
          <button className="link" onClick={onReset}>
            Upload new data
          </button>
        </div>
      </aside>

      <main className="main">
        {!twin && (
          <div className="card">
            <h2>Welcome</h2>
            <p>
              Pick a machine and a timestamp, then click <b>Run prediction</b>. The
              dashboard will show the digital-twin JSON returned by the FastAPI
              backend along with telemetry and event context around that time.
            </p>
            {session.pipeline?.runs && (
              <div className="metrics-table" style={{ marginTop: 16 }}>
                <h3>Models trained on your data</h3>
                <table>
                  <thead>
                    <tr>
                      <th>run</th><th>PR-AUC</th><th>ROC-AUC</th>
                      <th>precision</th><th>recall</th><th>F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(session.pipeline.runs).map(([name, m]) => (
                      <tr key={name} className={name === session.pipeline?.best_run ? "best" : ""}>
                        <td>{name}{name === session.pipeline?.best_run ? "  ★" : ""}</td>
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
        )}

        {!twin && <ModelPlotsCard />}

        {twin && (
          <div className="row-gap">
            <div className="grid">
              <DigitalTwinCard twin={twin} threshold={info?.threshold} />
              <div className="card">
                <h2>Telemetry (-24h / +24h)</h2>
                <TelemetryChart data={hist} />
              </div>
            </div>
            <div className="card">
              <h2>Event timeline</h2>
              <EventTimeline data={hist} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
