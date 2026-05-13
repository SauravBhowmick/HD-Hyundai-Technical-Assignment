import { DigitalTwin } from "../types";
import { RiskGauge } from "./RiskGauge";
import { EvidenceList } from "./EvidenceList";

type Props = { twin: DigitalTwin; threshold?: number };

const PRESCRIPTION_LABEL: Record<DigitalTwin["prescription"], string> = {
  continue: "Continue operation",
  monitor: "Increased monitoring",
  inspect: "Inspect on next shift",
  schedule_maintenance: "Schedule maintenance window",
  urgent_maintenance: "Urgent maintenance",
};

export function DigitalTwinCard({ twin, threshold }: Props) {
  return (
    <div className="card">
      <div className="twin-header">
        <div>
          <div className="twin-title">Machine #{twin.machineID}</div>
          <div className="twin-sub">{twin.timestamp}</div>
        </div>
        <span className={`health-badge health-${twin.health_state}`}>
          {twin.health_state}
        </span>
      </div>

      <h2>24h failure risk</h2>
      <RiskGauge value={twin.failure_risk_24h} threshold={threshold} />

      <div style={{ height: 16 }} />

      <div className="kv">
        <div className="k">Likely component</div>
        <div>{twin.likely_component ?? <span className="muted">-</span>}</div>

        <div className="k">Confidence</div>
        <div>{twin.confidence}</div>

        <div className="k">Prescription</div>
        <div>{PRESCRIPTION_LABEL[twin.prescription]}</div>
      </div>

      <div style={{ height: 16 }} />
      <h2>Main evidence</h2>
      <EvidenceList items={twin.main_evidence} />
    </div>
  );
}
