export type HealthState = "healthy" | "watch" | "degraded" | "critical";
export type Confidence = "low" | "medium" | "high";
export type Prescription =
  | "continue"
  | "monitor"
  | "inspect"
  | "schedule_maintenance"
  | "urgent_maintenance";

export interface DigitalTwin {
  machineID: number;
  timestamp: string;
  failure_risk_24h: number;
  health_state: HealthState;
  likely_component: "comp1" | "comp2" | "comp3" | "comp4" | null;
  confidence: Confidence;
  main_evidence: string[];
  prescription: Prescription;
}

export interface MachineInfo {
  machineID: number;
  model: string;
  age_years: number;
}

export interface DatasetInfo {
  min_datetime: string;
  max_datetime: string;
  n_machines: number;
  model_name: string;
  threshold: number;
}

export interface HistoryPoint {
  datetime: string;
  volt?: number | null;
  rotate?: number | null;
  pressure?: number | null;
  vibration?: number | null;
  errors: string[];
  maint: string[];
  failures: string[];
}
