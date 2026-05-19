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

export type SlotKey = "telemetry" | "errors" | "failures" | "machines" | "maint";

export interface SessionStatus {
  loaded: boolean;
  uploaded_at?: string | null;
  files?: Record<string, string> | null;
  pipeline?: {
    best_run?: string;
    best_threshold?: number;
    feature_hash?: string;
    train_rows?: number;
    test_rows?: number;
    runs?: Record<string, {
      pr_auc?: number;
      roc_auc?: number;
      precision?: number;
      recall?: number;
      f1?: number;
      threshold?: number;
      false_alarms_per_machine_month?: number;
    }>;
  } | null;
}

export interface StageResult {
  name: string;
  ok: boolean;
  seconds: number;
  info?: string | null;
}

export interface PipelineResult {
  ok: boolean;
  stages: StageResult[];
  metrics?: SessionStatus["pipeline"];
  error?: string | null;
  error_id?: string | null;
}

export interface PlotsManifest {
  count: number;
  available: string[];
  groups: {
    actual_comparison: string[];
    actual_per_model: string[];
    other: string[];
  };
}
