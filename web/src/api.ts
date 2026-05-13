import { DatasetInfo, DigitalTwin, HistoryPoint, MachineInfo } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function jsonOrThrow<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
  }
  return (await resp.json()) as T;
}

export async function getInfo(): Promise<DatasetInfo> {
  return jsonOrThrow<DatasetInfo>(await fetch(`${BASE}/info`));
}

export async function listMachines(): Promise<MachineInfo[]> {
  return jsonOrThrow<MachineInfo[]>(await fetch(`${BASE}/machines`));
}

export async function predict(
  machineID: number,
  timestamp: string
): Promise<DigitalTwin> {
  return jsonOrThrow<DigitalTwin>(
    await fetch(`${BASE}/predict`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ machineID, timestamp }),
    })
  );
}

export async function history(
  machineID: number,
  start: string,
  end: string
): Promise<HistoryPoint[]> {
  const q = new URLSearchParams({ start, end }).toString();
  return jsonOrThrow<HistoryPoint[]>(
    await fetch(`${BASE}/history/${machineID}?${q}`)
  );
}
