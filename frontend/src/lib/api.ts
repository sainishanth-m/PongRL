/**
 * API client for PongRL Pro.
 * Connects to the FastAPI backend with seamless local fallback data
 * so the frontend works standalone on Vercel or locally with Python.
 */
import { PROJECT_SUMMARY } from "@/data/summary";
import { REAL_TRAINING_POINTS, TRAINING_METRICS_SUMMARY } from "@/data/trainingData";
import { MODEL_BENCHMARKS, CHECKPOINTS_INVENTORY } from "@/data/modelsData";
import { HYPERPARAMETERS_DATA } from "@/data/hyperparamsData";
import { RECORDINGS_DATA } from "@/data/recordingsData";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchProjectSummary() {
  try {
    const res = await fetch(`${API_BASE}/api/project/summary`, { next: { revalidate: 60 } });
    if (res.ok) return await res.json();
  } catch (err) {
    // Backend offline or running standalone on Vercel
  }
  return PROJECT_SUMMARY;
}

export async function fetchTrainingMetrics(sampleStep = 25) {
  try {
    const res = await fetch(`${API_BASE}/api/training/metrics?sample_step=${sampleStep}`, { next: { revalidate: 60 } });
    if (res.ok) return await res.json();
  } catch (err) {
    // Fallback
  }
  return {
    total_episodes: TRAINING_METRICS_SUMMARY.totalEpisodes,
    initial_reward: TRAINING_METRICS_SUMMARY.initialReward,
    best_reward: TRAINING_METRICS_SUMMARY.bestReward,
    final_reward: TRAINING_METRICS_SUMMARY.finalReward,
    chart_points: REAL_TRAINING_POINTS,
    quartiles: {
      q1_exploration: TRAINING_METRICS_SUMMARY.quartiles[0],
      q2_early_stabilization: TRAINING_METRICS_SUMMARY.quartiles[1],
      q3_inflection: TRAINING_METRICS_SUMMARY.quartiles[2],
      q4_asymptotic_master: TRAINING_METRICS_SUMMARY.quartiles[3],
    }
  };
}

export async function fetchModelBenchmarks() {
  try {
    const res = await fetch(`${API_BASE}/api/models/comparison`, { next: { revalidate: 60 } });
    if (res.ok) return await res.json();
  } catch (err) {
    // Fallback
  }
  return MODEL_BENCHMARKS;
}

export async function fetchCheckpoints() {
  try {
    const res = await fetch(`${API_BASE}/api/models`, { next: { revalidate: 60 } });
    if (res.ok) return await res.json();
  } catch (err) {
    // Fallback
  }
  return CHECKPOINTS_INVENTORY;
}

export async function fetchConfig() {
  try {
    const res = await fetch(`${API_BASE}/api/config`, { next: { revalidate: 60 } });
    if (res.ok) return await res.json();
  } catch (err) {
    // Fallback
  }
  return {
    current_algorithm: "reinforce",
    groups: HYPERPARAMETERS_DATA
  };
}

export async function fetchRecordings() {
  try {
    const res = await fetch(`${API_BASE}/api/game/recordings`, { next: { revalidate: 60 } });
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) return data;
    }
  } catch (err) {
    // Fallback
  }
  return RECORDINGS_DATA;
}

export async function queryAIAction(courtState: {
  ball_x: number;
  ball_y: number;
  ball_vx: number;
  ball_vy: number;
  ai_y: number;
  human_y: number;
  court_w?: number;
  court_h?: number;
}) {
  try {
    const res = await fetch(`${API_BASE}/api/game/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(courtState),
    });
    if (res.ok) return await res.json();
  } catch (err) {
    // Fallback to client-side decision
  }
  
  // Client-side fallback mimicking trained policy
  const dy = courtState.ball_y - courtState.ai_y;
  const up_prob = 1 / (1 + Math.exp(dy / 25));
  const is_up = up_prob > 0.5;
  return {
    action: is_up ? 2 : 3,
    action_name: is_up ? "UP" : "DOWN",
    up_prob: Number(up_prob.toFixed(4)),
    latency_ms: 0.8
  };
}

export async function stepAtariSimulation(sessionId = "default", steps = 1) {
  try {
    const res = await fetch(`${API_BASE}/api/game/session/step?session_id=${sessionId}&steps=${steps}`, {
      method: "POST"
    });
    if (res.ok) return await res.json();
  } catch (err) {
    // Fallback
  }
  return null;
}
