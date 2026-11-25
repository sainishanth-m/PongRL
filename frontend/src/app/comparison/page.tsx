"use client";

import { useState } from "react";
import { 
  GitCompare, 
  Layers, 
  Zap, 
  CheckCircle2, 
  XCircle, 
  ShieldCheck, 
  TrendingUp,
  Cpu,
  Clock,
  HardDrive
} from "lucide-react";
import { MODEL_BENCHMARKS, CHECKPOINTS_INVENTORY } from "@/data/modelsData";
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid,
  Legend
} from "recharts";

export default function ComparisonPage() {
  const [selectedAlgo, setSelectedAlgo] = useState("all");

  const filteredModels = selectedAlgo === "all" 
    ? MODEL_BENCHMARKS 
    : MODEL_BENCHMARKS.filter(m => m.algorithm.toLowerCase() === selectedAlgo.toLowerCase());

  // Chart data for algorithms
  const chartData = MODEL_BENCHMARKS.map(m => ({
    algorithm: m.algorithm,
    peakReward: m.asymptoticReward,
    sampleEfficiency: m.sampleEfficiencyEpisodes,
    latency: m.inferenceLatencyMs,
    memory: m.memoryUsageMb
  }));

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-slate-200/80 pb-6">
        <div>
          <div className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-0.5 text-[11px] font-semibold text-slate-700 shadow-xs mb-2">
            <GitCompare className="h-3.5 w-3.5 text-purple-600" />
            Empirical Benchmark Comparison
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-900">
            Reinforcement Learning Algorithm Comparison
          </h1>
          <p className="mt-2 text-sm text-slate-600 max-w-2xl leading-relaxed">
            Direct head-to-head empirical comparison across the three implemented reinforcement learning paradigms in PongRL Pro: 
            <strong> REINFORCE</strong>, <strong>Deep Q-Networks (DQN)</strong>, and <strong>Proximal Policy Optimization (PPO)</strong>.
          </p>
        </div>

        {/* Filter */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500 font-medium">Filter Algorithm:</span>
          <select
            value={selectedAlgo}
            onChange={(e) => setSelectedAlgo(e.target.value)}
            className="rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-800 shadow-xs focus:outline-none"
          >
            <option value="all">All 3 Algorithms</option>
            <option value="reinforce">REINFORCE (Policy Gradient)</option>
            <option value="dqn">DQN (Value-Based)</option>
            <option value="ppo">PPO (Actor-Critic)</option>
          </select>
        </div>
      </div>

      {/* Model Benchmark Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {filteredModels.map((model) => (
          <div key={model.algorithm} className="card-minimal p-6 flex flex-col justify-between space-y-4">
            <div>
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <span className="text-lg font-bold text-slate-900 tracking-tight">{model.algorithm}</span>
                <span className="text-[10px] uppercase font-bold text-blue-700 bg-blue-50 px-2.5 py-0.5 rounded-full border border-blue-200/60">
                  {model.category}
                </span>
              </div>

              <div className="space-y-3 pt-3 text-xs">
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Neural Architecture:</span>
                  <strong className="text-slate-800 font-medium">{model.architecture}</strong>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Trainable Weights:</span>
                  <span className="font-mono font-bold text-slate-900">{model.parameters.toLocaleString()} params</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Asymptotic Peak Reward:</span>
                  <span className="font-mono font-bold text-emerald-700 text-sm">
                    {model.asymptoticReward > 0 ? `+${model.asymptoticReward}` : model.asymptoticReward}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Sample Efficiency (&gt;-10):</span>
                  <span className="font-mono text-slate-800">{model.sampleEfficiencyEpisodes.toLocaleString()} episodes</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Forward Latency:</span>
                  <span className="font-mono font-bold text-slate-800">{model.inferenceLatencyMs} ms / frame</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Host Memory Footprint:</span>
                  <span className="font-mono text-slate-800">{model.memoryUsageMb} MB</span>
                </div>
              </div>

              {/* Pros & Cons */}
              <div className="pt-4 border-t border-slate-100 space-y-2 text-[11px]">
                <div className="text-emerald-700 space-y-1">
                  {model.pros.map((pro, i) => (
                    <div key={i} className="flex items-start gap-1.5">
                      <CheckCircle2 className="h-3 w-3 shrink-0 mt-0.5" />
                      <span>{pro}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-100">
              <span className="inline-block text-[10px] font-semibold text-slate-500 bg-slate-50 border border-slate-200 px-2.5 py-1 rounded-md w-full text-center">
                Status: {model.status}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Comparison Visual Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Peak Reward Chart */}
        <div className="card-minimal p-6 space-y-3">
          <div className="border-b border-slate-100 pb-3">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-900">
              Asymptotic Peak Reward Comparison
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">Maximum sustained moving average reward in ALE/Pong-v5</p>
          </div>
          <div className="h-56 w-full pt-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 15, right: 15, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="algorithm" tick={{ fontSize: 11, fill: '#64748b' }} stroke="#e2e8f0" />
                <YAxis domain={[-3, 3]} tick={{ fontSize: 11, fill: '#64748b' }} stroke="#e2e8f0" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px', fontSize: '11px' }}
                  formatter={(val: any) => [`${val} Reward`, 'Asymptotic Peak']}
                />
                <Bar dataKey="peakReward" fill="#10b981" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Inference Latency Chart */}
        <div className="card-minimal p-6 space-y-3">
          <div className="border-b border-slate-100 pb-3">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-900">
              Forward Inference Latency (ms / frame)
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">Lower latency allows higher real-time frame rates in live gameplay</p>
          </div>
          <div className="h-56 w-full pt-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 15, right: 15, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="algorithm" tick={{ fontSize: 11, fill: '#64748b' }} stroke="#e2e8f0" />
                <YAxis domain={[0, 5]} tick={{ fontSize: 11, fill: '#64748b' }} stroke="#e2e8f0" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px', fontSize: '11px' }}
                  formatter={(val: any) => [`${val} ms`, 'Inference Latency']}
                />
                <Bar dataKey="latency" fill="#2563eb" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Model Checkpoint Inventory Table */}
      <div className="card-minimal p-6 space-y-4">
        <div className="border-b border-slate-100 pb-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-900">
            Available Weights & Checkpoints Inventory
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">Physical weights stored in the project directory</p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200/80">
              <tr>
                <th className="py-2.5 px-3">File Name</th>
                <th className="py-2.5 px-3">Format</th>
                <th className="py-2.5 px-3">Architecture</th>
                <th className="py-2.5 px-3">Episodes</th>
                <th className="py-2.5 px-3">Peak Reward</th>
                <th className="py-2.5 px-3">File Size</th>
                <th className="py-2.5 px-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700 font-medium">
              {CHECKPOINTS_INVENTORY.map((ckpt) => (
                <tr key={ckpt.name} className="hover:bg-slate-50/50">
                  <td className="py-2.5 px-3 font-mono font-bold text-slate-900">{ckpt.name}</td>
                  <td className="py-2.5 px-3 text-slate-500">{ckpt.format}</td>
                  <td className="py-2.5 px-3">{ckpt.architecture}</td>
                  <td className="py-2.5 px-3 font-mono">{ckpt.episodes.toLocaleString()}</td>
                  <td className="py-2.5 px-3 font-mono font-bold text-emerald-700">{ckpt.peakReward}</td>
                  <td className="py-2.5 px-3 font-mono text-slate-500">{ckpt.sizeMb} MB</td>
                  <td className="py-2.5 px-3">
                    <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
                      <CheckCircle2 className="h-3 w-3" /> Compatible
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
