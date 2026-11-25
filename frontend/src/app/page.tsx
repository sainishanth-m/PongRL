"use client";

import Link from "next/link";
import {
  Gamepad2,
  BarChart3,
  Layers,
  GitCompare,
  Award,
  Cpu,
  Zap,
  ArrowRight,
  CheckCircle2,
  TrendingUp,
  Clock,
  Sparkles
} from "lucide-react";
import MetricCard from "@/components/MetricCard";
import { PROJECT_SUMMARY } from "@/data/summary";
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from "recharts";
import { REAL_TRAINING_POINTS } from "@/data/trainingData";

export default function OverviewPage() {
  // Sample every 40 points for the quick summary sparkline
  const sparklineData = REAL_TRAINING_POINTS.filter((_, i) => i % 6 === 0);

  return (
    <div className="space-y-10 animate-in fade-in duration-300">
      {/* Hero Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 border-b border-slate-200/80 pb-8">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-700 shadow-xs mb-3">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            Reinforcement Learning Research Platform
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900">
            {PROJECT_SUMMARY.name}
          </h1>
          <p className="mt-2 text-base text-slate-600 max-w-2xl leading-relaxed">
            {PROJECT_SUMMARY.subtitle}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Link
            href="/gameplay"
            className="inline-flex items-center gap-2 rounded-xl bg-slate-900 px-5 py-2.5 text-xs font-semibold text-white shadow-xs hover:bg-slate-800 transition-colors"
          >
            <Gamepad2 className="h-4 w-4" />
            Play Against AI
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
          <Link
            href="/analytics"
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-5 py-2.5 text-xs font-semibold text-slate-700 shadow-xs hover:bg-slate-50 transition-colors"
          >
            <BarChart3 className="h-4 w-4" />
            View Convergence
          </Link>
        </div>
      </div>

      {/* KPI Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Total Training Episodes"
          value={PROJECT_SUMMARY.recordedEpisodes.toLocaleString()}
          change="Full Convergence"
          changeType="positive"
          subtitle="6,001 consecutive match episodes logged in ALE/Pong-v5"
          icon={Clock}
        />
        <MetricCard
          label="Peak Moving Reward"
          value={PROJECT_SUMMARY.peakRunningReward.toFixed(2)}
          change="+18.53 above baseline"
          changeType="positive"
          subtitle="Converged from -20.0 to near parity with Atari CPU"
          icon={TrendingUp}
        />
        <MetricCard
          label="Active Model Checkpoints"
          value={PROJECT_SUMMARY.activeModels}
          change="3 Architectures"
          changeType="neutral"
          subtitle="REINFORCE (Karpathy), Double DQN, and PPO benchmark"
          icon={Layers}
        />
        <MetricCard
          label="Training Environment"
          value="ALE/Pong-v5"
          change="Stella 60 FPS"
          changeType="positive"
          subtitle="Cycle-accurate Atari 2600 machine emulation via Gymnasium"
          icon={Zap}
        />
      </div>

      {/* Executive Summary & Real Learning Curve Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Executive Summary */}
        <div className="lg:col-span-5 card-minimal p-6 space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <Award className="h-4 w-4 text-slate-700" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-900">
              Executive Summary
            </h2>
          </div>

          <p className="text-sm text-slate-600 leading-relaxed">
            {PROJECT_SUMMARY.description}
          </p>

          <div className="space-y-2.5 pt-2">
            <div className="flex items-start gap-2.5 text-xs text-slate-700">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
              <span><strong>End-to-End Visual Perception:</strong> Learns strictly from raw pixels via 5-stage temporal difference preprocessing.</span>
            </div>
            <div className="flex items-start gap-2.5 text-xs text-slate-700">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
              <span><strong>Memory Footprint Reduction:</strong> Quantized uint8 circular replay buffer cuts host memory usage by <strong>93.75%</strong> (22.5 GB → 1.41 GB).</span>
            </div>
            <div className="flex items-start gap-2.5 text-xs text-slate-700">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
              <span><strong>Empirical Asymptotic Convergence:</strong> Sustains 18+ return rallies, achieving a 48.8% win rate against the hardcoded Atari CPU.</span>
            </div>
            <div className="flex items-start gap-2.5 text-xs text-slate-700">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
              <span><strong>Dual Deployment Strategy:</strong> Zero-latency interactive in-browser HTML5 Canvas court coupled with FastAPI inference backend.</span>
            </div>
          </div>
        </div>

        {/* Right Column: Mini Convergence Chart */}
        <div className="lg:col-span-7 card-minimal p-6 space-y-3">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-900">
                Empirical Convergence Trajectory
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">Real 6,001-Episode moving average rewards from -20.4 to -1.47</p>
            </div>
            <Link
              href="/analytics"
              className="text-xs font-semibold text-blue-600 hover:text-blue-700 flex items-center gap-1"
            >
              Full Analytics <ArrowRight className="h-3 w-3" />
            </Link>
          </div>

          <div className="h-64 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={sparklineData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="gradientReward" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="episode" tick={{ fontSize: 10, fill: '#64748b' }} stroke="#e2e8f0" />
                <YAxis domain={[-21, 0]} tick={{ fontSize: 10, fill: '#64748b' }} stroke="#e2e8f0" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    fontSize: '11px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)'
                  }}
                  formatter={(val: any) => [`${val} Reward`, 'Moving Average']}
                  labelFormatter={(label) => `Episode ${label}`}
                />
                <Area
                  type="monotone"
                  dataKey="ma50"
                  stroke="#10b981"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#gradientReward)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="flex items-center justify-between text-xs text-slate-500 pt-2 border-t border-slate-100">
            <span>Baseline: <strong className="text-slate-800">-20.00</strong> (Ep 0)</span>
            <span>Inflection: <strong className="text-slate-800">-11.20</strong> (Ep 3000)</span>
            <span>Peak: <strong className="text-emerald-700 font-bold">-1.47</strong> (Ep 6000)</span>
          </div>
        </div>
      </div>

      {/* Technology Stack Grid */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Cpu className="h-4 w-4 text-slate-700" />
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-900">
            System Technologies & Engineering Stack
          </h2>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {PROJECT_SUMMARY.techStack.map((tech) => (
            <div key={tech.name} className="card-minimal p-4">
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-semibold text-slate-900 text-sm">{tech.name}</span>
                <span className="text-[10px] uppercase font-semibold text-slate-500 bg-slate-100 px-2 py-0.5 rounded-full border border-slate-200">
                  {tech.category}
                </span>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">
                {tech.description}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Access Navigation Tiles */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 pt-2">
        <Link
          href="/gameplay"
          className="card-minimal p-6 group hover:border-slate-300 transition-all flex flex-col justify-between"
        >
          <div>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50 text-blue-600 mb-4 border border-blue-100">
              <Gamepad2 className="h-5 w-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-base group-hover:text-blue-600 transition-colors">
              Live Pong Gameplay
            </h3>
            <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
              Play 1-on-1 directly against the 6,000-ep trained PyTorch agent in your browser, or observe autonomous Gymnasium Atari matches.
            </p>
          </div>
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-600 mt-4">
            Launch Game Court <ArrowRight className="h-3.5 w-3.5 group-hover:translate-x-1 transition-transform" />
          </span>
        </Link>

        <Link
          href="/comparison"
          className="card-minimal p-6 group hover:border-slate-300 transition-all flex flex-col justify-between"
        >
          <div>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-50 text-purple-600 mb-4 border border-purple-100">
              <GitCompare className="h-5 w-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-base group-hover:text-purple-600 transition-colors">
              Algorithm Comparison
            </h3>
            <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
              Compare REINFORCE, Deep Q-Networks (DQN), and Proximal Policy Optimization (PPO) across parameters, sample efficiency, and latency.
            </p>
          </div>
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-purple-600 mt-4">
            Inspect Benchmarks <ArrowRight className="h-3.5 w-3.5 group-hover:translate-x-1 transition-transform" />
          </span>
        </Link>

        <Link
          href="/architecture"
          className="card-minimal p-6 group hover:border-slate-300 transition-all flex flex-col justify-between"
        >
          <div>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600 mb-4 border border-emerald-100">
              <Layers className="h-5 w-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-base group-hover:text-emerald-600 transition-colors">
              Model Architecture
            </h3>
            <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
              Explore the 10-step end-to-end RL workflow, Nature CNN tensor shapes, and the 2-layer Karpathy Policy MLP network specifications.
            </p>
          </div>
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-600 mt-4">
            View Layer Specs <ArrowRight className="h-3.5 w-3.5 group-hover:translate-x-1 transition-transform" />
          </span>
        </Link>
      </div>
    </div>
  );
}
