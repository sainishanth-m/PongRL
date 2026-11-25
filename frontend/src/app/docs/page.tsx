"use client";

import React, { useState } from "react";
import Link from "next/link";
import { 
  Search, 
  BookOpen, 
  ArrowRight, 
  ExternalLink, 
  Layers, 
  Cpu, 
  Database, 
  Activity, 
  CheckCircle2, 
  Copy, 
  Check, 
  Code2, 
  Sliders, 
  FileText, 
  Terminal, 
  Sparkles,
  ChevronRight
} from "lucide-react";

interface DocNavSection {
  title: string;
  items: { id: string; label: string }[];
}

const DOC_SECTIONS: DocNavSection[] = [
  {
    title: "Overview",
    items: [
      { id: "intro", label: "Introduction" },
      { id: "how-it-works", label: "How PongRL works" },
      { id: "quickstart", label: "Quickstart" },
    ],
  },
  {
    title: "Core Foundations",
    items: [
      { id: "env-setup", label: "ALE Atari 2600 Court" },
      { id: "frame-diff", label: "Temporal Differencing (Δst)" },
      { id: "nature-cnn", label: "Nature CNN vs MLP" },
    ],
  },
  {
    title: "Algorithms",
    items: [
      { id: "reinforce", label: "REINFORCE (Karpathy)" },
      { id: "dqn", label: "Deep Q-Networks (DQN)" },
      { id: "ppo", label: "Proximal Policy Opt (PPO)" },
    ],
  },
  {
    title: "Systems & Memory",
    items: [
      { id: "uint8-buffer", label: "uint8 Replay Compression" },
      { id: "convergence", label: "Empirical Convergence" },
      { id: "latency", label: "Inference Latency" },
    ],
  },
  {
    title: "Self-hosting & APIs",
    items: [
      { id: "fastapi", label: "FastAPI REST API" },
      { id: "vercel", label: "Vercel Deployment" },
      { id: "cli-commands", label: "CLI & Scripts" },
    ],
  },
  {
    title: "Academic & Viva",
    items: [
      { id: "viva-qa", label: "Viva Voce Defense Q&A" },
      { id: "references", label: "IEEE References" },
    ],
  },
];

const ON_THIS_PAGE_LINKS = [
  { id: "how-it-works", label: "How PongRL works" },
  { id: "quickstart", label: "Get started" },
  { id: "algorithms-overview", label: "Supported algorithms" },
  { id: "frame-diff-detail", label: "Frame differencing" },
  { id: "memory-savings", label: "Memory compression" },
  { id: "self-hosting", label: "Self-hosting" },
  { id: "defense", label: "Academic defense" },
  { id: "resources", label: "More resources" },
];

export default function DocsPage() {
  const [activeItem, setActiveItem] = useState("intro");
  const [searchQuery, setSearchQuery] = useState("");
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div className="flex gap-8 max-w-7xl mx-auto pt-2 pb-16">
      {/* ============================================================== */}
      {/* 1. LEFT SIDEBAR: DOCUMENTATION NAVIGATION                       */}
      {/* ============================================================== */}
      <aside className="hidden lg:block w-64 flex-shrink-0 sticky top-24 h-[calc(100vh-7rem)] overflow-y-auto pr-4 text-sm border-r border-slate-200/80">
        {/* Search input with shortcut badge (matching screenshot) */}
        <div className="relative mb-6">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search the docs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-8 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:bg-white focus:border-slate-400 transition-colors"
          />
          <kbd className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-200/70 text-slate-500">
            ⌘K
          </kbd>
        </div>

        {/* Navigation Categories */}
        <div className="space-y-6">
          {DOC_SECTIONS.map((section, sIdx) => {
            const filteredItems = section.items.filter((item) =>
              item.label.toLowerCase().includes(searchQuery.toLowerCase())
            );
            if (filteredItems.length === 0) return null;

            return (
              <div key={sIdx} className="space-y-1.5">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-900">
                  {section.title}
                </h3>
                <ul className="space-y-0.5">
                  {filteredItems.map((item) => {
                    const isActive = activeItem === item.id;
                    return (
                      <li key={item.id}>
                        <button
                          onClick={() => {
                            setActiveItem(item.id);
                            const el = document.getElementById(item.id);
                            if (el) el.scrollIntoView({ behavior: "smooth" });
                          }}
                          className={`w-full text-left px-2.5 py-1.5 rounded-lg text-xs transition-colors flex items-center justify-between ${
                            isActive
                              ? "bg-slate-100 text-slate-950 font-semibold"
                              : "text-slate-600 hover:text-slate-950 hover:bg-slate-50"
                          }`}
                        >
                          <span>{item.label}</span>
                          {isActive && <ChevronRight className="w-3 h-3 text-slate-500" />}
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            );
          })}
        </div>
      </aside>

      {/* ============================================================== */}
      {/* 2. CENTER CONTENT: MAIN DOCUMENTATION                          */}
      {/* ============================================================== */}
      <main className="flex-1 min-w-0 max-w-3xl space-y-12">
        {/* Breadcrumb (matching screenshot: 📖 VeriRag documentation) */}
        <div>
          <div className="flex items-center gap-2 text-xs font-medium text-slate-500 mb-3">
            <BookOpen className="w-4 h-4 text-slate-600" />
            <span>PongRL Pro documentation</span>
          </div>

          {/* Large Title (matching screenshot: Introduction) */}
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-slate-950">
            Introduction
          </h1>

          {/* Lead paragraph */}
          <p className="mt-4 text-base sm:text-lg text-slate-600 leading-relaxed font-normal">
            PongRL Pro transforms raw 210×160 RGB Atari 2600 video frames into an autonomous, competitive game-playing agent using Deep Reinforcement Learning. Ask technical questions, inspect PyTorch architectures, and explore the mathematical foundations of visual policy acquisition.
          </p>

          {/* Hero Action Buttons (matching screenshot: [Open workspace ->] [Upload documents ->]) */}
          <div className="flex flex-wrap items-center gap-3 mt-6">
            <Link
              href="/gameplay"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-slate-950 text-white text-xs font-semibold hover:bg-slate-800 transition-all shadow-xs active:scale-[0.98]"
            >
              <span>Play live match</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
            <Link
              href="/architecture"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-white border border-slate-200 text-slate-800 text-xs font-semibold hover:bg-slate-50 transition-all shadow-2xs active:scale-[0.98]"
            >
              <span>Explore architecture</span>
              <ArrowRight className="w-3.5 h-3.5 text-slate-400" />
            </Link>
          </div>
        </div>

        {/* Section 1: Turn visual pixels into optimal paddle velocities (matching screenshot headline style) */}
        <section id="how-it-works" className="space-y-4 pt-4 border-t border-slate-200">
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-950">
            Turn visual pixels into optimal paddle velocities
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            Standard supervised learning requires massive datasets of human gameplay demonstrations.
            PongRL Pro operates under the <strong>Markov Decision Process (MDP)</strong> framework where the agent learns purely through trial and error, receiving sparse scalar rewards (&#123;-1, 0, +1&#125;) when a point is won or conceded.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
            <div className="p-4 rounded-xl border border-slate-200 bg-white">
              <span className="text-[11px] font-bold uppercase tracking-wider text-blue-600">Observation</span>
              <h3 className="text-sm font-semibold text-slate-900 mt-1">210×160 RGB Frames</h3>
              <p className="text-xs text-slate-500 mt-1">100,800 raw color pixels emitted at 60 FPS by the Stella 2600 engine.</p>
            </div>
            <div className="p-4 rounded-xl border border-slate-200 bg-white">
              <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-600">State Vector</span>
              <h3 className="text-sm font-semibold text-slate-900 mt-1">80×80 Difference</h3>
              <p className="text-xs text-slate-500 mt-1">Temporal differencing Δst = st - st-1 directly encodes ball velocity.</p>
            </div>
            <div className="p-4 rounded-xl border border-slate-200 bg-white">
              <span className="text-[11px] font-bold uppercase tracking-wider text-purple-600">Action Space</span>
              <h3 className="text-sm font-semibold text-slate-900 mt-1">{`{ UP, DOWN, IDLE }`}</h3>
              <p className="text-xs text-slate-500 mt-1">Discrete joystick commands executed with a 4-frame action repeat.</p>
            </div>
          </div>
        </section>

        {/* Section 2: Quickstart */}
        <section id="quickstart" className="space-y-4 pt-4 border-t border-slate-200">
          <h2 className="text-xl sm:text-2xl font-bold text-slate-950">
            Quickstart: Running Locally
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            You can launch the complete full-stack environment with a single command or double-click:
          </p>

          <div className="bg-slate-950 text-slate-200 rounded-xl p-4 font-mono text-xs relative">
            <div className="flex items-center justify-between text-slate-400 border-b border-slate-800 pb-2 mb-3">
              <span>Terminal (Powershell / CMD)</span>
              <button
                onClick={() => copyToClipboard(".\\start_website.bat", "run-bat")}
                className="flex items-center gap-1 hover:text-white"
              >
                {copiedCode === "run-bat" ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copiedCode === "run-bat" ? "Copied" : "Copy"}</span>
              </button>
            </div>
            <pre className="text-emerald-400 font-mono">
              .\\start_website.bat
            </pre>
            <p className="text-slate-400 text-[11px] mt-2">
              This starts the FastAPI Python server on port 8000 and the Next.js frontend on port 3000, then automatically opens your browser.
            </p>
          </div>
        </section>

        {/* Section 3: Supported Algorithms */}
        <section id="algorithms-overview" className="space-y-4 pt-4 border-t border-slate-200">
          <h2 className="text-xl sm:text-2xl font-bold text-slate-950">
            Implemented Algorithms
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            The project provides clean, modularized PyTorch implementations of the three landmark algorithms in Deep RL:
          </p>

          <div className="space-y-3">
            <div className="p-4 rounded-xl border border-slate-200 bg-white">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-slate-900">1. REINFORCE (Policy Gradient)</h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-50 text-blue-700 font-semibold">Active Run (6,001 Eps)</span>
              </div>
              <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                Direct policy search optimizing expected cumulative returns with standardized rewards. Reaches a peak 100-episode moving average reward of -1.47.
              </p>
            </div>

            <div className="p-4 rounded-xl border border-slate-200 bg-white">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-slate-900">2. Deep Q-Networks (DQN)</h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-semibold">Nature CNN</span>
              </div>
              <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                Off-policy temporal difference Q-learning using experience replay buffer (100k capacity) and target network synchronization every 1,000 steps.
              </p>
            </div>

            <div className="p-4 rounded-xl border border-slate-200 bg-white">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-slate-900">3. Proximal Policy Optimization (PPO)</h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-semibold">Actor-Critic</span>
              </div>
              <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                Clipped surrogate objective with Generalized Advantage Estimation (GAE λ = 0.95), preventing catastrophic policy collapse.
              </p>
            </div>
          </div>
        </section>

        {/* Section 4: Frame Differencing */}
        <section id="frame-diff-detail" className="space-y-4 pt-4 border-t border-slate-200">
          <h2 className="text-xl sm:text-2xl font-bold text-slate-950">
            Temporal Frame Differencing (Δst)
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            In Atari Pong, a single static frame cannot reveal whether the ball is traveling toward the agent or away from it. Rather than maintaining a memory-intensive 4-frame queue, we compute the difference between the current frame and previous frame:
          </p>

          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 font-mono text-xs text-slate-800 text-center">
            Δs_t = Current_Frame(t) - Previous_Frame(t - 1)
          </div>

          <p className="text-xs text-slate-600 leading-relaxed">
            In the resulting binary matrix, $+1$ marks the current ball position and $-1$ marks its previous position. The spatial displacement between $+1$ and $-1$ directly supplies the 2-layer MLP with the velocity vector $(v_x, v_y)$.
          </p>
        </section>

        {/* Section 5: Memory Savings */}
        <section id="memory-savings" className="space-y-4 pt-4 border-t border-slate-200">
          <h2 className="text-xl sm:text-2xl font-bold text-slate-950">
            uint8 Replay Buffer Compression (93.75% Savings)
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            Storing 100,000 observations naively as standard PyTorch 32-bit floats consumes:
          </p>

          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs font-mono text-slate-700 space-y-1">
            <p>100,000 transitions × 4 frames × 84 × 84 pixels × 4 bytes = 11.29 GB (State) + 11.29 GB (Next State) = 22.58 GB RAM</p>
            <p className="text-emerald-700 font-bold pt-1">
              With uint8 quantization (1 byte / pixel): 1.41 GB RAM total (93.75% memory compression).
            </p>
          </div>
        </section>

        {/* Section 6: Self-hosting & API */}
        <section id="self-hosting" className="space-y-4 pt-4 border-t border-slate-200">
          <h2 className="text-xl sm:text-2xl font-bold text-slate-950">
            FastAPI REST Endpoints
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            The Python backend exposes interactive endpoints for telemetry and real-time inference:
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 uppercase font-semibold">
                  <th className="py-2 px-3">Method</th>
                  <th className="py-2 px-3">Endpoint</th>
                  <th className="py-2 px-3">Description</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                <tr>
                  <td className="py-2 px-3 font-mono font-bold text-emerald-600">GET</td>
                  <td className="py-2 px-3 font-mono text-slate-900">/api/summary</td>
                  <td className="py-2 px-3">Project KPIs, algorithm configurations, and hardware stats</td>
                </tr>
                <tr>
                  <td className="py-2 px-3 font-mono font-bold text-emerald-600">GET</td>
                  <td className="py-2 px-3 font-mono text-slate-900">/api/metrics/training</td>
                  <td className="py-2 px-3">Full 6,001-episode training curve and moving averages</td>
                </tr>
                <tr>
                  <td className="py-2 px-3 font-mono font-bold text-blue-600">POST</td>
                  <td className="py-2 px-3 font-mono text-slate-900">/api/game/action</td>
                  <td className="py-2 px-3">Real-time PyTorch forward pass returning action and P(UP) probability</td>
                </tr>
                <tr>
                  <td className="py-2 px-3 font-mono font-bold text-blue-600">POST</td>
                  <td className="py-2 px-3 font-mono text-slate-900">/api/game/session/step</td>
                  <td className="py-2 px-3">Steps the headless ALE/Pong-v5 Gymnasium environment</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Section 7: Academic Defense */}
        <section id="defense" className="space-y-4 pt-4 border-t border-slate-200">
          <h2 className="text-xl sm:text-2xl font-bold text-slate-950">
            Viva Voce Defense Preparation
          </h2>
          <div className="p-4 rounded-xl border border-slate-200 bg-white space-y-2">
            <h3 className="text-xs font-bold text-slate-900">Q: Why did REINFORCE plateau at -1.47 instead of +21?</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              In Atari Pong, games are played to 21 points. A reward of -1.47 means the agent won approximately 48.8% of individual rallies against the built-in heuristic opponent, producing match scores like 19–21 or 20–21 with sustained 18+ return rallies.
            </p>
          </div>
        </section>

        {/* Section 8: More Resources */}
        <section id="resources" className="pt-6 border-t border-slate-200 flex flex-col sm:flex-row gap-4">
          <Link
            href="/comparison"
            className="flex-1 p-4 rounded-xl border border-slate-200 bg-white hover:border-slate-300 transition-colors"
          >
            <span className="text-xs text-blue-600 font-semibold">Next Section</span>
            <h4 className="text-sm font-bold text-slate-900 mt-1">Algorithm Comparison →</h4>
            <p className="text-xs text-slate-500 mt-1">Review REINFORCE vs DQN vs PPO benchmarks.</p>
          </Link>
          <Link
            href="/info"
            className="flex-1 p-4 rounded-xl border border-slate-200 bg-white hover:border-slate-300 transition-colors"
          >
            <span className="text-xs text-purple-600 font-semibold">Capstone Dossier</span>
            <h4 className="text-sm font-bold text-slate-900 mt-1">Full Viva Q&A & IEEE Citations →</h4>
            <p className="text-xs text-slate-500 mt-1">Access thesis formulations and references.</p>
          </Link>
        </section>
      </main>

      {/* ============================================================== */}
      {/* 3. RIGHT SIDEBAR: "ON THIS PAGE" (MATCHING SCREENSHOT)         */}
      {/* ============================================================== */}
      <aside className="hidden xl:block w-52 flex-shrink-0 sticky top-24 h-[calc(100vh-7rem)] overflow-y-auto text-xs pl-2">
        <h4 className="font-semibold text-slate-900 mb-3 uppercase tracking-wider text-[11px]">
          On this page
        </h4>
        <ul className="space-y-2 border-l border-slate-200 pl-3">
          {ON_THIS_PAGE_LINKS.map((link) => (
            <li key={link.id}>
              <a
                href={`#${link.id}`}
                className="text-slate-500 hover:text-slate-950 transition-colors block py-0.5 leading-snug"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  );
}
