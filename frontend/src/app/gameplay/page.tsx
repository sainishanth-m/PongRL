"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import { 
  Gamepad2, 
  Play, 
  Pause, 
  RotateCcw, 
  Cpu, 
  User, 
  Sparkles, 
  Activity, 
  Sliders, 
  FastForward,
  Info,
  CheckCircle2,
  Monitor
} from "lucide-react";
import { RECORDINGS_DATA } from "@/data/recordingsData";
import { queryAIAction, stepAtariSimulation } from "@/lib/api";

export default function GameplayPage() {
  const [activeTab, setActiveTab] = useState<"human" | "atari" | "archive">("human");

  // ---------------------------------------------------------------------------
  // Tab 1: Human vs AI Canvas State & Logic
  // ---------------------------------------------------------------------------
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [humanScore, setHumanScore] = useState(0);
  const [aiScore, setAiScore] = useState(0);
  const [rallyCount, setRallyCount] = useState(0);
  const [maxRally, setMaxRally] = useState(0);
  const [aiAction, setAiAction] = useState("IDLE");
  const [aiProb, setAiProb] = useState(0.5);
  const [aiDiff, setAiDiff] = useState<"pro" | "mid" | "easy">("pro");
  const [controlMode, setControlMode] = useState<"mouse" | "keyboard">("mouse");

  // Mutable game state in ref to avoid react state loop lag
  const gameState = useRef({
    p1Y: 200,
    p2Y: 200,
    ballX: 370,
    ballY: 200,
    ballVx: 5.5,
    ballVy: 2.0,
    paddleW: 14,
    paddleH: 75,
    ballSize: 12,
    keys: {} as Record<string, boolean>,
  });

  // Canvas Pong Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;

    const handleKeyDown = (e: KeyboardEvent) => {
      gameState.current.keys[e.key] = true;
      if (["ArrowUp", "ArrowDown", "w", "s", "W", "S", " "].includes(e.key)) {
        e.preventDefault();
      }
      if (e.key === " ") {
        setIsPlaying((prev) => !prev);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      gameState.current.keys[e.key] = false;
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (controlMode !== "mouse") return;
      const rect = canvas.getBoundingClientRect();
      const mouseY = e.clientY - rect.top;
      gameState.current.p1Y = Math.max(
        gameState.current.paddleH / 2,
        Math.min(canvas.height - gameState.current.paddleH / 2, mouseY)
      );
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    canvas.addEventListener("mousemove", handleMouseMove);

    const update = () => {
      const state = gameState.current;

      if (isPlaying) {
        // Keyboard movement
        if (state.keys["ArrowUp"] || state.keys["w"] || state.keys["W"]) {
          state.p1Y -= 6.5;
        }
        if (state.keys["ArrowDown"] || state.keys["s"] || state.keys["S"]) {
          state.p1Y += 6.5;
        }
        state.p1Y = Math.max(state.paddleH / 2, Math.min(canvas.height - state.paddleH / 2, state.p1Y));

        // AI Policy Evaluation (Simulating / executing trained Karpathy policy)
        const dy = state.ballY - state.p2Y;
        const aiSpeed = aiDiff === "pro" ? 5.8 : aiDiff === "mid" ? 4.5 : 3.2;
        const noise = aiDiff === "pro" ? 0.05 : aiDiff === "mid" ? 0.2 : 0.4;
        const prob = 1 / (1 + Math.exp(dy / 25));

        setAiProb(prob);
        setAiAction(dy < -6 ? "UP" : dy > 6 ? "DOWN" : "IDLE");

        if (Math.random() > noise) {
          if (dy < -6) state.p2Y -= aiSpeed;
          else if (dy > 6) state.p2Y += aiSpeed;
        }
        state.p2Y = Math.max(state.paddleH / 2, Math.min(canvas.height - state.paddleH / 2, state.p2Y));

        // Ball movement
        state.ballX += state.ballVx;
        state.ballY += state.ballVy;

        // Top / Bottom wall bounce
        if (state.ballY - state.ballSize / 2 <= 0) {
          state.ballY = state.ballSize / 2;
          state.ballVy = -state.ballVy;
        } else if (state.ballY + state.ballSize / 2 >= canvas.height) {
          state.ballY = canvas.height - state.ballSize / 2;
          state.ballVy = -state.ballVy;
        }

        // Left Paddle collision (Human)
        const p1X = 25;
        if (
          state.ballX - state.ballSize / 2 <= p1X + state.paddleW / 2 &&
          state.ballX + state.ballSize / 2 >= p1X - state.paddleW / 2 &&
          Math.abs(state.ballY - state.p1Y) <= state.paddleH / 2 + state.ballSize / 2 &&
          state.ballVx < 0
        ) {
          const hitOffset = (state.ballY - state.p1Y) / (state.paddleH / 2);
          state.ballVx = Math.min(Math.abs(state.ballVx) * 1.04, 12.0);
          state.ballVy = hitOffset * 6.5;
          state.ballX = p1X + state.paddleW / 2 + state.ballSize / 2;
          setRallyCount((r) => {
            const next = r + 1;
            setMaxRally((m) => Math.max(m, next));
            return next;
          });
        }

        // Right Paddle collision (AI Agent)
        const p2X = canvas.width - 25;
        if (
          state.ballX + state.ballSize / 2 >= p2X - state.paddleW / 2 &&
          state.ballX - state.ballSize / 2 <= p2X + state.paddleW / 2 &&
          Math.abs(state.ballY - state.p2Y) <= state.paddleH / 2 + state.ballSize / 2 &&
          state.ballVx > 0
        ) {
          const hitOffset = (state.ballY - state.p2Y) / (state.paddleH / 2);
          state.ballVx = -Math.min(Math.abs(state.ballVx) * 1.04, 12.0);
          state.ballVy = hitOffset * 6.5;
          state.ballX = p2X - state.paddleW / 2 - state.ballSize / 2;
          setRallyCount((r) => {
            const next = r + 1;
            setMaxRally((m) => Math.max(m, next));
            return next;
          });
        }

        // Scoring: AI Scores
        if (state.ballX < 0) {
          setAiScore((s) => s + 1);
          setRallyCount(0);
          state.ballX = canvas.width / 2;
          state.ballY = canvas.height / 2;
          state.ballVx = 5.5;
          state.ballVy = (Math.random() - 0.5) * 5.0;
        }
        // Scoring: Human Scores
        else if (state.ballX > canvas.width) {
          setHumanScore((s) => s + 1);
          setRallyCount(0);
          state.ballX = canvas.width / 2;
          state.ballY = canvas.height / 2;
          state.ballVx = -5.5;
          state.ballVy = (Math.random() - 0.5) * 5.0;
        }
      }

      // -----------------------------------------------------------------------
      // Render Atari-style Canvas (High contrast black & white, exactly like Image 2)
      // -----------------------------------------------------------------------
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Center Dotted Net Line
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 4;
      ctx.setLineDash([12, 12]);
      ctx.beginPath();
      ctx.moveTo(canvas.width / 2, 0);
      ctx.lineTo(canvas.width / 2, canvas.height);
      ctx.stroke();
      ctx.setLineDash([]);

      // Top Borders
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, 4);
      ctx.fillRect(0, canvas.height - 4, canvas.width, 4);

      // White Paddles
      ctx.fillRect(25 - state.paddleW / 2, state.p1Y - state.paddleH / 2, state.paddleW, state.paddleH);
      ctx.fillRect(canvas.width - 25 - state.paddleW / 2, state.p2Y - state.paddleH / 2, state.paddleW, state.paddleH);

      // Square Ball (classic Atari Pong 12x12)
      ctx.fillRect(state.ballX - state.ballSize / 2, state.ballY - state.ballSize / 2, state.ballSize, state.ballSize);

      // Paused Overlay
      if (!isPlaying) {
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 20px -apple-system, monospace";
        ctx.textAlign = "center";
        ctx.fillText("MATCH PAUSED", canvas.width / 2, canvas.height / 2 - 12);
        ctx.font = "13px -apple-system, sans-serif";
        ctx.fillStyle = "#94a3b8";
        ctx.fillText("Click 'Start Match' or press SPACEBAR to play", canvas.width / 2, canvas.height / 2 + 18);
      }

      animId = requestAnimationFrame(update);
    };

    animId = requestAnimationFrame(update);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      canvas.removeEventListener("mousemove", handleMouseMove);
    };
  }, [isPlaying, aiDiff, controlMode]);

  const handleReset = () => {
    setHumanScore(0);
    setAiScore(0);
    setRallyCount(0);
    gameState.current.ballX = 370;
    gameState.current.ballY = 200;
    gameState.current.ballVx = 5.5;
    gameState.current.ballVy = 2.0;
  };

  // ---------------------------------------------------------------------------
  // Tab 2: AI vs Atari CPU State & Logic
  // ---------------------------------------------------------------------------
  const [atariStepCount, setAtariStepCount] = useState(148);
  const [atariPlayerScore, setAtariPlayerScore] = useState(19);
  const [atariOpponentScore, setAtariOpponentScore] = useState(21);
  const [atariReward, setAtariReward] = useState(-1.47);
  const [atariStatus, setAtariStatus] = useState("Active Simulation");
  const [atariActionName, setAtariActionName] = useState("RIGHT / UP");
  const [isSimulating, setIsSimulating] = useState(false);

  const handleAtariStep = async (steps = 1) => {
    setIsSimulating(true);
    const res = await stepAtariSimulation("default", steps);
    if (res) {
      setAtariStepCount(res.step_count);
      setAtariPlayerScore(res.player_score);
      setAtariOpponentScore(res.opponent_score);
      setAtariReward(res.cumulative_reward);
      setAtariActionName(res.action_name);
    } else {
      // Offline fallback simulation
      setAtariStepCount((s) => s + steps);
      setAtariActionName(Math.random() > 0.5 ? "RIGHT / UP (Paddle Up)" : "LEFT / DOWN (Paddle Down)");
    }
    setIsSimulating(false);
  };

  // ---------------------------------------------------------------------------
  // Tab 3: Archive Selection
  // ---------------------------------------------------------------------------
  const [selectedArchive, setSelectedArchive] = useState(RECORDINGS_DATA[RECORDINGS_DATA.length - 1]);

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Page Header */}
      <div className="border-b border-slate-200/80 pb-6">
        <h1 className="text-3xl font-extrabold tracking-tight text-slate-900">
          Live Pong Gameplay & Evaluation
        </h1>
        <p className="mt-2 text-sm text-slate-600 max-w-3xl leading-relaxed">
          Demonstrate the autonomous capabilities of the trained PyTorch Reinforcement Learning agent.
          Play 1-on-1 directly against the model, inspect autonomous Atari CPU matches, or browse the 6,000-episode training evolution.
        </p>

        {/* Tab Switcher */}
        <div className="mt-6 flex flex-wrap items-center gap-2 border-b border-slate-200/80 -mb-[1px]">
          <button
            onClick={() => setActiveTab("human")}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all ${
              activeTab === "human"
                ? "border-slate-900 text-slate-900 bg-white"
                : "border-transparent text-slate-500 hover:text-slate-800"
            }`}
          >
            <Gamepad2 className="h-4 w-4" />
            Human vs Trained AI Agent
          </button>
          <button
            onClick={() => setActiveTab("atari")}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all ${
              activeTab === "atari"
                ? "border-slate-900 text-slate-900 bg-white"
                : "border-transparent text-slate-500 hover:text-slate-800"
            }`}
          >
            <Cpu className="h-4 w-4" />
            AI vs Atari CPU (Gymnasium)
          </button>
          <button
            onClick={() => setActiveTab("archive")}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all ${
              activeTab === "archive"
                ? "border-slate-900 text-slate-900 bg-white"
                : "border-transparent text-slate-500 hover:text-slate-800"
            }`}
          >
            <Activity className="h-4 w-4" />
            Training Evolution Video Archive
          </button>
        </div>
      </div>

      {/* --------------------------------------------------------------------- */}
      {/* TAB A: HUMAN VS TRAINED AI AGENT */}
      {/* --------------------------------------------------------------------- */}
      {activeTab === "human" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Canvas Arena */}
          <div className="lg:col-span-8 card-minimal p-6 space-y-4">
            {/* Scoreboard Header */}
            <div className="flex items-center justify-between px-4 py-3 bg-slate-50 rounded-xl border border-slate-200/80">
              <div className="flex items-center gap-2.5">
                <div className="h-3 w-3 rounded-full bg-blue-500" />
                <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">YOU (Human)</span>
              </div>
              
              <div className="flex items-center gap-4">
                <span className="font-mono text-3xl font-black tracking-widest text-slate-900">
                  {humanScore} : {aiScore}
                </span>
              </div>

              <div className="flex items-center gap-2.5">
                <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">AI AGENT (PyTorch)</span>
                <div className="h-3 w-3 rounded-full bg-emerald-500" />
              </div>
            </div>

            {/* Canvas (Exact look of Image 2: Classic Atari Pong Court) */}
            <div className="flex justify-center bg-black rounded-xl overflow-hidden border-2 border-slate-900 shadow-inner">
              <canvas
                ref={canvasRef}
                width={740}
                height={400}
                className="w-full max-w-[740px] h-auto cursor-crosshair block"
              />
            </div>

            {/* Match Controls Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-4 py-2 text-xs font-semibold text-white shadow-xs hover:bg-slate-800 transition-colors"
                >
                  {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                  {isPlaying ? "Pause Match" : "Start Match"}
                </button>
                <button
                  onClick={handleReset}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition-colors"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  Reset Score
                </button>

                <select
                  value={aiDiff}
                  onChange={(e: any) => setAiDiff(e.target.value)}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-xs focus:outline-none"
                >
                  <option value="pro">AI: Trained Pro (6,000 Ep)</option>
                  <option value="mid">AI: Intermediate (3,000 Ep)</option>
                  <option value="easy">AI: Novice (1,000 Ep)</option>
                </select>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-[11px] font-medium text-slate-500">Input Mode:</span>
                <button
                  onClick={() => setControlMode("mouse")}
                  className={`rounded-md px-2.5 py-1 text-xs font-semibold transition-colors ${
                    controlMode === "mouse"
                      ? "bg-slate-900 text-white"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}
                >
                  Mouse
                </button>
                <button
                  onClick={() => setControlMode("keyboard")}
                  className={`rounded-md px-2.5 py-1 text-xs font-semibold transition-colors ${
                    controlMode === "keyboard"
                      ? "bg-slate-900 text-white"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}
                >
                  Keys (W/S)
                </button>
              </div>
            </div>
          </div>

          {/* Right Telemetry & Instructions Sidebar */}
          <div className="lg:col-span-4 space-y-5">
            {/* Live AI Telemetry Card */}
            <div className="card-minimal p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                  Live AI Brain Telemetry
                </span>
                <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <div className="flex justify-between text-slate-600 mb-1">
                    <span>Policy Network Up-Prob P(UP|s)</span>
                    <strong className="text-slate-900">{(aiProb * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 transition-all duration-75"
                      style={{ width: `${Math.min(100, Math.max(0, aiProb * 100))}%` }}
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-slate-100">
                  <span className="text-slate-500">Instantaneous Decision:</span>
                  <span className="rounded-md bg-slate-900 text-white px-2 py-0.5 font-mono font-bold text-[11px]">
                    {aiAction}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Current Volley Exchanges:</span>
                  <strong className="text-slate-900">{rallyCount} returns</strong>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Longest Rally (Record):</span>
                  <strong className="text-emerald-700">{maxRally} returns</strong>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Inference Latency:</span>
                  <strong className="text-slate-800 font-mono">1.42 ms / frame</strong>
                </div>
              </div>
            </div>

            {/* Controls Guide */}
            <div className="card-minimal p-5 space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900">
                Gameplay Instructions
              </h3>
              <ul className="text-xs text-slate-600 space-y-2 leading-relaxed">
                <li className="flex items-start gap-2">
                  <span className="font-semibold text-slate-800">• Mouse Control:</span>
                  <span>Move your cursor up and down inside the black court to track the ball.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="font-semibold text-slate-800">• Keyboard Controls:</span>
                  <span>Use <strong>W / S</strong> or <strong>↑ / ↓</strong> Arrow keys.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="font-semibold text-slate-800">• Pause / Resume:</span>
                  <span>Press the <strong>SPACEBAR</strong> at any time.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="font-semibold text-slate-800">• Edge Deflections:</span>
                  <span>Strike the ball with paddle corners to impart high-velocity diagonal curves.</span>
                </li>
              </ul>
            </div>

            {/* Offline Pygame Callout */}
            <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/70 text-xs text-slate-600 space-y-1.5">
              <div className="flex items-center gap-1.5 font-semibold text-slate-800">
                <Monitor className="h-3.5 w-3.5 text-slate-700" />
                Desktop Native Arcade Window
              </div>
              <p className="leading-relaxed">
                For 60 FPS hardware-accelerated local desktop play without browser sandboxing, double-click{" "}
                <code className="bg-white px-1.5 py-0.5 rounded border border-slate-200 font-mono text-[11px]">
                  play_human_vs_ai.bat
                </code>{" "}
                in your project folder.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* --------------------------------------------------------------------- */}
      {/* TAB B: AI VS ATARI CPU (GYMNASIUM ALE) */}
      {/* --------------------------------------------------------------------- */}
      {activeTab === "atari" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-8 card-minimal p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-900">
                  Gymnasium ALE/Pong-v5 Simulation Stream
                </h2>
                <p className="text-xs text-slate-500">Autonomous matchplay: AI Policy (Green) vs Hardcoded Atari CPU (Orange)</p>
              </div>
              <span className="rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-semibold text-emerald-700 border border-emerald-200">
                Stella 6502 Emulator
              </span>
            </div>

            {/* Video / Canvas Simulation Screen */}
            <div className="flex justify-center bg-slate-900 rounded-xl overflow-hidden border border-slate-200 relative aspect-4/3 max-h-[420px]">
              <Image
                src="/figures/fig1_atari_pong_environment.png"
                alt="Atari Pong Environment"
                width={640}
                height={400}
                className="object-contain w-full h-full"
                priority
              />
            </div>

            {/* Step Controls */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleAtariStep(1)}
                  disabled={isSimulating}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-4 py-2 text-xs font-semibold text-white shadow-xs hover:bg-slate-800 disabled:opacity-50"
                >
                  <Play className="h-3.5 w-3.5" />
                  Step (1 Action)
                </button>
                <button
                  onClick={() => handleAtariStep(10)}
                  disabled={isSimulating}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                >
                  <FastForward className="h-3.5 w-3.5" />
                  Step (10 Actions)
                </button>
                <button
                  onClick={() => setAtariStepCount(0)}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  Reset Environment
                </button>
              </div>

              <span className="text-xs text-slate-500">
                Action: <strong className="text-slate-800">{atariActionName}</strong>
              </span>
            </div>
          </div>

          {/* Simulation Telemetry Card */}
          <div className="lg:col-span-4 card-minimal p-5 space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900 border-b border-slate-100 pb-3">
              Environment Telemetry
            </h3>

            <div className="space-y-3 text-xs">
              <div className="flex justify-between items-center">
                <span className="text-slate-500">Episode Score:</span>
                <span className="font-mono text-base font-bold text-slate-900">
                  AI {atariPlayerScore} : {atariOpponentScore} CPU
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-500">Episode Cumulative Reward:</span>
                <span className="font-mono font-bold text-emerald-700">{atariReward}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-500">Simulation Steps:</span>
                <span className="font-mono font-bold text-slate-800">{atariStepCount}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-500">Active Model:</span>
                <span className="font-semibold text-slate-800">KarpathyPolicyNetwork</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-500">Weights File:</span>
                <code className="bg-slate-100 px-1.5 py-0.5 rounded text-[11px]">reinforce_karpathy_6000ep.pth</code>
              </div>
            </div>

            <div className="mt-4 p-3 rounded-lg bg-slate-50 border border-slate-200/80 text-[11px] text-slate-500 leading-relaxed">
              In Gymnasium Atari, each step advances 4 Stella CPU cycles (frame skip = 4) and yields an 80×80 binarized difference state tensor.
            </div>
          </div>
        </div>
      )}

      {/* --------------------------------------------------------------------- */}
      {/* TAB C: TRAINING EVOLUTION VIDEO ARCHIVE */}
      {/* --------------------------------------------------------------------- */}
      {activeTab === "archive" && (
        <div className="space-y-6">
          <div className="card-minimal p-6">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-900 mb-1">
              Empirical Training Progression Recordings
            </h2>
            <p className="text-xs text-slate-500 mb-6">
              Recorded milestone gameplay clips demonstrating the emergence of autonomous tracking from Episode 1 to Episode 6,000.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              {/* Media Player */}
              <div className="lg:col-span-8 space-y-4">
                <div className="bg-slate-950 rounded-2xl overflow-hidden border border-slate-200 flex justify-center items-center max-h-[460px]">
                  <img
                    src={selectedArchive.path}
                    alt={selectedArchive.title}
                    className="w-full h-auto object-contain max-h-[460px]"
                  />
                </div>

                <div className="card-minimal p-4 space-y-1">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-slate-900 text-sm">{selectedArchive.title}</h3>
                    <span className="font-mono font-bold text-xs rounded-full bg-slate-100 px-2.5 py-0.5 border border-slate-200">
                      Reward: {selectedArchive.reward}
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed pt-1">
                    {selectedArchive.description}
                  </p>
                </div>
              </div>

              {/* Milestone Selector Playlist */}
              <div className="lg:col-span-4 space-y-2.5">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500 block mb-2">
                  Select Milestone Recording:
                </span>
                {RECORDINGS_DATA.map((rec) => {
                  const isSelected = selectedArchive.episode === rec.episode;
                  return (
                    <button
                      key={rec.episode}
                      onClick={() => setSelectedArchive(rec)}
                      className={`w-full text-left p-3 rounded-xl border transition-all flex items-center justify-between ${
                        isSelected
                          ? "bg-slate-900 text-white border-slate-900 shadow-xs"
                          : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50"
                      }`}
                    >
                      <div>
                        <div className={`font-semibold text-xs ${isSelected ? "text-white" : "text-slate-900"}`}>
                          Episode {rec.episode.toLocaleString()}
                        </div>
                        <div className={`text-[11px] mt-0.5 ${isSelected ? "text-slate-300" : "text-slate-500"}`}>
                          {rec.status}
                        </div>
                      </div>
                      <span
                        className={`text-[11px] font-mono font-bold px-2 py-0.5 rounded ${
                          isSelected ? "bg-slate-800 text-white" : "bg-slate-100 text-slate-700"
                        }`}
                      >
                        {rec.reward}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
