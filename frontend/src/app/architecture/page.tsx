"use client";

import React, { useState } from "react";
import { 
  Cpu, 
  Layers, 
  ArrowRight, 
  CheckCircle2, 
  Sliders, 
  Binary, 
  Eye, 
  Database,
  Activity,
  Zap,
  Sparkles
} from "lucide-react";
import { NATURE_CNN_LAYERS, KARPATHY_MLP_LAYERS, WORKFLOW_STAGES } from "@/data/architectureData";

export default function ArchitecturePage() {
  const [activeTab, setActiveTab] = useState<"pipeline" | "cnn" | "mlp" | "math">("pipeline");

  const totalCnnParams = NATURE_CNN_LAYERS.reduce((acc, curr) => acc + curr.parameters, 0);
  const totalMlpParams = KARPATHY_MLP_LAYERS.reduce((acc, curr) => acc + curr.parameters, 0);

  return (
    <div className="space-y-10 max-w-6xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-200 pb-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-slate-100 text-slate-700 rounded-full text-xs font-semibold mb-3">
          <Layers className="w-3.5 h-3.5 text-blue-600" />
          <span>Deep Neural Network Topologies</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">
          Model Architecture & Pipeline
        </h1>
        <p className="text-slate-500 mt-2 text-base max-w-3xl">
          Detailed architectural breakdown of the 5-stage computer vision preprocessing pipeline,
          Mnih et al. Nature CNN, Andrej Karpathy Policy MLP, and rigorous mathematical formulations.
        </p>

        {/* Tab Navigation */}
        <div className="flex gap-2 mt-6 border-b border-slate-200">
          {[
            { id: "pipeline", label: "10-Step RL Pipeline", icon: Activity },
            { id: "cnn", label: "Nature CNN (DQN / PPO)", icon: Cpu },
            { id: "mlp", label: "Policy MLP (REINFORCE)", icon: Binary },
            { id: "math", label: "Mathematical Formulations", icon: Zap },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as typeof activeTab)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-all -mb-px ${
                  isActive
                    ? "border-blue-600 text-blue-600"
                    : "border-transparent text-slate-500 hover:text-slate-800 hover:border-slate-300"
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* TAB 1: 10-STEP PIPELINE */}
      {activeTab === "pipeline" && (
        <div className="space-y-8">
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
            <h2 className="text-lg font-semibold text-slate-900 mb-2">
              End-to-End Reinforcement Learning Pipeline
            </h2>
            <p className="text-sm text-slate-500 mb-6">
              How raw emulator memory transforms into discrete Atari controller signals in 16.6 milliseconds:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {WORKFLOW_STAGES.map((stage) => (
                <div
                  key={stage.step}
                  className="flex items-start gap-4 p-4 rounded-xl border border-slate-100 bg-slate-50/60 hover:bg-slate-50 transition-colors"
                >
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-50 text-blue-600 border border-blue-200 flex items-center justify-center font-bold text-xs">
                    {stage.step}
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900">{stage.title}</h3>
                    <p className="text-xs text-slate-600 mt-1 leading-relaxed">{stage.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Computer Vision Preprocessing Showcase */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Step 1</span>
                <span className="text-xs px-2 py-0.5 bg-slate-100 rounded text-slate-600 font-mono">210×160 RGB</span>
              </div>
              <h3 className="text-sm font-semibold text-slate-900">Raw Atari Frame</h3>
              <p className="text-xs text-slate-500 mt-1 mb-4">
                Full Stella emulator render with 100,800 scalar values, including margins and scoreboard.
              </p>
              <div className="h-32 bg-slate-900 rounded-lg flex items-center justify-center text-slate-500 text-xs font-mono border border-slate-800">
                [ Raw Atari 2600 Frame ]
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Step 2-4</span>
                <span className="text-xs px-2 py-0.5 bg-blue-50 text-blue-700 rounded text-blue-600 font-mono">80×80 Bin</span>
              </div>
              <h3 className="text-sm font-semibold text-slate-900">Arena Crop & Binarize</h3>
              <p className="text-xs text-slate-500 mt-1 mb-4">
                Scoreboard cropped, downscaled 2:1, background masked, paddles thresholded to {`{0, 1}`}.
              </p>
              <div className="h-32 bg-slate-950 rounded-lg flex items-center justify-center text-slate-400 text-xs font-mono border border-slate-800">
                [ 80×80 Cropped Matrix ]
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-blue-600 font-bold">Step 5 (Velocity)</span>
                <span className="text-xs px-2 py-0.5 bg-emerald-50 text-emerald-700 rounded font-mono">Δst Vector</span>
              </div>
              <h3 className="text-sm font-semibold text-slate-900">Frame Differencing</h3>
              <p className="text-xs text-slate-500 mt-1 mb-4">
                Δst = st - st-1 explicitly yields motion vectors (+1 ball current, -1 ball previous).
              </p>
              <div className="h-32 bg-slate-900 rounded-lg flex items-center justify-center text-emerald-400 text-xs font-mono border border-slate-800">
                [ Δst Temporal Vector ]
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: NATURE CNN */}
      {activeTab === "cnn" && (
        <div className="space-y-6">
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">
                  Nature CNN Architecture (Mnih et al., 2015)
                </h2>
                <p className="text-sm text-slate-500 mt-1">
                  Used by Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO).
                </p>
              </div>
              <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-slate-100 rounded-lg text-xs font-mono font-medium text-slate-700">
                <span>Total Parameters:</span>
                <span className="font-bold text-blue-600">{totalCnnParams.toLocaleString()}</span>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-xs font-semibold text-slate-500 uppercase">
                    <th className="py-3 px-4">Layer</th>
                    <th className="py-3 px-4">Type</th>
                    <th className="py-3 px-4">Configuration</th>
                    <th className="py-3 px-4">Output Dimensions</th>
                    <th className="py-3 px-4 text-right">Parameters</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  {NATURE_CNN_LAYERS.map((layer, idx) => (
                    <tr key={idx} className="hover:bg-slate-50/70 transition-colors">
                      <td className="py-3 px-4 font-semibold text-slate-900">{layer.layer}</td>
                      <td className="py-3 px-4">{layer.type}</td>
                      <td className="py-3 px-4 font-mono text-xs text-slate-600">{layer.config}</td>
                      <td className="py-3 px-4 font-mono text-xs text-blue-600 font-semibold">{layer.outputShape}</td>
                      <td className="py-3 px-4 text-right font-mono text-xs text-slate-500">
                        {layer.parameters > 0 ? layer.parameters.toLocaleString() : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* CNN Highlights */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-5 rounded-2xl border border-slate-200 bg-white shadow-xs">
              <h4 className="text-sm font-semibold text-slate-900">Spatio-Temporal Stacking</h4>
              <p className="text-xs text-slate-500 mt-2 leading-relaxed">
                Stacks 4 consecutive frames (4 × 84 × 84) to eliminate partial observability, giving the convolutional filters complete trajectory information.
              </p>
            </div>
            <div className="p-5 rounded-2xl border border-slate-200 bg-white shadow-xs">
              <h4 className="text-sm font-semibold text-slate-900">Hierarchical Feature Maps</h4>
              <p className="text-xs text-slate-500 mt-2 leading-relaxed">
                Conv1 extracts spatial edge orientations; Conv2 captures paddle contours; Conv3 models complex ball-paddle collision geometry.
              </p>
            </div>
            <div className="p-5 rounded-2xl border border-slate-200 bg-white shadow-xs">
              <h4 className="text-sm font-semibold text-slate-900">Dual Head Adaptability</h4>
              <p className="text-xs text-slate-500 mt-2 leading-relaxed">
                In DQN, outputs 6 discrete Q-values Q(s, a). In PPO, branches into an Actor policy head π(a|s) and a Critic state-value head V(s).
              </p>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: KARPATHY MLP */}
      {activeTab === "mlp" && (
        <div className="space-y-6">
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">
                  Karpathy Policy Network (REINFORCE Baseline)
                </h2>
                <p className="text-sm text-slate-500 mt-1">
                  High-speed 2-layer Multi-Layer Perceptron operating on flattened 80×80 difference frames.
                </p>
              </div>
              <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-slate-100 rounded-lg text-xs font-mono font-medium text-slate-700">
                <span>Total Parameters:</span>
                <span className="font-bold text-emerald-600">{totalMlpParams.toLocaleString()}</span>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-xs font-semibold text-slate-500 uppercase">
                    <th className="py-3 px-4">Layer</th>
                    <th className="py-3 px-4">Type</th>
                    <th className="py-3 px-4">Configuration</th>
                    <th className="py-3 px-4">Output Dimensions</th>
                    <th className="py-3 px-4 text-right">Parameters</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  {KARPATHY_MLP_LAYERS.map((layer, idx) => (
                    <tr key={idx} className="hover:bg-slate-50/70 transition-colors">
                      <td className="py-3 px-4 font-semibold text-slate-900">{layer.layer}</td>
                      <td className="py-3 px-4">{layer.type}</td>
                      <td className="py-3 px-4 font-mono text-xs text-slate-600">{layer.config}</td>
                      <td className="py-3 px-4 font-mono text-xs text-emerald-600 font-semibold">{layer.outputShape}</td>
                      <td className="py-3 px-4 text-right font-mono text-xs text-slate-500">
                        {layer.parameters > 0 ? layer.parameters.toLocaleString() : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-slate-900 text-slate-200 rounded-2xl p-6 shadow-xs font-mono text-xs space-y-4">
            <div className="flex items-center justify-between text-slate-400 border-b border-slate-800 pb-2">
              <span className="font-bold text-slate-300">PyTorch Implementation Snippet (`src/model.py`)</span>
              <span>KarpathyPolicyNet</span>
            </div>
            <pre className="text-slate-300 overflow-x-auto leading-relaxed">
{`class KarpathyPolicyNet(nn.Module):
    def __init__(self, input_dim=6400, hidden_dim=200):
        super(KarpathyPolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out  # Scalar probability P(UP)`}
            </pre>
          </div>
        </div>
      )}

      {/* TAB 4: MATHEMATICAL FORMULATIONS */}
      {activeTab === "math" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Card 1: Policy Gradient */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-blue-600">REINFORCE</span>
              <span className="text-xs px-2 py-0.5 bg-slate-100 rounded text-slate-600 font-mono">Williams (1992)</span>
            </div>
            <h3 className="text-base font-bold text-slate-900">Policy Gradient Theorem</h3>
            <div className="p-4 bg-slate-50 rounded-xl font-mono text-sm text-center border border-slate-200 text-slate-800 overflow-x-auto">
              ∇_θ J(θ) = E [ ∑_t ∇_θ log π_θ(a_t | s_t) · G_t ]
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Ascends the gradient of expected cumulative reward. Monte Carlo return G_t is discounted by γ = 0.99 and standardized:
            </p>
            <div className="p-3 bg-slate-50 rounded-lg font-mono text-xs text-slate-700 text-center border border-slate-200">
              A_t = (G_t - μ_G) / (σ_G + 1e-5)
            </div>
          </div>

          {/* Card 2: Deep Q-Learning */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-600">DQN</span>
              <span className="text-xs px-2 py-0.5 bg-slate-100 rounded text-slate-600 font-mono">Mnih et al. (2015)</span>
            </div>
            <h3 className="text-base font-bold text-slate-900">Bellman Mean Squared Error</h3>
            <div className="p-4 bg-slate-50 rounded-xl font-mono text-sm text-center border border-slate-200 text-slate-800 overflow-x-auto">
              L(θ) = E [ ( r + γ max_a' Q(s', a'; θ^-) - Q(s, a; θ) )² ]
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Minimizes temporal difference error using quasi-static target network parameters θ^- updated every 1,000 steps to prevent divergence.
            </p>
          </div>

          {/* Card 3: PPO Clipped Surrogate */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-purple-600">PPO</span>
              <span className="text-xs px-2 py-0.5 bg-slate-100 rounded text-slate-600 font-mono">Schulman et al. (2017)</span>
            </div>
            <h3 className="text-base font-bold text-slate-900">Clipped Surrogate Objective</h3>
            <div className="p-4 bg-slate-50 rounded-xl font-mono text-sm text-center border border-slate-200 text-slate-800 overflow-x-auto">
              L^CLIP(θ) = E [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Where r_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t). Clipping bounds the probability ratio to [0.8, 1.2], preserving stability in stochastic policy spaces.
            </p>
          </div>

          {/* Card 4: Memory Optimization Math */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-amber-600">Systems Engineering</span>
              <span className="text-xs px-2 py-0.5 bg-slate-100 rounded text-slate-600 font-mono">Buffer Quantization</span>
            </div>
            <h3 className="text-base font-bold text-slate-900">uint8 Memory Compression</h3>
            <div className="p-4 bg-slate-50 rounded-xl font-mono text-sm text-center border border-slate-200 text-slate-800 overflow-x-auto">
              ΔMemory = 1 - (1 byte / 16 bytes) = 93.75% Savings
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Standard PyTorch float32 storage requires 22.5 GB for 100k transitions. By storing raw frames as 8-bit uint8 and lazily dividing by 255.0 on GPU batch transfer, memory is compressed to 1.41 GB.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
