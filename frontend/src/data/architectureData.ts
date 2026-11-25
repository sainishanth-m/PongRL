export interface NetworkLayer {
  layer: string;
  type: string;
  config: string;
  outputShape: string;
  parameters: number;
}

export const NATURE_CNN_LAYERS: NetworkLayer[] = [
  { layer: "Input", type: "State Stack", config: "4 Grayscale Frames", outputShape: "(4, 84, 84)", parameters: 0 },
  { layer: "Conv1", type: "2D Convolution + ReLU", config: "32 filters, 8×8 kernel, stride=4", outputShape: "(32, 20, 20)", parameters: 8224 },
  { layer: "Conv2", type: "2D Convolution + ReLU", config: "64 filters, 4×4 kernel, stride=2", outputShape: "(64, 9, 9)", parameters: 32832 },
  { layer: "Conv3", type: "2D Convolution + ReLU", config: "64 filters, 3×3 kernel, stride=1", outputShape: "(64, 7, 7)", parameters: 36928 },
  { layer: "Flatten", type: "Tensor Reshape", config: "32 × 7 × 7 feature map", outputShape: "(3136)", parameters: 0 },
  { layer: "FC1", type: "Dense Linear + ReLU", config: "512 Hidden Units", outputShape: "(512)", parameters: 1606144 },
  { layer: "Output", type: "Linear Head", config: "6 Discrete Action Logits", outputShape: "(6)", parameters: 3078 },
];

export const KARPATHY_MLP_LAYERS: NetworkLayer[] = [
  { layer: "Input", type: "Difference State x", config: "80×80 Flattened Binarized Δs", outputShape: "(6400)", parameters: 0 },
  { layer: "Hidden FC1", type: "Linear + ReLU", config: "200 Neurons with ReLU activation", outputShape: "(200)", parameters: 1280200 },
  { layer: "Output FC2", type: "Linear + Sigmoid", config: "1 Scalar Probability P(UP)", outputShape: "(1)", parameters: 201 },
];

export const WORKFLOW_STAGES = [
  { step: 1, title: "ALE Atari 2600 Emulation", desc: "Stella cycle-accurate engine renders frames at 60 FPS in 210×160 RGB." },
  { step: 2, title: "Arena Cropping", desc: "Top scoreboard and bottom border are cropped out, isolating the 160×160 court." },
  { step: 3, title: "Spatial Downsampling", desc: "Downsampled 2:1 to an 80×80 single-channel luminance representation." },
  { step: 4, title: "Binary Thresholding", desc: "Background court noise is masked out; active paddles and ball are thresholded to 1." },
  { step: 5, title: "Temporal Frame Differencing", desc: "Δst = st - st-1 explicitly encodes ball and paddle velocity vectors." },
  { step: 6, title: "Neural Forward Inference", desc: "Policy Network predicts action distribution or state-action Q-values." },
  { step: 7, title: "Action Execution", desc: "Discrete commands {UP, DOWN} executed in the simulated Atari joystick interface." },
  { step: 8, title: "uint8 Memory Buffering", desc: "Transitions stored in quantized 8-bit circular buffer, saving 93.75% host RAM." },
  { step: 9, title: "Gradient Optimization", desc: "RMSprop/Adam backpropagates policy loss based on discounted returns Gt." },
  { step: 10, title: "Telemetry & Live Evaluation", desc: "Real-time probability gauge and episode tracking streamed to web dashboard." }
];
