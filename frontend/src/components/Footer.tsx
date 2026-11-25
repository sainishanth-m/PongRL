export default function Footer() {
  return (
    <footer className="mt-auto border-t border-slate-200/80 bg-white py-8 text-slate-600">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs">
        <div>
          <span className="font-semibold text-slate-900">PongRL Pro</span> — Autonomous Atari Pong Game Agent via Deep Reinforcement Learning
          <p className="text-slate-400 mt-0.5">B.Tech Capstone Project • SCOPE, VIT-AP University, Amaravati</p>
        </div>

        <div className="flex items-center gap-4 text-slate-500">
          <span>Python 3.10+</span>
          <span>•</span>
          <span>PyTorch 2.x</span>
          <span>•</span>
          <span>Gymnasium ALE</span>
          <span>•</span>
          <span>Vercel Deploy</span>
        </div>
      </div>
    </footer>
  );
}
