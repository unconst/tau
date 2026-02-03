import { useState, useEffect, useRef } from 'react';
import { Copy, Check } from 'lucide-react';

const asciiArt = `___________               _______  .__            __        
\\__    ___/____   __ __   \\      \\ |__| ____     |__|____   
  |    |  \\__  \\ |  |  \\  /   |   \\|  |/    \\    |  \\__  \\  
  |    |   / __ \\|  |  / /    |    \\  |   |  \\   |  |/ __ \\_
  |____|  (____  /____/  \\____|__  /__|___|  /\\__|  (____  /
               \\/                \\/        \\/\\______|    \\/ `;

function App() {
  const [copied, setCopied] = useState(false);
  const [mousePos, setMousePos] = useState({ x: 0.5, y: 0.5 });
  const containerRef = useRef<HTMLDivElement>(null);

  const curlCommand = 'curl -fsSL https://tau.ninja/install.sh | bash';

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        setMousePos({ x, y });
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(curlCommand);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  // Calculate gradient positions based on mouse
  const gradientX = mousePos.x * 100;
  const gradientY = mousePos.y * 100;

  return (
    <div
      ref={containerRef}
      className="relative min-h-screen w-full overflow-hidden"
      style={{
        background: `
          radial-gradient(ellipse at ${gradientX}% ${gradientY - 30}%, rgba(100, 149, 237, 0.8) 0%, transparent 50%),
          radial-gradient(ellipse at ${gradientX + 20}% ${gradientY}%, rgba(138, 43, 226, 0.7) 0%, transparent 45%),
          radial-gradient(ellipse at ${gradientX - 20}% ${gradientY + 20}%, rgba(255, 105, 180, 0.6) 0%, transparent 40%),
          radial-gradient(ellipse at ${gradientX}% ${gradientY + 50}%, rgba(255, 165, 0, 0.7) 0%, transparent 50%),
          linear-gradient(180deg, 
            rgba(224, 247, 250, 1) 0%, 
            rgba(187, 222, 251, 0.9) 15%,
            rgba(147, 197, 253, 0.8) 30%,
            rgba(167, 139, 250, 0.8) 50%,
            rgba(244, 114, 182, 0.8) 70%,
            rgba(251, 191, 36, 0.9) 85%,
            rgba(254, 240, 138, 1) 100%
          )
        `,
      }}
    >
      {/* Animated gradient overlay for continuous movement */}
      <div 
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `
            radial-gradient(ellipse at 30% 20%, rgba(100, 149, 237, 0.4) 0%, transparent 40%),
            radial-gradient(ellipse at 70% 40%, rgba(138, 43, 226, 0.3) 0%, transparent 35%),
            radial-gradient(ellipse at 40% 60%, rgba(255, 105, 180, 0.3) 0%, transparent 30%),
            radial-gradient(ellipse at 60% 80%, rgba(255, 165, 0, 0.4) 0%, transparent 40%)
          `,
          animation: 'gradientShift 15s ease-in-out infinite',
        }}
      />

      {/* Top left decorative element */}
      <div className="absolute top-6 left-6">
        <div className="w-8 h-8 rounded-full border border-white/40 flex items-center justify-center backdrop-blur-sm">
          <div className="w-1.5 h-1.5 rounded-full bg-white/60" />
        </div>
      </div>

      {/* Center content - ASCII Art + Code Copy Bar */}
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-8">
        {/* ASCII Art Header */}
        <pre
          className="ascii-float text-center select-none"
          style={{
            color: 'rgba(30, 27, 75, 0.85)',
            fontSize: 'clamp(0.4rem, 1.2vw, 0.9rem)',
            lineHeight: 1.2,
            fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
            textShadow: `
              0 0 20px rgba(168, 85, 247, 0.4),
              0 0 40px rgba(236, 72, 153, 0.3),
              0 0 60px rgba(168, 85, 247, 0.2)
            `,
            filter: 'drop-shadow(0 4px 20px rgba(139, 92, 246, 0.3))',
          }}
        >
          {asciiArt}
        </pre>

        {/* Code Copy Bar */}
        <div 
          className="group relative flex items-center gap-3 px-6 py-4 rounded-2xl backdrop-blur-xl transition-transform duration-200 hover:scale-105"
          style={{
            background: 'rgba(255, 255, 255, 0.15)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.2)',
            border: '1px solid rgba(255, 255, 255, 0.25)',
          }}
        >
          <code className="text-xs sm:text-sm font-mono text-slate-800/90 tracking-tight">
            {curlCommand}
          </code>
          <button
            onClick={handleCopy}
            className="flex items-center justify-center w-9 h-9 rounded-xl transition-all duration-200 hover:bg-white/20"
            aria-label="Copy to clipboard"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-600" strokeWidth={2.5} />
            ) : (
              <Copy className="w-4 h-4 text-slate-700/70 group-hover:text-slate-800" strokeWidth={2} />
            )}
          </button>
        </div>
      </div>

      {/* Bottom left info */}
      <div className="absolute bottom-6 left-6 flex items-center gap-3 text-slate-700/80">
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
          <circle cx="12" cy="10" r="3" />
        </svg>
        <span className="text-sm font-medium">🥷 Tau Ninja</span>
      </div>

      {/* Global styles for animation */}
      <style>{`
        @keyframes gradientShift {
          0%, 100% {
            opacity: 0.5;
            transform: translate(0, 0) scale(1);
          }
          25% {
            opacity: 0.7;
            transform: translate(2%, -2%) scale(1.05);
          }
          50% {
            opacity: 0.6;
            transform: translate(-1%, 1%) scale(0.98);
          }
          75% {
            opacity: 0.8;
            transform: translate(1%, 2%) scale(1.02);
          }
        }
        
        @keyframes float {
          0%, 100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-10px);
          }
        }
        
        .ascii-float {
          animation: float 6s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}

export default App;
