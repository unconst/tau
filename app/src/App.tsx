import { useState, useEffect, useRef, useCallback } from 'react';
import { Copy, Check, Send, Zap, Shield, Globe, Terminal, ChevronDown } from 'lucide-react';

const API_URL = 'https://vercel-app-rosy-kappa.vercel.app';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button onClick={handleCopy} className="p-1.5 rounded-lg hover:bg-white/10 transition-colors" aria-label="Copy">
      {copied ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
    </button>
  );
}

function GlassCard({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-2xl backdrop-blur-xl ${className}`}
      style={{
        background: 'rgba(15, 15, 30, 0.7)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
      }}>
      {children}
    </div>
  );
}

function CodeBlock({ code, lang = 'bash' }: { code: string; lang?: string }) {
  return (
    <div className="relative group">
      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <CopyButton text={code} />
      </div>
      <pre className="p-4 rounded-xl text-xs sm:text-sm font-mono overflow-x-auto" style={{
        background: 'rgba(0, 0, 0, 0.4)',
        border: '1px solid rgba(255, 255, 255, 0.06)',
      }}>
        <code className={`language-${lang} text-slate-300`}>{code}</code>
      </pre>
    </div>
  );
}

function ChatPlayground() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(scrollToBottom, [messages, streamingContent]);

  const sendMessage = useCallback(async () => {
    if (!input.trim() || loading) return;
    const userMsg: Message = { role: 'user', content: input.trim() };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setLoading(true);
    setStreamingContent('');

    try {
      const res = await fetch(`${API_URL}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'qwen-7b',
          messages: newMessages.map(m => ({ role: m.role, content: m.content })),
          max_tokens: 512,
          stream: true,
        }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body?.getReader();
      if (!reader) throw new Error('No reader');
      const decoder = new TextDecoder();
      let accumulated = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta?.content;
            if (delta) {
              accumulated += delta;
              setStreamingContent(accumulated);
            }
          } catch { /* skip malformed */ }
        }
      }

      setMessages(prev => [...prev, { role: 'assistant', content: accumulated }]);
      setStreamingContent('');
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err instanceof Error ? err.message : 'Request failed'}` }]);
    } finally {
      setLoading(false);
    }
  }, [input, messages, loading]);

  return (
    <GlassCard className="flex flex-col h-[480px]">
      <div className="px-5 py-3 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-sm font-medium text-white/80">Qwen 7B &mdash; Subnet 97</span>
        </div>
        <span className="text-xs text-slate-500">Streaming</span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && !loading && (
          <div className="flex items-center justify-center h-full text-slate-500 text-sm">
            Send a message to try decentralized inference
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
              m.role === 'user'
                ? 'bg-indigo-500/30 text-white/90 rounded-br-md'
                : 'bg-white/5 text-slate-300 rounded-bl-md'
            }`}>
              {m.content}
            </div>
          </div>
        ))}
        {streamingContent && (
          <div className="flex justify-start">
            <div className="max-w-[80%] px-4 py-2.5 rounded-2xl rounded-bl-md text-sm leading-relaxed bg-white/5 text-slate-300">
              {streamingContent}<span className="animate-pulse">|</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-3 border-t border-white/5">
        <div className="flex items-end gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
            placeholder="Type a message..."
            rows={1}
            className="flex-1 resize-none rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 outline-none"
            style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="p-2.5 rounded-xl bg-indigo-500/80 hover:bg-indigo-500 disabled:opacity-30 transition-all"
          >
            <Send className="w-4 h-4 text-white" />
          </button>
        </div>
      </div>
    </GlassCard>
  );
}

function App() {
  const [mousePos, setMousePos] = useState({ x: 0.5, y: 0.5 });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setMousePos({ x: (e.clientX - rect.left) / rect.width, y: (e.clientY - rect.top) / rect.height });
      }
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const gx = mousePos.x * 100;
  const gy = mousePos.y * 100;

  const curlExample = `curl ${API_URL}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "stream": true
  }'`;

  const pythonExample = `from openai import OpenAI

client = OpenAI(
    base_url="${API_URL}/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="qwen-7b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")`;

  const jsExample = `const response = await fetch("${API_URL}/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "qwen-7b",
    messages: [{ role: "user", content: "Hello!" }],
    max_tokens: 256,
    stream: false,
  }),
});

const data = await response.json();
console.log(data.choices[0].message.content);`;

  const [activeTab, setActiveTab] = useState<'curl' | 'python' | 'js'>('curl');
  const codeExamples = { curl: curlExample, python: pythonExample, js: jsExample };

  return (
    <div
      ref={containerRef}
      className="relative min-h-screen w-full overflow-x-hidden"
      style={{
        background: `
          radial-gradient(ellipse at ${gx}% ${gy - 30}%, rgba(60, 80, 180, 0.5) 0%, transparent 50%),
          radial-gradient(ellipse at ${gx + 20}% ${gy}%, rgba(100, 30, 180, 0.4) 0%, transparent 45%),
          radial-gradient(ellipse at ${gx - 20}% ${gy + 20}%, rgba(180, 60, 120, 0.3) 0%, transparent 40%),
          linear-gradient(180deg, #0a0a1a 0%, #0d0d2b 30%, #12102e 60%, #0a0a1a 100%)
        `,
      }}
    >
      {/* Animated overlay */}
      <div className="absolute inset-0 pointer-events-none" style={{
        background: `
          radial-gradient(ellipse at 30% 20%, rgba(60, 80, 200, 0.15) 0%, transparent 40%),
          radial-gradient(ellipse at 70% 40%, rgba(100, 30, 180, 0.1) 0%, transparent 35%)
        `,
        animation: 'gradientShift 15s ease-in-out infinite',
      }} />

      {/* Nav */}
      <nav className="relative z-10 flex items-center justify-between px-6 sm:px-10 py-5">
        <span className="text-lg font-semibold text-white/90 tracking-tight">Constantinople</span>
        <div className="flex items-center gap-6 text-sm text-slate-400">
          <a href="#playground" className="hover:text-white transition-colors">Playground</a>
          <a href="#api" className="hover:text-white transition-colors">API</a>
          <a href="https://github.com/unconst/Constantinople" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">GitHub</a>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative z-10 flex flex-col items-center text-center px-6 pt-16 pb-24 max-w-4xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium mb-6"
          style={{ background: 'rgba(99, 102, 241, 0.15)', border: '1px solid rgba(99, 102, 241, 0.3)', color: 'rgba(165, 180, 252, 1)' }}>
          <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
          Subnet 97 &mdash; Live on Bittensor Mainnet
        </div>

        <h1 className="text-4xl sm:text-6xl font-bold text-white tracking-tight leading-tight mb-5">
          Decentralized LLM<br />
          <span style={{ background: 'linear-gradient(135deg, #818cf8, #c084fc, #f472b6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Inference
          </span>
        </h1>

        <p className="text-lg text-slate-400 max-w-xl mb-10 leading-relaxed">
          OpenAI-compatible API powered by a decentralized network of GPU miners.
          Fast, censorship-resistant, always available.
        </p>

        <div className="flex flex-col sm:flex-row items-center gap-3">
          <a href="#playground" className="px-6 py-3 rounded-xl font-medium text-sm text-white transition-all hover:scale-105"
            style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', boxShadow: '0 4px 20px rgba(99, 102, 241, 0.4)' }}>
            Try it now
          </a>
          <a href="#api" className="px-6 py-3 rounded-xl font-medium text-sm text-slate-300 hover:text-white transition-all"
            style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)' }}>
            View API docs
          </a>
        </div>

        <a href="#playground" className="mt-16 animate-bounce text-slate-500 hover:text-slate-300 transition-colors">
          <ChevronDown className="w-5 h-5" />
        </a>
      </section>

      {/* Features */}
      <section className="relative z-10 grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-4xl mx-auto px-6 pb-20">
        {[
          { icon: Zap, title: 'Fast', desc: 'Real GPU inference from RTX 5090 miners. Sub-50ms time to first token.' },
          { icon: Shield, title: 'Verified', desc: 'Hidden state challenges cryptographically verify miners run real models.' },
          { icon: Globe, title: 'Decentralized', desc: 'No single point of failure. Miners compete on speed and quality for emission.' },
        ].map(({ icon: Icon, title, desc }) => (
          <GlassCard key={title} className="p-5">
            <Icon className="w-5 h-5 text-indigo-400 mb-3" />
            <h3 className="text-white font-medium text-sm mb-1">{title}</h3>
            <p className="text-slate-500 text-xs leading-relaxed">{desc}</p>
          </GlassCard>
        ))}
      </section>

      {/* Playground */}
      <section id="playground" className="relative z-10 max-w-4xl mx-auto px-6 pb-20">
        <h2 className="text-2xl font-bold text-white mb-2">Playground</h2>
        <p className="text-slate-500 text-sm mb-6">Chat with the model directly. Responses stream from decentralized miners on Bittensor Subnet 97.</p>
        <ChatPlayground />
      </section>

      {/* API Docs */}
      <section id="api" className="relative z-10 max-w-4xl mx-auto px-6 pb-20">
        <h2 className="text-2xl font-bold text-white mb-2">API</h2>
        <p className="text-slate-500 text-sm mb-6">
          OpenAI-compatible. Use any existing client library&mdash;just change the base URL.
        </p>

        <GlassCard className="p-5">
          <div className="flex items-center gap-2 mb-4 text-xs">
            <span className="px-2 py-0.5 rounded-md bg-green-500/20 text-green-400 font-medium">POST</span>
            <code className="text-slate-400">/v1/chat/completions</code>
          </div>

          <div className="flex gap-1 mb-4">
            {(['curl', 'python', 'js'] as const).map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  activeTab === tab ? 'bg-white/10 text-white' : 'text-slate-500 hover:text-slate-300'
                }`}>
                {tab === 'js' ? 'JavaScript' : tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          <CodeBlock code={codeExamples[activeTab]} lang={activeTab === 'curl' ? 'bash' : activeTab === 'python' ? 'python' : 'javascript'} />

          <div className="mt-5 space-y-3">
            <h4 className="text-sm font-medium text-white/80">Parameters</h4>
            <div className="grid gap-2 text-xs">
              {[
                ['model', 'string', '"qwen-7b"', 'Model to use for inference'],
                ['messages', 'array', '—', 'Array of {role, content} message objects'],
                ['max_tokens', 'integer', '256', 'Maximum tokens to generate'],
                ['stream', 'boolean', 'false', 'Stream response as Server-Sent Events'],
                ['temperature', 'float', '1.0', 'Sampling temperature (0.0 - 2.0)'],
              ].map(([name, type, def, desc]) => (
                <div key={name} className="flex items-start gap-3 py-2 border-b border-white/5 last:border-0">
                  <code className="text-indigo-400 font-medium min-w-[100px]">{name}</code>
                  <span className="text-slate-600 min-w-[60px]">{type}</span>
                  <span className="text-slate-600 min-w-[60px]">{def}</span>
                  <span className="text-slate-400">{desc}</span>
                </div>
              ))}
            </div>
          </div>
        </GlassCard>
      </section>

      {/* Network Info */}
      <section className="relative z-10 max-w-4xl mx-auto px-6 pb-20">
        <h2 className="text-2xl font-bold text-white mb-6">Network</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { label: 'Subnet', value: '97' },
            { label: 'Model', value: 'Qwen 7B' },
            { label: 'Miners', value: 'Active' },
            { label: 'Verification', value: 'Hidden State' },
          ].map(({ label, value }) => (
            <GlassCard key={label} className="p-4 text-center">
              <div className="text-xs text-slate-500 mb-1">{label}</div>
              <div className="text-white font-semibold text-sm">{value}</div>
            </GlassCard>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/5 px-6 py-8">
        <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <span className="text-sm text-slate-500">Constantinople &mdash; Bittensor Subnet 97</span>
          <div className="flex items-center gap-4 text-xs text-slate-600">
            <a href="https://github.com/unconst/Constantinople" target="_blank" rel="noopener noreferrer" className="hover:text-slate-400 transition-colors flex items-center gap-1">
              <Terminal className="w-3.5 h-3.5" /> GitHub
            </a>
          </div>
        </div>
      </footer>

      <style>{`
        @keyframes gradientShift {
          0%, 100% { opacity: 0.5; transform: translate(0, 0) scale(1); }
          25% { opacity: 0.7; transform: translate(2%, -2%) scale(1.05); }
          50% { opacity: 0.6; transform: translate(-1%, 1%) scale(0.98); }
          75% { opacity: 0.8; transform: translate(1%, 2%) scale(1.02); }
        }
      `}</style>
    </div>
  );
}

export default App;
