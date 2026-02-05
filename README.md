<p align="center">
  <img src="tauninja.png" alt="Tau Ninja" width="800" />
</p>

<p align="center">
  <strong><span style="font-size:1.5em;">An agent that can do stuff and upgrade its own code in real time.</span></strong>
</p>

## How to install?

```bash
curl -fsSL https://tau.ninja/install.sh | bash
```

## How to use?

Just chat naturally in Telegram. Tau understands what you mean.

**Examples:**
- "remind me to call mom at 5pm"
- "what's the capital of France?"
- "every morning, send me a motivational quote"
- "research the best noise-canceling headphones and get back to me"
- "add a feature that lets me check my calendar"

**Power user commands** (optional):
- `/adapt <prompt>` — modify Tau's own code
- `/cron <interval> <prompt>` — create recurring jobs
- `/task <prompt>` — add a background task 

## How does it work?

Tau maintains a directory on your computer (where you installed it) this is the full agent context. You can add files to it etc and it will all come into contact with your agent.

## License

MIT License - see [LICENSE](LICENSE) for details.
