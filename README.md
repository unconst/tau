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

Use `<prompt>` calls to ask questions normally 
> (i.e. "how do I use you?"). 

Use `/adapt <prompt>` to change the agent code directly 
> (i.e. "/adapt Change the code to accept a new `/plan` command").

Use `/cron <interval> <prompt>` to create an agent which runs every `<interval>`
Use `/crons` to lists all active cron jobs and `/uncron <id>` to remove them.

Use `/task <prompt>` to add a longer running challenge which the agent comes back to periodically. 
> (i.e. "/task send me a message at 5pm") 

## How does it work?

Tau maintains a directory on your computer (where you installed it) this is the full agent context. You can add files to it etc and it will all come into contact with your agent.

## License

MIT License - see [LICENSE](LICENSE) for details.
