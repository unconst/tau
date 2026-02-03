# Tau

An agent that can do stuff and upgrade its own code in real time.

## How to install?

```bash
curl -fsSL https://raw.githubusercontent.com/unconst/tau/main/install.sh | bash
```

## How to use?

Use `<prompt>` calls to ask questions normally 
> (i.e. "how do I use you?"). 

Use `/adapt <prompt>` to change the agent code directly 
> (i.e. "Change the code to accept a new `/plan` command").

Use `/task <prompt>` to add a longer running challenge which the agent comes back to periodically. 
> (i.e. "send me a message at 5pm") 

## How does it work?

Tau maintains a directory on your computer (where you installed it) this is the full agent context. You can add files to it etc and it will all come into contact with your agent.


