# Running TIIM 24/7 on a cloud server

Your Mac doesn't need to stay on. TIIM runs on a small Linux server (a "VPS", about $5-6/month), restarts
itself after crashes or reboots, and installs updates from GitHub every 15 minutes, but only after all of
its checks pass. A broken update is rolled back automatically.

## 1. Rent a server
Any provider works: Hetzner (CX22), DigitalOcean (Basic droplet), Vultr, Linode.
* Image: **Ubuntu 24.04**
* Size: **1-2 GB RAM** is enough
* Region: anywhere (close to New York or London is nice for latency, not required)
* Login: choose a **root password** (simplest) or an SSH key

## 2. Stop TIIM on your Mac
Only one TIIM may trade the account. Press Ctrl + C in the TIIM window on your Mac.

## 3. Connect and install
From your Mac's Terminal (use the server's IP address):
```
ssh root@YOUR_SERVER_IP
```
Then, on the server:
```
curl -fsSL https://raw.githubusercontent.com/tradingwithedgar-del/tradingbot-dev1/claude/clever-lamport-2atw36/deploy/setup_server.sh | bash
```
It asks for your TradeLocker login and a dashboard password, then starts TIIM.

## 4. Let Claude read the news with your subscription (optional, recommended)
On the server:
```
sudo -u tiim -i
claude setup-token
```
Open the link it shows on your Mac or phone, approve, and copy the token it prints (starts `sk-ant-oat01-`).
Then `exit`, open the settings with `nano /opt/tiim/.env`, paste the token after `CLAUDE_CODE_OAUTH_TOKEN=`,
save (Ctrl+O, Enter, Ctrl+X) and restart: `systemctl restart tiim`. Check with `tiim news`; it should
say "Reading headlines with: claude".
This uses your Claude plan's usage allowance (one call every 5 minutes at most).

## Everyday use
| What | How |
|---|---|
| Dashboard | `http://YOUR_SERVER_IP:8000` in any browser (phone too), any name + your dashboard password |
| Live log | `ssh root@YOUR_SERVER_IP` then `journalctl -u tiim -f` (Ctrl+C to leave) |
| Status / commands | `tiim status`, `tiim strategies`, `tiim news`, `tiim stop`, `tiim resume` |
| Restart | `systemctl restart tiim` |
| Change settings | `nano /opt/tiim/.env`, then `systemctl restart tiim` |
| Update log | `journalctl -u tiim-update` |

## Adding your strategies
Describe the strategy to Claude. It's added to `my_strategies/` in GitHub, and the server picks it up within
15 minutes after TIIM's checks pass. `tiim strategies` shows it in the library.

## Security notes
* `.env` (passwords, tokens) lives only on the server and is readable only by root and the tiim user.
* The dashboard is read-only (it can't place trades) and needs your password. It uses plain HTTP, so prefer
  trusted networks, or view it through an SSH tunnel: `ssh -L 8000:localhost:8000 root@YOUR_SERVER_IP`
  and then open http://localhost:8000.
