# Step-by-step setup (Windows and Mac)

You only do steps 1–5 once. Step 6 is how you start it each time.

---

## Step 1: Install Python

1. Go to https://www.python.org/downloads/ and download Python **3.11 or newer**.
2. Run the installer.
   * **Windows:** on the first screen, tick **"Add python.exe to PATH"**, then click *Install Now*.
   * **Mac:** just click through the installer.

## Step 2: Download the bot

1. Go to https://github.com/tradingwithedgar-del/tradingbot-dev1
2. Click the branch dropdown (top left, above the file list) and choose **`claude/clever-lamport-2atw36`**.
3. Click the green **Code** button, then **Download ZIP**.
4. Unzip it somewhere easy, for example `Documents\tradingbot` (Windows) or `Documents/tradingbot` (Mac).

Inside that folder you'll see `README.md`, `SETUP.md`, `requirements.txt`, a `tradingbot` folder and a file
called **`.env.example`**. That's the settings template.
(On a Mac, files starting with a dot are hidden in Finder. Press **Cmd + Shift + .** to show them. You don't
need to see it, though, because the commands below handle it.)

## Step 3: Open a terminal in the bot folder

* **Windows:** open the folder in File Explorer, click the address bar at the top, type `cmd` and press Enter.
  A black window opens, already in the right folder.
* **Mac:** open the *Terminal* app, type `cd ` (with a space), drag the bot folder into the Terminal window,
  and press Enter.

## Step 4: Install the bot (one time)

Copy and paste these lines one at a time.

**Windows**
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

**Mac**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
open -e .env
```

After `activate`, the start of the line shows `(.venv)`. That means you're in the bot's environment.
The last line opens your settings file `.env` in a text editor.

## Step 5: Fill in your settings (`.env`)

Edit these lines, then save and close:

```
TL_ENVIRONMENT=https://demo.tradelocker.com
TL_EMAIL=the email you log in to TradeLocker with
TL_PASSWORD=your TradeLocker password
TL_SERVER=the server name
BOT_MODE=demo
ALLOW_LIVE_TRADING=NO
BOT_SYMBOLS=US30,US500,USTECH,XAUUSD,NVDA,AAPL,TSLA,XTIUSD,BTCUSD,ETHUSD,SOLUSD
BOT_TIMEFRAME=15m
```

* **TL_SERVER**: go to https://demo.tradelocker.com and look at the login form. The **Server** field
  shows PlexyTrade's demo server name. Copy it exactly.
* **TL_ACC_NUM**: leave it empty unless you have several demo accounts on the same login.
* Never share this file or upload it anywhere. It holds your password. It is already excluded from git.

### Check the login and symbol names
Brokers name instruments differently (for example `NAS100` vs `US100` vs `USTEC`). Run:
```
python -m tiim symbols
```
It logs in, lists every instrument on your account, and tells you which of your `BOT_SYMBOLS` it can't find,
with similar names. To search for one:
```
python -m tiim symbols --search NAS
python -m tiim symbols --search OIL
```
Fix any names in `.env`, save, and run `python -m tiim symbols` again until it says
**"all symbols found - ready to run"**.

## Step 6: Use it

Every time you open a new terminal, first go into the folder (Step 3) and activate the environment:
* Windows: `.venv\Scripts\activate`
* Mac: `source .venv/bin/activate`

### a) Backtest on real price history (no trades are placed)
```
python -m tiim backtest --tradelocker --days 60
```
This downloads 60 days of history for your symbols with your demo login, then lets the agent trade it in a
simulator. It takes a few minutes. Then look at the results:
```
python -m tiim dashboard --db data/backtest.db
```
Open **http://127.0.0.1:8000** in your browser. Press **Ctrl + C** in the terminal to stop the dashboard.

### b) Start trading the demo account
```
python -m tiim run
```
Leave this window open. The agent checks the market every 20 seconds and acts each time a 15-minute bar closes.
**Your computer has to stay on and awake while it runs.** Turn off sleep in your power settings. A cheap VPS
can run it 24/7 later.

To watch it, open a **second** terminal (Step 3, then activate) and run:
```
python -m tiim dashboard
```
and open **http://127.0.0.1:8000**.

### Useful commands
| Command | What it does |
|---|---|
| `python -m tiim status` | Quick look: open trades, latest journal entries, halted or not |
| `python -m tiim stop` | No new trades. Open trades keep their stop loss and take profit |
| `python -m tiim resume` | Allow new trades again (also clears a drawdown halt) |
| **Ctrl + C** in the `run` window | Stops the program. Open trades stay on TradeLocker with their SL/TP |

When you restart `python -m tiim run`, it picks up where it left off: its memory and lessons are saved in
`data/bot.db`. Don't delete that file.

## Step 7: Before going live
Keep it on demo until it has **100+ real demo trades** and positive expectancy on the dashboard. Then we'll
switch it over together. It needs `TL_ENVIRONMENT=https://live.tradelocker.com`, `BOT_MODE=live`,
`ALLOW_LIVE_TRADING=YES` and your live login.

## Something went wrong?
| Message | Fix |
|---|---|
| `'python' is not recognized` (Windows) | Reinstall Python and tick "Add python.exe to PATH" |
| `TL_EMAIL, TL_PASSWORD and TL_SERVER must be set` | Fill in `.env` and save it (Step 5) |
| `These symbols don't exist on your TradeLocker account` | Use the suggested names in `BOT_SYMBOLS` |
| Login / 401 errors | Check email, password and server name. Try logging in at demo.tradelocker.com with the same details |
| `No journal at data/bot.db` when opening the dashboard | Run the agent or a backtest first |
