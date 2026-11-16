# Instruction sheet: Real-Time Network Intrusion Detection System

**What this is.** The system replays recorded network traffic, one connection at a
time. A machine-learning model checks each one, and the results appear live on a web
dashboard. Your job is to try five short tasks on your own. You are not being tested;
the system is. If you get stuck, say so — that is useful to know.

## Part A — Start the system

1. Open a terminal in the project folder and type:
   `source .venv/bin/activate`
2. Then type:
   `python run.py`
   Wait until the terminal shows a line starting `[run] dashboard: http://127.0.0.1:8501`
   (about 10–20 seconds).
3. The dashboard opens in your web browser by itself. If it does not, open the browser
   and go to **http://127.0.0.1:8501**.

## Part B — Choose a model

4. Look at the panel on the left, under **Run settings**.
5. In **Model**, pick **Random Forest**. Leave the other settings as they are
   (Task: Binary, Seed: 42, Emission interval Δt: 100 ms, Load: Steady).

## Part C — Start a run

6. Under **Actions**, press **Start**.
7. Check that the run has started. The line under the buttons shows a run number and a
   count of processed records that keeps going up. The settings above are now greyed
   out until the run ends.

## Part D — Find an alert

8. Scroll down to the table called **Latest alerts**. Each row is a connection the
   model flagged as an attack. (If it says "No alerts in this run yet", wait a few
   seconds.)
9. Pick any row and tell the observer:
   - its **Severity** (High or Medium), and
   - what the model **Predicted**.

## Part E — Read the latency

10. Look at the row of boxes at the top of the page. Find the box labelled **L p95**.
    It appears after the first 100 records (about 10 seconds).
11. Tell the observer the number in that box, in milliseconds (ms). It means: 95% of
    connections were checked and stored within this time after they were sent.

## Finish

12. Press **Stop** in the left panel. Then go back to the terminal and press
    **Ctrl+C** to shut the system down.
