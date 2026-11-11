# Methodology presentation (slide text)

--- slide 1
 METHODOLOGY PRESENTATION
DEVELOPMENT OF A REAL-TIME
NETWORK INTRUSION DETECTION
SYSTEM USING MACHINE LEARNING
PRESENTED BY
Kayode Akinola Samuel  •  236336
SUPERVISOR
Mr. I. O. Adeyemi
Department of Computer Science  |  University of Ibadan  |  October 2026
--- slide 2
Introduction
01 / 16
Modern networks face growing threats such as Distributed Denial of Service attacks, port scanning, brute-force login attempts, and botnet activity (Magar et al., 2025; Wijethilaka et al., 2025). Traditional signature-based Intrusion Detection Systems (IDS) struggle to identify unknown or zero-day threats due to their reliance on static rule databases, making behavioral machine learning essential for detecting suspicious network activity (Arreche et al., 2024; Suresh Kumar et al., 2026).
Despite advances in machine learning-based intrusion detection, most studies focus on offline accuracy rather than real-time performance. This project addresses this gap by developing a real-time NIDS using Isolation Forest, Random Forest, and LSTM models on the IDS2025 dataset, while evaluating detection accuracy, latency, and throughput.
RESEARCH GAP
Most behavioral IDS research reports detection accuracy from offline, static evaluation only, without an accompanying system proving the model can classify traffic within real-time latency bounds. This project treats real-time operation as a design requirement, not an assumption.


--- slide 3
Literature review
Author(s), year | Focus | Methodology | Research gap
Maseer et al. (2021) | Benchmarking ML for anomaly-based IDS | 10 algorithms on CICIDS2017 | No real-time streaming deployment
Zhang et al. (2023) | Real-time detection over SD-WAN | Online Isolation Forest | Isolation Forest only; no RF or LSTM comparison
Wijethilaka et al. (2025) | Low-cost NIDS on Raspberry Pi | RF, LSTM, ANN, XGBoost with SMOTE | 50 ms response shown; no live dashboard
Magar et al. (2025) | Hybrid RF-LSTM detection | CICIDS2017 with SMOTE | Offline batch evaluation only
Panigrahi and Borah (2025) | IDS2025 refined dataset | Rebalanced, cleaned CICIDS2017 | Few published benchmarks yet
3
--- slide 4
Problem statement
The problem
Signature-based IDS cannot identify attacks with no known pattern. Behavioural ML addresses this but brings higher false-positive rates and no demonstrated real-time viability.
Existing state
Most studies stop at offline, static evaluation. They omit end-to-end latency, throughput and a live dashboard, and there is no pipeline comparing three detectors on IDS2025.
Proposed solution
A real-time streaming pipeline that evaluates Isolation Forest, Random Forest and LSTM, measuring detection accuracy and system latency and throughput together.
4
--- slide 5
Aim and objectives
 Aim
Develop a real-time network intrusion detection system that applies machine learning to detect malicious traffic, evaluating both detection accuracy and real-time system performance.
Objective 1
Develop and compare Isolation Forest, Random Forest and LSTM models.
Objective 2
Implement a real-time streaming pipeline with an inference engine and live monitoring dashboard.
Objective 3
Evaluate with Precision, Recall, F1-score, ROC-AUC, latency and throughput.
5
--- slide 6
Proposed methodology: DSRM
6
--- slide 7
System architecture
Data layer: IDS2025 streamer replays records at a set interval.
Transport: local TCP socket, one JSON message per record.
Processing: inference engine with saved scaler and one detector interface for all three models.
Storage: SQLite log and latency logger.
Presentation: Streamlit dashboard and control channel.
7
--- slide 8
Use case diagram
Actors: network administrator (primary); IDS2025 dataset source (secondary).
UC1 Start or stop stream (includes UC7)
UC2 Select model
UC3 Set emission rate
UC4 View alerts
UC5 View latency and throughput
UC6 View live traffic and protocol mix
UC7 Provide labelled flow records
8
--- slide 9
Data flow diagram: Level 0 and Level 1
Level 0: streamer (1.0), detector (2.0), logger (3.0) and dashboard (4.0), with stores D1 to D3.
Level 1: inference splits into receive (t1), validate, scale (t2), predict (t3) and build result.
9
--- slide 10
Research design
Design science: the artefact is a real-time NIDS, evaluated on detection quality and end-to-end latency.
CRISP-DM governs the data and model work inside DSRM stage 3.
Iterative: Waterfall was rejected because latency and tuning unknowns only emerge by building and measuring.
Replay, not live capture: the IDS2025 test partition is streamed over a local socket on one machine, with five seeds per model.
10
--- slide 11
Dataset acquisition
Source
IDS2025 (Panigrahi and Borah, 2025): a refined CICIDS2017, free on Mendeley Data. Balanced, cleaned and relabelled.
Structure
Daily CSV files, Monday to Friday. About 80 CICFlowMeter features per network flow.
Classes
Normal, Botnet, Brute Force, DoS/DDoS, Infiltration, PortScan and Web Attack.
Labels and split
Binary label for the main comparison; seven-class label for RF and LSTM. Stratified 70/15/15 split, seed 42.
11
--- slide 12
Data preprocessing
Merge daily files; remove duplicate rows
Replace infinite values; drop or median-fill missing values
Drop constant, identifier and highly correlated (above 0.98) columns
Stratified split before any scaling
Standardise with training statistics; log(1 + x) for skewed features
12
--- slide 13
Model development
 | Isolation Forest | Random Forest | LSTM
Learning | Unsupervised | Supervised ensemble | Supervised, sequential
Trained on | Normal records only | All labelled records | Windows of W records
Tuning | Grid search | Randomised search, 5-fold CV | Random search, early stopping
Decision | Score threshold | Probability threshold | Output threshold
All three share partitions and scaler. Five seeds (42 to 46) give mean and standard deviation. Class weights handle imbalance; thresholds are set on validation data.
13
--- slide 14
Implementation and software tools
Data and ML
Python 3, pandas and NumPy for data; scikit-learn for Isolation Forest and Random Forest; TensorFlow/Keras for the LSTM.
Streaming and storage
Python socket and json for the TCP stream; SQLite for the log; joblib for saved models and scaler.
Dashboard
Streamlit web dashboard with live traffic, alerts, protocol mix and latency panels.
Development & Version Control
Visual Studio Code, Github and Git.
14
--- slide 15
Model evaluation
Detection metrics
Precision, recall, F1-score and ROC-AUC; macro-F1 and per-class recall for seven classes.
System metrics
End-to-end latency (median, p95, p99) and throughput. Target: 95% of records with L at most 100 ms.
Test scenarios
S1 baseline 100 ms; S2 high rate 10 ms; S3 rate search to 1 ms; S4 burst of 1,000 records.
15
--- slide 16
Streaming Producer: Stamps each network flow with a timestamp (t₀), converts it into JSON format, and transmits it through a local TCP socket.
Inference engine: Validates incoming flow data, applies the saved feature scaler, and classifies traffic using the selected machine learning model.
Database Logger: Stores prediction results, threat alerts, and latency metrics in SQLite, recording the final timestamp (t₄).
Streamlit Dashboard: Retrieves live data from SQLite to display threats and performance metrics while allowing users to select models, adjust streaming rates, and control traffic streaming.
System integration
16
--- slide 17
Conclusion
This study adopts Design Science Research Methodology (DSRM), integrating CRISP-DM to guide the development and evaluation of a real-time Network Intrusion Detection System (NIDS). The proposed system uses a layered streaming architecture, a common interface for Isolation Forest, Random Forest, and LSTM models, and an SQLite database for recording predictions and performance metrics. The methodology emphasizes both detection accuracy and real-time processing efficiency, particularly latency and throughput. Chapter Four presents the system implementation, testing, and experimental results obtained on a single machine.

17
--- slide 18
References
Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
Chapman, P. et al. (2000). CRISP-DM 1.0: Step-by-step data mining guide. SPSS Inc.
Hevner, A. R. et al. (2004). Design science in information systems research. MIS Quarterly, 28(1), 75-105.
Hochreiter, S., and Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.
Liu, F. T., Ting, K. M., and Zhou, Z.-H. (2008). Isolation forest. IEEE ICDM, 413-422.
Maseer, Z. K. et al. (2021). Benchmarking of machine learning for anomaly-based IDS in the CICIDS2017 dataset. IEEE Access, 9, 22351-22370.
Panigrahi, R., and Borah, S. (2025). IDS2025 [Data set]. Mendeley Data, V1.
Peffers, K. et al. (2007). A design science research methodology for information systems research. J. Management Information Systems, 24(3), 45-77.
Wijethilaka et al. (2025). A cost effective machine learning based network intrusion detection system using Raspberry Pi for real time analysis.
Zhang, P. et al. (2023). Real-time malicious traffic detection with online isolation forest over SD-WAN. IEEE TIFS, 18, 2076-2090.
18
--- slide 19
THANK YOU
DeDEVELOPMENT OF A REAL-TIME NETWORK INTRUSION DETECTION SYSTEM USING MACHINE LEARNING

Kayode Akinola Samuel  |  Matric No: 236336
Supervised by: Mr. I.O. Adeyemi
Department of Computer Science, Faculty of Computing, University of Ibadan  |  October 2026
13
