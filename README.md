# PhishGuard — Real-Time Phishing URL Threat Detection App

An end-to-end, lightweight cybersecurity application engineered with **Python**, **Scikit-learn**, and **Streamlit** to detect malicious hyperlinks, deceptive brand spoofing, and credential harvesting landing pages in real time.

---

## 🚀 Key Highlights & Architecture

- **100% Lexical & Zero-Network Latency**: Parses URLs using Python's standard `urllib.parse`, `re`, and `ipaddress`. No external DNS resolution, WHOIS queries, or network lookups are performed, preventing attackers from logging scanner probes and ensuring sub-millisecond inference.
- **Embedded Seed Dataset & Auto-Training**: Contains an integrated realistic synthetic dataset generator (400+ legitimate and attack variations). The app trains a `RandomForestClassifier` on initial launch and caches it (`model.pkl`), ensuring immediate out-of-the-box execution without external CSV downloads.
- **Probabilistic Risk Scoring**: Produces both binary decisions (`Phishing` vs `Legitimate`) and continuous confidence metrics (`0.0% – 100.0% Risk`).
- **Interactive Security Dashboard**: Modern cybersecurity HUD built in Streamlit featuring instant one-click scenario presets, color-coded threat badges, dynamic risk gauges, an itemized heuristic red-flags checklist, and deep-dive lexical metric inspection.
- **Dual Execution Modes**: Structured for modular production (`features.py` + `app.py`) as well as single-file portability (`single_file_app.py`).

---

## 🛠️ Project Structure

```text
phishing_detector/
├── features.py          # Lexical feature extractor, heuristic flags, synthetic dataset generator, RF training
├── app.py               # Modern Streamlit UI dashboard (modular: imports features.py)
├── single_file_app.py   # Standalone 1-file version with embedded extractor and model pipeline
├── requirements.txt     # Minimal dependencies (streamlit, scikit-learn, pandas, numpy, joblib)
└── README.md            # Comprehensive documentation & engineering guide
```

---

## 🔬 Monitored Lexical Indicators (16 Features)

| Feature | Extraction Method | Threat Heuristic Rationale |
| :--- | :--- | :--- |
| **`url_length`** | `len(url)` | Long URLs (>75 chars) are frequently crafted to push deceptive elements past mobile browser viewports. |
| **`hostname_length`** | `len(parsed.netloc)` | Excessively long hostnames are typical of DNS spoofing and multi-level subdomains. |
| **`path_length`** | `len(parsed.path)` | Deep folder structures used to hide phishing scripts (`/login/bank/verify.php`). |
| **`count_dots`** | `url.count('.')` | Multiple dot separators commonly indicate subdomain impersonation (`login.paypal.com.account-update.xyz`). |
| **`count_hyphens`** | `url.count('-')` | High hyphen count indicates typosquatting (e.g. `apple-security-check-login.com`). |
| **`count_at`** | `url.count('@')` | `@` in URL tricks browser into treating prefix as credentials and routing to hostname after `@`. |
| **`count_slash`** | `url.count('/')` | Deep directory nesting for malware dropping or phishing pages. |
| **`count_question` / `count_equal`** | `url.count('?')`, `url.count('=')` | Query parameter density used for tracking victim tokens and session poisoning. |
| **`has_ip_address`** | `ipaddress.ip_address()` | Raw IPv4/IPv6 hostname directly targets unindexed servers without domain registration. |
| **`is_https`** | `parsed.scheme == 'https'` | Absence of TLS encryption indicates insecure or rogue cleartext transport. |
| **`is_shortened`** | Known shortener domain match | URL shorteners (`bit.ly`, `tinyurl.com`, `t.co`) mask true landing endpoints. |
| **`keyword_count`** | Regex search for sensitive terms | Presence of credential harvester words (`login`, `verify`, `banking`, `secure`, `update`, `signin`). |
| **`subdomain_depth`** | Dot segments minus TLD parts | Excessive subdomain hierarchy mimicking target brand trust. |
| **`has_double_slash_path`** | `'//' in parsed.path` | Open redirection evasion pattern. |
| **`count_digits`** | `sum(c.isdigit())` | High numeric density common in dynamically generated disposable attack domains. |

---

## ⚡ Quickstart & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application (Option A: Modular Architecture)
```bash
streamlit run app.py
```

### 3. Run Application (Option B: Standalone Single File)
```bash
streamlit run single_file_app.py
```

### 4. Optional: CLI Diagnostic Mode
You can also run `features.py` directly to benchmark extraction, train the model, and view diagnostic predictions on sample URLs:
```bash
python features.py
```
