"""
features.py - Feature Extraction and ML Pipeline for Phishing URL Threat Detection.
Senior Full-Stack & ML implementation using standard urllib, regex, and scikit-learn.
"""

from __future__ import annotations

import os
import re
import difflib
import ipaddress
import urllib.parse
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# Known URL shortener hostnames
KNOWN_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "mcaf.ee", "rebrand.ly", "lnkd.in",
    "tiny.cc", "soo.gd", "s2r.co", "clickmeter.com", "cutt.ly"
}

# Sensitive phishing and credential-harvesting keywords
PHISHING_KEYWORDS = [
    "login", "verify", "update", "banking", "secure",
    "signin", "account", "security", "confirm", "password",
    "credential", "wallet", "authenticate", "validation", "support",
    "recover", "billing", "access", "lock", "suspend"
]

# Ordered numeric feature column names for scikit-learn models
FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "count_dots",
    "count_hyphens",
    "count_at",
    "count_question",
    "count_equal",
    "count_slash",
    "has_ip_address",
    "is_https",
    "is_shortened",
    "keyword_count",
    "subdomain_depth",
    "has_double_slash_path",
    "count_digits",
]


# Popular brand targets for typosquatting / impersonation defense
POPULAR_TARGET_BRANDS: List[str] = [
    "google.com",
    "paypal.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "netflix.com",
    "facebook.com",
    "instagram.com",
]


def check_brand_typosquatting(
    host: str,
    target_brands: List[str] = POPULAR_TARGET_BRANDS
) -> Tuple[bool, str, float]:
    """
    Check if a domain is a typosquatting / impersonation attempt of popular brands.
    Uses difflib.SequenceMatcher. If similarity ratio > 0.82 and not an exact match,
    flags as typosquatting. Purely lexical without blocking network calls.
    """
    clean = host.split(":")[0].strip().lower()
    if clean.startswith("www."):
        clean = clean[4:]
    if not clean:
        return False, "", 0.0

    # Exact match or authentic subdomain is legitimate
    for brand in target_brands:
        if clean == brand or clean.endswith(f".{brand}"):
            return False, "", 0.0

    host_parts = [p for p in clean.split(".") if p]
    if len(host_parts) >= 2:
        domain_name = host_parts[-2]
    elif host_parts:
        domain_name = host_parts[0]
    else:
        return False, "", 0.0

    best_brand = ""
    highest_ratio = 0.0

    for brand in target_brands:
        brand_name = brand.split(".")[0]
        full_ratio = difflib.SequenceMatcher(None, clean, brand).ratio()
        base_ratio = difflib.SequenceMatcher(None, domain_name, brand_name).ratio()
        max_ratio = max(full_ratio, base_ratio)

        if max_ratio > highest_ratio and max_ratio > 0.82:
            highest_ratio = max_ratio
            best_brand = brand

    if highest_ratio > 0.82 and best_brand:
        return True, best_brand, round(highest_ratio, 2)

    return False, "", 0.0


class URLLexicalExtractor:
    """
    Automated lexical feature extractor for URL threat analysis.
    Uses standard library urllib.parse and regex without external network/DNS dependencies.
    """

    def __init__(self, shorteners: set = None, keywords: list = None):
        self.shorteners = shorteners or KNOWN_SHORTENERS
        self.keywords = keywords or PHISHING_KEYWORDS

    def _normalize_and_parse(self, raw_url: str) -> Tuple[str, urllib.parse.ParseResult, bool]:
        """
        Normalize input URL string, refang brackets like [.] and [dot],
        and parse into components with try-except fallback.
        Returns: (normalized_url, parsed_result, had_explicit_scheme)
        """
        raw_url = raw_url.strip()
        # Refang schemes
        raw_url = re.sub(r"^hxxps?://", lambda m: "https://" if "s" in m.group(0).lower() else "http://", raw_url, flags=re.IGNORECASE)
        raw_url = re.sub(r"^h\*\*ps?://", lambda m: "https://" if "s" in m.group(0).lower() else "http://", raw_url, flags=re.IGNORECASE)
        # Refang brackets
        raw_url = re.sub(r"\[\.\]|\(\.\)|\{\.\}", ".", raw_url)
        raw_url = re.sub(r"\[dot\]|\(dot\)|\{dot\}", ".", raw_url, flags=re.IGNORECASE)
        # Clean rogue non-IPv6 brackets
        raw_url = re.sub(r"\[([^:\]]*)\]", r"\1", raw_url)

        had_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw_url))
        url_to_parse = raw_url if had_scheme else f"http://{raw_url}"

        try:
            parsed = urllib.parse.urlparse(url_to_parse)
        except ValueError:
            # Fallback if invalid IPv6 brackets are present
            safe_url = url_to_parse.replace("[", "").replace("]", "")
            try:
                parsed = urllib.parse.urlparse(safe_url)
            except Exception:
                # Basic fallback
                parsed = urllib.parse.urlparse("http://unknown-host/")

        return raw_url, parsed, had_scheme

    def _check_ip_address(self, hostname: str) -> int:
        """Check if hostname is a raw IPv4 or IPv6 address."""
        clean_host = hostname.split(":")[0].strip("[]")
        try:
            ipaddress.ip_address(clean_host)
            return 1
        except ValueError:
            # Fallback regex for standard IPv4 formats
            ip_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
            return 1 if re.match(ip_pattern, clean_host) else 0

    def _calculate_subdomain_depth(self, hostname: str) -> int:
        """Calculate subdomain depth count."""
        clean_host = hostname.split(":")[0]
        if self._check_ip_address(clean_host):
            return 0

        parts = [p for p in clean_host.split(".") if p]
        if len(parts) <= 2:
            return 0

        # Adjust for two-part country TLDs (e.g. .co.uk, .com.au)
        common_compound_tlds = {"co.uk", "com.au", "gov.uk", "ac.uk", "org.uk", "co.jp", "com.br"}
        if len(parts) >= 3 and f"{parts[-2]}.{parts[-1]}".lower() in common_compound_tlds:
            return max(0, len(parts) - 3)

        return max(0, len(parts) - 2)

    def extract(self, url: str) -> Dict[str, Any]:
        """
        Extract numerical lexical features for model inference.
        Returns a dictionary conforming to FEATURE_NAMES.
        """
        normalized_url, parsed, had_scheme = self._normalize_and_parse(url)
        hostname = (parsed.netloc or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
        full_text = normalized_url.lower()

        # Keyword search across hostname, path, and query
        detected_keywords = sum(
            1 for kw in self.keywords if kw in full_text
        )

        # Check for URL shorteners
        clean_host_for_shortener = hostname.split(":")[0]
        is_short = 1 if clean_host_for_shortener in self.shorteners else 0

        # Protocol check (strictly HTTPS if explicit, or defaults to 0 if HTTP / missing scheme)
        is_https_flag = 1 if (had_scheme and parsed.scheme.lower() == "https") else 0

        # Special character counters
        features = {
            "url_length": len(normalized_url),
            "hostname_length": len(hostname),
            "path_length": len(path),
            "count_dots": normalized_url.count("."),
            "count_hyphens": normalized_url.count("-"),
            "count_at": normalized_url.count("@"),
            "count_question": normalized_url.count("?"),
            "count_equal": normalized_url.count("="),
            "count_slash": normalized_url.count("/"),
            "has_ip_address": self._check_ip_address(hostname),
            "is_https": is_https_flag,
            "is_shortened": is_short,
            "keyword_count": detected_keywords,
            "subdomain_depth": self._calculate_subdomain_depth(hostname),
            "has_double_slash_path": 1 if "//" in path else 0,
            "count_digits": sum(c.isdigit() for c in normalized_url),
        }
        return features

    def get_heuristic_flags(self, url: str, features: Dict[str, Any] = None) -> List[str]:
        """
        Evaluate heuristic red flags and return human-readable alerts for UI checklist.
        """
        if features is None:
            features = self.extract(url)

        normalized_url, parsed, had_scheme = self._normalize_and_parse(url)
        hostname = (parsed.netloc or "").lower()
        full_text = normalized_url.lower()
        flags: List[str] = []

        # 1. Raw IP Hostname
        if features["has_ip_address"]:
            flags.append(f"Uses raw IP address instead of domain name: `{hostname}`")

        # 2. Protocol Check
        if not features["is_https"]:
            if had_scheme and parsed.scheme.lower() == "http":
                flags.append("Lacks HTTPS encryption (Cleartext HTTP connection)")
            else:
                flags.append("No secure HTTPS protocol specified")

        # 3. URL Shortener
        if features["is_shortened"]:
            flags.append(f"Known URL shortener detected (`{hostname}`), concealing true destination")

        # 4. Phishing Keywords
        found_kws = [kw for kw in self.keywords if kw in full_text]
        if found_kws:
            flags.append(f"Contains sensitive phishing keyword(s): {', '.join(f'`{kw}`' for kw in found_kws)}")

        # 5. Suspicious Subdomain Depth
        if features["subdomain_depth"] >= 2:
            flags.append(f"Excessive subdomain depth ({features['subdomain_depth']} subdomains) often used for brand impersonation")

        # 6. '@' Symbol in URL
        if features["count_at"] > 0:
            flags.append("Contains '@' symbol (Browser credential injection / redirect exploit)")

        # 7. Double Slash in Path
        if features["has_double_slash_path"]:
            flags.append("Contains double slash (`//`) in URL path (Redirection evasion tactic)")

        # 8. Excessive URL Length
        if features["url_length"] > 75:
            flags.append(f"Abnormal URL length ({features['url_length']} characters) intended to push payload off screen")

        # 9. Excessive Hyphens
        if features["count_hyphens"] >= 3:
            flags.append(f"High hyphen count ({features['count_hyphens']}), characteristic of typosquatting")

        # 10. Query Parameter Density
        if features["count_equal"] >= 3 or features["count_question"] >= 2:
            flags.append("High query parameter density used to inject tracking tokens or payloads")

        return flags


def generate_synthetic_dataset() -> Tuple[List[str], List[int]]:
    """
    Generates a balanced synthetic seed dataset of legitimate and phishing URLs
    with realistic variations for instant out-of-the-box model training.
    Returns: (urls, labels) where 0 = Legitimate, 1 = Phishing.
    """
    urls: List[str] = []
    labels: List[int] = []

    # 1. Base Legitimate Seeds
    safe_bases = [
        "https://www.google.com",
        "https://www.google.com/search?q=machine+learning+tutorial",
        "https://en.wikipedia.org/wiki/Cybersecurity",
        "https://en.wikipedia.org/wiki/Random_forest",
        "https://en.wikipedia.org/wiki/Transport_Layer_Security",
        "https://github.com",
        "https://github.com/torvalds/linux",
        "https://github.com/scikit-learn/scikit-learn/pulls",
        "https://github.com/streamlit/streamlit/releases",
        "https://stackoverflow.com/questions/1234567/python-url-parsing",
        "https://stackoverflow.com/tags/machine-learning/info",
        "https://www.nytimes.com/section/technology",
        "https://www.nytimes.com/2026/10/20/business/economy.html",
        "https://www.amazon.com/dp/B08N5WRWNW",
        "https://www.amazon.com/gp/help/customer/display.html",
        "https://docs.python.org/3/library/urllib.parse.html",
        "https://docs.python.org/3/library/re.html",
        "https://news.ycombinator.com/item?id=38192014",
        "https://news.ycombinator.com/newsguidelines.html",
        "https://www.microsoft.com/en-us/windows",
        "https://www.microsoft.com/en-us/surface/devices",
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview",
        "https://developer.mozilla.org/en-US/docs/Web/Security",
        "https://www.apple.com/iphone-16-pro/",
        "https://www.apple.com/macos/sequoia/",
        "https://arxiv.org/abs/2301.00001",
        "https://medium.com/@engineering-lead/clean-architecture-in-python",
        "https://www.reddit.com/r/MachineLearning/",
        "https://www.reddit.com/r/Python/comments/xyz123/",
        "https://www.linkedin.com/feed/",
        "https://www.linkedin.com/learning/browse",
        "https://hub.docker.com/_/python",
        "https://pypi.org/project/scikit-learn/",
        "https://pypi.org/project/streamlit/",
        "https://www.cloudflare.com/learning/security/what-is-phishing/",
        "https://blog.cloudflare.com/introducing-new-ddos-protections/",
        "https://www.bbc.com/news/world-us-canada",
        "https://www.cnn.com/world",
        "https://www.nature.com/articles/d41586-024-00001-x",
        "https://www.coursera.org/learn/machine-learning",
        "https://www.kaggle.com/datasets",
        "https://huggingface.co/models",
    ]

    for base in safe_bases:
        urls.append(base)
        labels.append(0)

    # Programmatic variations of safe URLs
    safe_domains = [
        "https://www.google.com", "https://github.com", "https://stackoverflow.com",
        "https://docs.python.org", "https://developer.mozilla.org", "https://www.amazon.com",
        "https://en.wikipedia.org", "https://pypi.org", "https://www.microsoft.com",
        "https://www.cloudflare.com", "https://hub.docker.com", "https://www.reddit.com"
    ]
    safe_subpaths = [
        "/about", "/contact-us", "/pricing", "/features", "/documentation/v2",
        "/search?category=books&sort=desc", "/products/item-849302", "/blog/post/how-to-code",
        "/terms-of-service", "/privacy-policy", "/faq/general", "/explore/trending"
    ]
    for dom in safe_domains:
        for sub in safe_subpaths[:8]:
            urls.append(f"{dom}{sub}")
            labels.append(0)

    # 2. Base Phishing Seeds
    phish_bases = [
        # Raw IP addresses
        "http://192.168.1.1/secure-login/bank/verify.php",
        "http://192.168.0.101/signin-account-verification/",
        "http://45.33.32.156/secure/bank/verify.php?token=92834",
        "http://218.45.12.9/update-account-credentials/webscr.php",
        "http://104.244.42.1/webscr?cmd=_login-run&dispatch=auth",
        "http://89.208.29.11/chase/online-banking/login.html",
        "http://178.62.204.10/apple-support-unlock/verify.php",
        "http://194.87.139.12/paypal-security-update/login.htm",
        "http://185.220.101.4/account-suspended-action-required/",
        "http://198.51.100.45/citibank-online-login/auth.php",

        # Impersonation & Typosquatting
        "http://paypal-security-update.com.verify-auth.xyz/signin?account=locked",
        "http://apple-id-verify.account-unlock.xyz/security-check",
        "http://secure-login.bankofamerica.com.evil-redirect.net/update",
        "http://login.microsoftonline.com-security.support/auth",
        "http://chase-bank-verify-identity.info/login.aspx",
        "http://wellsfargo-account-reactivate.biz/secure/login",
        "http://amazon-account-suspension-review.cc/billing-update",
        "http://netflix-billing-update-center.tk/verify-card",
        "http://secure.paypal.com-webscr.cc/cgi-bin/webscr?cmd=_login",
        "http://login.yahoo.com-account-recovery.support/auth/index.php",
        "http://facebook-security-verification-team.site/login",
        "http://google-drive-shared-file-access.top/signin",
        "http://dropbox-document-viewer-secure.club/authenticate",

        # URL Shorteners with phishing context
        "http://bit.ly/secure-login-bank-verify",
        "http://tinyurl.com/account-suspended-update",
        "http://t.co/urgent-security-action-login",
        "http://bit.ly/chase-online-verify-2024",
        "http://tinyurl.com/paypal-dispute-resolve",
        "http://is.gd/bank_account_confirm",
        "http://buff.ly/secure-wallet-validation",
        "http://rebrand.ly/billing-update-warning",

        # Credential @ injection & double slashes
        "http://google.com@evil-site.com/login-verification",
        "http://paypal.com@verify-identity-security.info/account",
        "http://apple.com@id-recovery-server.xyz/auth/step1",
        "http://legit-portal.com//phishing-payload/login.php",
        "http://secure-banking-portal.net//redirect/verify?id=9482",

        # Multi-subdomain nesting & keyword cramming
        "http://secure.account.update.verify.paypal.com.fraud-network.net/login",
        "http://signin.ebay.com.account-protection.tk/login.php",
        "http://login.live.com.id-authentication-service.click/auth.htm",
        "http://verify.wallet.crypto-auth-center.online/signin?token=18237",
        "http://account.update.security.bank.portal.xyz/login.aspx",
    ]

    for base in phish_bases:
        urls.append(base)
        labels.append(1)

    # Programmatic variations of phishing patterns
    phish_fake_domains = [
        "http://verify-bank-security-center.com",
        "http://account-update-alert.xyz",
        "http://secure-online-signin-portal.net",
        "http://billing-issue-action-required.org",
        "http://paypal-support-ticket-resolution.info",
        "http://appleid-security-validation.live",
        "http://microsoft-outlook-credential-verify.top",
        "http://wallet-security-phrase-validation.online",
        "http://auth-login-account-recovery.click",
        "http://chase-security-fraud-alert.vip",
        "http://wells-fargo-banking-update.support",
        "http://citi-account-reactivation.space"
    ]
    phish_payload_paths = [
        "/login.php?session=expired&ref=secure",
        "/verify/identity-documents/account/update",
        "/signin/password-reset?token=xyz982348234",
        "/banking/customer-service/security-check.aspx",
        "/webscr?cmd=_login_verification&auth_token=847294",
        "/credential-recovery/step1/confirm.html",
        "/wallet/auth/restore-passphrase-validation",
        "/security-alert/suspend-prevention.php?user=alert"
    ]
    for dom in phish_fake_domains:
        for pth in phish_payload_paths:
            urls.append(f"{dom}{pth}")
            labels.append(1)

    # More IP variations
    for ip in ["192.168.1.50", "45.76.12.89", "185.193.65.22", "194.26.29.110"]:
        for pth in phish_payload_paths[:4]:
            urls.append(f"http://{ip}{pth}")
            labels.append(1)

    return urls, labels


def train_phishing_model(
    model_path: str = None,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Train a lightweight RandomForestClassifier using the synthetic seed dataset.
    Saves model to model_path if provided.
    Returns: (model, evaluation_metrics)
    """
    urls, labels = generate_synthetic_dataset()
    extractor = URLLexicalExtractor()

    # Extract features for all dataset samples
    feature_records = [extractor.extract(u) for u in urls]
    df = pd.DataFrame(feature_records)[FEATURE_NAMES]
    y = np.array(labels)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.25, random_state=random_state, stratify=y
    )

    # Initialize and fit RandomForest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    metrics = {
        "accuracy": acc,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(df),
    }

    if model_path:
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        joblib.dump(model, model_path)

    return model, metrics


def load_or_train_model(model_path: str = "model.pkl") -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Load pre-trained model from disk if available, or train a new one and persist it.
    """
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            metrics = {"status": "loaded_from_disk", "model_path": model_path}
            return model, metrics
        except Exception:
            pass  # Fall back to training if file corrupted

    return train_phishing_model(model_path=model_path)


def predict_url(
    url: str,
    model: RandomForestClassifier,
    extractor: URLLexicalExtractor = None
) -> Dict[str, Any]:
    """
    Comprehensive inference pipeline for a single URL:
    Extracts features, runs RF model prediction, computes risk score,
    and collects heuristic red flags.
    """
    extractor = extractor or URLLexicalExtractor()
    features = extractor.extract(url)
    heuristic_flags = extractor.get_heuristic_flags(url, features)

    # Prepare DataFrame for scikit-learn
    df_features = pd.DataFrame([features])[FEATURE_NAMES]

    # Model inference
    prediction_class = int(model.predict(df_features)[0])
    probabilities = model.predict_proba(df_features)[0]

    # Phishing risk score is the probability of class 1 (Phishing)
    # Binary classes: [P(Safe), P(Phishing)]
    phishing_prob = float(probabilities[1]) if len(probabilities) > 1 else float(prediction_class)
    risk_score_pct = round(phishing_prob * 100.0, 1)

    # Check for Typosquatting / Brand Impersonation
    normalized_url, parsed, _ = extractor._normalize_and_parse(url)
    hostname = (parsed.netloc or "").lower()
    is_typosquat, spoofed_brand, sim_score = check_brand_typosquatting(hostname)
    if is_typosquat:
        heuristic_flags.insert(0, f"⚠️ Possible Typosquatting / Brand Impersonation of {spoofed_brand}")
        risk_score_pct = max(risk_score_pct, 78.5)
        status_label = "Phishing"
        status_color = "danger"
    elif risk_score_pct >= 60.0 or prediction_class == 1:
        status_label = "Phishing"
        status_color = "danger"
    elif risk_score_pct >= 35.0 or len(heuristic_flags) >= 2:
        status_label = "Suspicious"
        status_color = "warning"
    else:
        status_label = "Legitimate"
        status_color = "success"

    return {
        "url": url,
        "status": status_label,
        "status_color": status_color,
        "is_phishing": bool(prediction_class == 1 or risk_score_pct >= 50.0),
        "risk_score": risk_score_pct,
        "confidence_safe": round(float(probabilities[0]) * 100.0, 1),
        "confidence_phishing": risk_score_pct,
        "heuristic_flags": heuristic_flags,
        "features": features,
    }


if __name__ == "__main__":
    print("[*] Initializing Phishing Threat Detection Pipeline...")
    model, metrics = train_phishing_model("model.pkl")
    print(f"[+] Model trained successfully. Metrics: {metrics}")

    test_urls = [
        "https://en.wikipedia.org/wiki/Computer_security",
        "http://192.168.1.1/secure-login/bank/verify.php",
        "http://bit.ly/secure-login-bank",
        "https://github.com/scikit-learn/scikit-learn",
        "http://paypal-security-update.com.verify-auth.xyz/signin?account=locked"
    ]

    print("\n[*] Running diagnostic tests:")
    for test_url in test_urls:
        result = predict_url(test_url, model)
        print(f"\nURL: {test_url}")
        print(f"Status: {result['status']} | Risk Score: {result['risk_score']}%")
        print(f"Heuristic Flags ({len(result['heuristic_flags'])}): {result['heuristic_flags']}")
