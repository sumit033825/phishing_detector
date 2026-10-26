"""
app.py - Link Safety Checker
A friendly, human-centered web app to help everyday people check unfamiliar links
before clicking. Powered by Scikit-learn and Streamlit.
"""

from __future__ import annotations

import os
import sys
import re
import difflib
import urllib.parse
from typing import List, Dict, Any, Tuple

import pandas as pd
import streamlit as st

# Ensure local directory is in path for modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features import (
    URLLexicalExtractor,
    FEATURE_NAMES,
    load_or_train_model,
    predict_url,
)

# -----------------------------------------------------------------------------
# PAGE SETUP & FRIENDLY THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Link Safety Checker — Check Before You Click",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Calm, Human-Centered Dark/Navy Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #121826 !important;
        color: #F1F5F9 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Cards */
    .app-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    /* Traffic-Light Verdict Cards */
    .verdict-box-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, #1E293B 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 6px solid #10B981;
    }
    .verdict-box-caution {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, #1E293B 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 6px solid #F59E0B;
    }
    .verdict-box-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.14) 0%, #1E293B 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-left: 6px solid #EF4444;
    }

    .verdict-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .badge-safe {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-caution {
        background: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.45);
    }

    .verdict-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin-bottom: 8px;
        line-height: 1.35;
    }
    .advice-callout {
        background: rgba(0, 0, 0, 0.25);
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.96rem;
        line-height: 1.55;
        margin-top: 14px;
        color: #E2E8F0;
        border-left: 3px solid rgba(255, 255, 255, 0.2);
    }

    /* URL Inspector Pills */
    .inspector-pill {
        background: #0F172A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        height: 100%;
    }
    .inspector-label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .inspector-val {
        font-size: 1.05rem;
        font-weight: 600;
        color: #F8FAFC;
        word-break: break-all;
        font-family: 'JetBrains Mono', monospace;
    }
    .inspector-sub {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-top: 6px;
    }

    /* What We Found Lists */
    .finding-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 0.92rem;
        line-height: 1.45;
        margin-bottom: 8px;
    }
    .finding-safe {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #D1FAE5;
    }
    .finding-danger {
        background: rgba(239, 68, 68, 0.09);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #FEE2E2;
    }

    /* Text Input & Buttons */
    div[data-testid="stTextInput"] input {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        padding: 14px 18px !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.25) !important;
    }

    div[data-testid="stButton"] > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: #1E293B !important;
        color: #F8FAFC !important;
        padding: 10px 16px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #38BDF8 !important;
        background: #334155 !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: #2563EB !important;
        border: none !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #1D4ED8 !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED MODEL ENGINE (UNDER THE HOOD)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Checking security model...")
def get_cached_detector():
    """Load or train model in background."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    model, metrics = load_or_train_model(model_path=model_path)
    extractor = URLLexicalExtractor()
    return model, extractor, metrics


model, extractor, _ = get_cached_detector()


# -----------------------------------------------------------------------------
# HUMAN-FRIENDLY HELPERS & BRAND TYPOSQUATTING DETECTION
# -----------------------------------------------------------------------------
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


def extract_domain_base(host: str) -> str:
    """
    Extract clean host without port or leading 'www.'.
    Safely handles single-character domains and compound domains without network calls.
    """
    clean = host.split(":")[0].strip().lower()
    if clean.startswith("www."):
        clean = clean[4:]
    return clean


def check_brand_typosquatting(
    host: str,
    target_brands: List[str] = POPULAR_TARGET_BRANDS
) -> Tuple[bool, str, float]:
    """
    Checks whether a domain is a typosquatting or impersonation attempt of popular brands.
    Uses difflib.SequenceMatcher. If ratio > 0.82 and not an exact match, flags as typosquatting.
    Returns: (is_typosquat, matched_brand, similarity_ratio)
    """
    clean_host = extract_domain_base(host)
    if not clean_host:
        return False, "", 0.0

    # Exact matches or valid subdomains of the authentic brand are safe (e.g. google.com or mail.google.com)
    for brand in target_brands:
        if clean_host == brand or clean_host.endswith(f".{brand}"):
            return False, "", 0.0

    # Safe domain name extraction (works cleanly on single-character domains like o.com as well)
    host_parts = [p for p in clean_host.split(".") if p]
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

        # Compare full domain string (e.g., 'gogle.com' vs 'google.com')
        full_ratio = difflib.SequenceMatcher(None, clean_host, brand).ratio()
        # Compare core brand root name (e.g., 'gogle' vs 'google', 'paypa1' vs 'paypal')
        base_ratio = difflib.SequenceMatcher(None, domain_name, brand_name).ratio()
        max_ratio = max(full_ratio, base_ratio)

        if max_ratio > highest_ratio and max_ratio > 0.82:
            highest_ratio = max_ratio
            best_brand = brand

    if highest_ratio > 0.82 and best_brand:
        return True, best_brand, round(highest_ratio, 2)

    return False, "", 0.0


def clean_input_link(raw_url: str) -> str:
    """
    Clean, refang, and normalize the input link so urllib.parse never crashes
    on defanged brackets like [.], (.), [dot], or invalid IPv6 brackets.
    """
    cleaned = raw_url.strip()
    if not cleaned:
        return ""

    # 1. Normalize common defanged schemes (hxxp -> http, hxxps -> https)
    cleaned = re.sub(r"^hxxps?://", lambda m: "https://" if "s" in m.group(0).lower() else "http://", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^h\*\*ps?://", lambda m: "https://" if "s" in m.group(0).lower() else "http://", cleaned, flags=re.IGNORECASE)

    # 2. Refang defanged brackets: [.], (.), {.} -> .
    cleaned = re.sub(r"\[\.\]|\(\.\)|\{\.\}", ".", cleaned)

    # 3. Refang [dot], (dot), {dot} -> .
    cleaned = re.sub(r"\[dot\]|\(dot\)|\{dot\}", ".", cleaned, flags=re.IGNORECASE)

    # 4. Remove rogue non-IPv6 brackets (prevents "ValueError: Invalid IPv6 URL")
    cleaned = re.sub(r"\[([^:\]]*)\]", r"\1", cleaned)

    # 5. Make sure plain links without https:// still parse cleanly
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", cleaned):
        cleaned = f"http://{cleaned}"
    return cleaned


def parse_link_parts(normalized_url: str) -> Dict[str, str]:
    """
    Break link into simple, human-understandable pieces.
    Includes a try-except fallback block around urlsplit so invalid brackets never crash the app.
    """
    try:
        parsed = urllib.parse.urlsplit(normalized_url)
        scheme = (parsed.scheme or "http").lower()
        host = parsed.netloc or "Unknown Website"
        path = parsed.path if parsed.path else "/"
        if parsed.query:
            path += f"?{parsed.query}"
    except ValueError:
        # Fallback if urlsplit encounters malformed brackets or invalid IPv6
        safe_url = normalized_url.replace("[", "").replace("]", "")
        try:
            parsed = urllib.parse.urlsplit(safe_url)
            scheme = (parsed.scheme or "http").lower()
            host = parsed.netloc or "Unknown Website"
            path = parsed.path if parsed.path else "/"
            if parsed.query:
                path += f"?{parsed.query}"
        except Exception:
            # Extreme fallback using simple regex
            match = re.match(r"^([a-zA-Z]+)://([^/?#]+)(.*)$", safe_url)
            if match:
                scheme = match.group(1).lower()
                host = match.group(2)
                path = match.group(3) or "/"
            else:
                scheme = "http"
                host = safe_url
                path = "/"

    return {
        "scheme": scheme,
        "is_https": scheme == "https",
        "host": host,
        "path": path,
    }


def generate_plain_english_findings(
    url: str,
    features: Dict[str, Any],
    risk_score: float
) -> Tuple[List[str], List[str]]:
    """
    Translates technical feature numbers into clear, plain-English observations
    that any everyday user can understand immediately.
    """
    good_findings: List[str] = []
    red_flags: List[str] = []

    # 1. Connection Security
    if features.get("is_https", 0) == 1:
        good_findings.append("Uses a standard secure connection (HTTPS)")
    else:
        red_flags.append("Not using a secure connection (HTTP instead of HTTPS)")

    # 2. Website Name & Hosting
    if features.get("has_ip_address", 0) == 1:
        red_flags.append("Uses an IP number (like 192.168.1.1) instead of a real website name")
    else:
        good_findings.append("Belongs to a recognized website name format")

    # 3. Short links / Concealment
    if features.get("is_shortened", 0) == 1:
        red_flags.append("Uses a shortened link (like bit.ly) which hides where it really goes")
    else:
        good_findings.append("Direct address (not a hidden short link)")

    # 4. Sensitive & Urgent Keywords
    kw_count = features.get("keyword_count", 0)
    if kw_count > 0:
        red_flags.append("Contains urgent login, banking, or account words often found in scam messages")
    else:
        good_findings.append("No urgent banking or login keywords detected")

    # 5. Tricks & Exploits
    if features.get("count_at", 0) > 0:
        red_flags.append("Has an '@' symbol, which scam links use to fool your browser into visiting a different site")
    else:
        good_findings.append("No tricky '@' characters used to disguise the real site")

    if features.get("has_double_slash_path", 0) == 1:
        red_flags.append("Contains double slashes ('//') inside the address, often used for sneaky redirects")

    if features.get("subdomain_depth", 0) >= 2:
        red_flags.append("Has multiple confusing extra domain names, often used to imitate real brands (e.g. paypal.fake-site.com)")
    else:
        good_findings.append("Standard website name without confusing extra parts")

    if features.get("count_hyphens", 0) >= 3:
        red_flags.append("Contains an unusual number of hyphens, often used to create look-alike scam domains")

    if features.get("url_length", 0) > 75:
        red_flags.append("Unusually long web link (scammers make links long to hide fake names on phone screens)")

    return good_findings, red_flags


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px; padding: 4px 0;">
        <span style="font-size: 1.8rem;">🛡️</span>
        <div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC;">Link Safety Checker</div>
            <div style="font-size: 0.8rem; color: #94A3B8;">Check before you click</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Menu",
        ["🔗 Check a Link", "📁 Check a File of Links (Batch)", "💡 Safety Tips"],
        index=0
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6; background: #1E293B; padding: 14px; border-radius: 12px;">
        <b style="color: #F8FAFC;">🔒 Your Privacy First:</b><br>
        Links are checked entirely in memory. We never share, sell, or log the links you enter.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 1: 🔗 CHECK A LINK
# =============================================================================
if page == "🔗 Check a Link":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-bottom: 8px;">
            Link Safety Checker
        </h1>
        <p style="font-size: 1.08rem; color: #94A3B8; margin: 0; line-height: 1.5;">
            Check any link before you click. Paste an unfamiliar email link, SMS text, or social media message to see if it's safe.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # State Persistence for user input
    if "user_link" not in st.session_state:
        st.session_state["user_link"] = "https://en.wikipedia.org/wiki/Computer_security"
    if "check_active" not in st.session_state:
        st.session_state["check_active"] = True

    def set_sample(url_to_set: str):
        st.session_state["user_link"] = url_to_set
        st.session_state["check_active"] = True

    # 1-Click Test Pills
    st.markdown("##### 💡 Try an Example:")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.button("🟢 Try Google", on_click=set_sample, args=("https://www.google.com/search?q=weather+today",), use_container_width=True)
    with t2:
        st.button("🟢 Try Short Link (o.com)", on_click=set_sample, args=("https://www.o.com/search?q=weather+today",), use_container_width=True)
    with t3:
        st.button("🔴 Try Typosquat (gogle.com)", on_click=set_sample, args=("https://gogle.com",), use_container_width=True)
    with t4:
        st.button("🔴 Try Fake PayPal Link", on_click=set_sample, args=("http://paypal-security-update.com.verify-auth.xyz/signin?account=locked",), use_container_width=True)

    # URL Input Box
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    entered_url = st.text_input(
        "Paste the link you want to check:",
        key="user_link",
        placeholder="e.g., netflix-billing-update.com or https://google.com"
    )

    btn_col, _ = st.columns([1.2, 4])
    with btn_col:
        check_clicked = st.button("🔍 Check This Link", type="primary", use_container_width=True)

    if check_clicked:
        st.session_state["check_active"] = True

    # Run check & display results
    if st.session_state.get("check_active", False) and entered_url.strip():
        clean_url = clean_input_link(entered_url)
        url_parts = parse_link_parts(clean_url)

        # Under the hood model evaluation
        result = predict_url(clean_url, model=model, extractor=extractor)
        risk_score = result["risk_score"]
        status_label = result["status"]
        features = result["features"]

        # Plain-English translation
        good_findings, red_flags = generate_plain_english_findings(clean_url, features, risk_score)

        # Check for Brand Typosquatting / Impersonation (e.g. gogle.com -> google.com)
        is_typosquat, spoofed_brand, sim_score = check_brand_typosquatting(url_parts["host"])
        if is_typosquat:
            # Automatically spike Phishing Risk Score into the High/Danger tier (>= 75%)
            risk_score = max(risk_score, 78.5)
            status_label = "Phishing"

            # High Threat Trigger: itemized warning
            typo_warning = f"⚠️ Possible Typosquatting / Brand Impersonation of {spoofed_brand}"
            red_flags.insert(0, typo_warning)

            # Remove misleading "recognized website format" if present
            good_findings = [f for f in good_findings if "recognized website name format" not in f]

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 1. TRAFFIC-LIGHT VERDICT WITH CLEAR ADVICE
        # ---------------------------------------------------------------------
        if is_typosquat:
            verdict_box_class = "verdict-box-danger"
            badge_class = "badge-danger"
            badge_text = "🔴 High Risk / Brand Impersonation"
            verdict_text = f"Warning: This link appears to imitate {spoofed_brand}!"
            advice_text = f"⛔ <b>Advice:</b> This address looks deceivingly similar to the authentic brand <b>{spoofed_brand}</b>. Do NOT click, do not enter passwords, and do not type sensitive info."
            score_color = "#EF4444"
        elif risk_score >= 65.0 or status_label == "Phishing":
            verdict_box_class = "verdict-box-danger"
            badge_class = "badge-danger"
            badge_text = "🔴 High Risk / Dangerous Phishing"
            verdict_text = "Warning: High risk of phishing or credential theft!"
            advice_text = "⛔ <b>Advice:</b> Do NOT click this link, do not enter any passwords or codes, and do not download anything from it."
            score_color = "#EF4444"
        elif risk_score >= 35.0 or status_label == "Suspicious":
            verdict_box_class = "verdict-box-caution"
            badge_class = "badge-caution"
            badge_text = "🟡 Proceed with Caution"
            verdict_text = "Exercise caution. This link contains odd patterns like redirect shorteners or multiple hyphens."
            advice_text = "⚠️ <b>Advice:</b> If this link came from an unexpected text message or email, do not enter personal or billing info."
            score_color = "#F59E0B"
        else:
            verdict_box_class = "verdict-box-safe"
            badge_class = "badge-safe"
            badge_text = "🟢 Safe to Browse"
            verdict_text = "Looks good! This link uses standard official domain practices and secure HTTPS."
            advice_text = "💡 <b>Advice:</b> Safe to open, but always verify you are on the right official website before typing your password."
            score_color = "#10B981"

        st.markdown(f"""
        <div class="app-card {verdict_box_class}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                <span class="verdict-badge {badge_class}">{badge_text}</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: {score_color};">
                    {risk_score}% Risk Level
                </span>
            </div>
            <div class="verdict-title">{verdict_text}</div>
            <div class="advice-callout">{advice_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 2. INTERACTIVE URL INSPECTOR (3 SIMPLE PILLS)
        # ---------------------------------------------------------------------
        st.markdown("#### 🔍 Where does this link actually go?")
        st.caption("Here is where your browser will actually connect if you click:")

        col1, col2, col3 = st.columns(3)

        with col1:
            host_sub = f"⚠️ Imitating {spoofed_brand}!" if is_typosquat else "This is the actual company or server hosting the site."
            host_style = "color: #F87171;" if is_typosquat else ""
            st.markdown(f"""
            <div class="inspector-pill">
                <div class="inspector-label">[1] Real Website Name</div>
                <div class="inspector-val" style="{host_style}">{url_parts['host']}</div>
                <div class="inspector-sub">{host_sub}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            sec_title = "🔒 Secure (HTTPS)" if url_parts['is_https'] else "⚠️ Not Secure (HTTP)"
            sec_color = "#34D399" if url_parts['is_https'] else "#F87171"
            st.markdown(f"""
            <div class="inspector-pill">
                <div class="inspector-label">[2] Security Connection</div>
                <div class="inspector-val" style="color: {sec_color};">{sec_title}</div>
                <div class="inspector-sub">{'Your connection is encrypted.' if url_parts['is_https'] else 'Data sent to this link is unencrypted.'}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            path_display = url_parts['path'] if len(url_parts['path']) < 38 else f"{url_parts['path'][:35]}..."
            st.markdown(f"""
            <div class="inspector-pill">
                <div class="inspector-label">[3] Target Page</div>
                <div class="inspector-val">{path_display}</div>
                <div class="inspector-sub">The specific folder or page the link opens.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 3. WHAT WE FOUND (PLAIN-ENGLISH LISTS)
        # ---------------------------------------------------------------------
        st.markdown("#### 📋 What We Found in this Link")

        g_col, r_col = st.columns(2)

        with g_col:
            st.markdown("""
            <div class="app-card" style="border-top: 3px solid #10B981; height: 100%;">
                <h4 style="color: #34D399; margin-top: 0; margin-bottom: 14px;">✅ Things that look good</h4>
            """, unsafe_allow_html=True)

            if good_findings:
                for item in good_findings:
                    st.markdown(f'<div class="finding-item finding-safe">✓ {item}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="finding-item" style="background: rgba(255,255,255,0.03); color: #94A3B8;">No reassuring safety markers found.</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with r_col:
            st.markdown("""
            <div class="app-card" style="border-top: 3px solid #EF4444; height: 100%;">
                <h4 style="color: #F87171; margin-top: 0; margin-bottom: 14px;">⚠️ Red flags to be aware of</h4>
            """, unsafe_allow_html=True)

            if red_flags:
                for item in red_flags:
                    st.markdown(f'<div class="finding-item finding-danger">✗ {item}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="finding-item finding-safe">✓ No common scam red flags found in this address!</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 2: 📁 CHECK A FILE OF LINKS (BATCH)
# =============================================================================
elif page == "📁 Check a File of Links (Batch)":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-bottom: 8px;">
            Check Multiple Links at Once
        </h1>
        <p style="font-size: 1.05rem; color: #94A3B8; margin: 0;">
            Have a spreadsheet or document with several links? Upload it here to check them all together.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-card">
        <h4 style="margin-top: 0; margin-bottom: 8px; color: #F8FAFC;">Upload Your File</h4>
        <p style="font-size: 0.92rem; color: #94A3B8; margin-bottom: 16px;">
            You can upload a <b>.txt</b> file (with one link per line) or a <b>.csv</b> spreadsheet containing a column with links.
        </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose file:", type=["csv", "txt"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        links_list: List[str] = []
        try:
            if uploaded_file.name.endswith(".csv"):
                df_in = pd.read_csv(uploaded_file)
                found_col = None
                for col in df_in.columns:
                    if str(col).strip().lower() in ["url", "urls", "link", "links", "website", "address"]:
                        found_col = col
                        break
                if found_col is None:
                    found_col = df_in.columns[0]
                links_list = df_in[found_col].dropna().astype(str).str.strip().tolist()
            else:
                raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                links_list = [line.strip() for line in raw_text.splitlines() if line.strip() and not line.startswith("#")]
        except Exception as e:
            st.error(f"Could not read file: {e}")

        if links_list:
            st.info(f"Found **{len(links_list)}** links in `{uploaded_file.name}` ready to check.")

            if st.button("🚀 Check All Links", type="primary"):
                pbar = st.progress(0)
                status_text = st.empty()

                rows = []
                for idx, raw_l in enumerate(links_list):
                    cleaned_l = clean_input_link(raw_l)
                    parts_l = parse_link_parts(cleaned_l)
                    res = predict_url(cleaned_l, model=model, extractor=extractor)

                    score = res["risk_score"]
                    is_typo_b, spoof_b, sim_b = check_brand_typosquatting(parts_l["host"])
                    if is_typo_b:
                        score = max(score, 78.5)
                        verdict = "🔴 High Risk"
                        key_finding = f"⚠️ Possible Typosquatting / Brand Impersonation of {spoof_b}"
                    elif score >= 65.0 or res["status"] == "Phishing":
                        verdict = "🔴 High Risk"
                        good_f, red_f = generate_plain_english_findings(cleaned_l, res["features"], score)
                        key_finding = red_f[0] if red_f else "Phishing risk indicators detected"
                    elif score >= 35.0 or res["status"] == "Suspicious":
                        verdict = "🟡 Caution"
                        good_f, red_f = generate_plain_english_findings(cleaned_l, res["features"], score)
                        key_finding = red_f[0] if red_f else "Unusual link structure"
                    else:
                        verdict = "🟢 Safe"
                        key_finding = "Passed all standard safety checks"

                    rows.append({
                        "Link": raw_l,
                        "Verdict": verdict,
                        "Risk Level": f"{score}%",
                        "Key Findings": key_finding,
                        "Secure Connection": "Yes (HTTPS)" if res["features"]["is_https"] else "No (HTTP)",
                    })
                    pbar.progress((idx + 1) / len(links_list))
                    status_text.text(f"Checking link {idx + 1} of {len(links_list)}...")

                status_text.empty()
                batch_df = pd.DataFrame(rows)

                # Quick Summary Counts
                safe_count = sum(1 for r in rows if "Safe" in r["Verdict"])
                caution_count = sum(1 for r in rows if "Caution" in r["Verdict"])
                danger_count = sum(1 for r in rows if "High Risk" in r["Verdict"])

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Links", len(rows))
                with c2:
                    st.metric("🟢 Safe Links", safe_count)
                with c3:
                    st.metric("🟡 Caution Links", caution_count)
                with c4:
                    st.metric("🔴 High Risk Links", danger_count)

                st.markdown("#### 📋 Full Results List")
                st.dataframe(batch_df, use_container_width=True)

                csv_output = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_output,
                    file_name="link_safety_results.csv",
                    mime="text/csv",
                    type="primary"
                )
        else:
            st.warning("No readable links were found in this file.")


# =============================================================================
# PAGE 3: 💡 SAFETY TIPS
# =============================================================================
elif page == "💡 Safety Tips":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-bottom: 8px;">
            Simple Tips to Stay Safe Online
        </h1>
        <p style="font-size: 1.05rem; color: #94A3B8; margin: 0;">
            Most scam links rely on tricking human psychology. Here is how you can spot and avoid them easily.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tip_c1, tip_c2 = st.columns(2)

    with tip_c1:
        st.markdown("""
        <div class="app-card" style="height: 100%;">
            <h3 style="color: #38BDF8; margin-top: 0;">1. Look closely at the website name</h3>
            <p style="color: #CBD5E1; line-height: 1.6;">
                Scammers often buy look-alike names. For example, instead of <code>netflix.com</code>, they might use:
            </p>
            <ul style="color: #CBD5E1; line-height: 1.7;">
                <li><code>netflix-billing-update.com</code></li>
                <li><code>netflix.com.verify-account.xyz</code></li>
            </ul>
            <p style="color: #CBD5E1; line-height: 1.6;">
                The <b>very last word before the slash</b> is the actual website. In <code>netflix.com.verify-account.xyz/</code>, the real owner is <code>verify-account.xyz</code>, not Netflix!
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="app-card" style="height: 100%;">
            <h3 style="color: #38BDF8; margin-top: 0;">2. Beware of false urgency</h3>
            <p style="color: #CBD5E1; line-height: 1.6;">
                Almost all phishing scams try to rush you into making a mistake:
            </p>
            <ul style="color: #CBD5E1; line-height: 1.7;">
                <li><i>"Your bank account is locked! Verify within 24 hours."</i></li>
                <li><i>"Package delivery failed. Update address immediately."</i></li>
            </ul>
            <p style="color: #CBD5E1; line-height: 1.6;">
                Take a breath. Legitimate companies will rarely suspend your account without notice or give you an urgent 1-hour ultimatum.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tip_c2:
        st.markdown("""
        <div class="app-card" style="height: 100%;">
            <h3 style="color: #38BDF8; margin-top: 0;">3. Watch out for raw numbers in links</h3>
            <p style="color: #CBD5E1; line-height: 1.6;">
                If a link starts with numbers like <code>http://192.168.1.1/</code> or <code>http://45.33.32.1/</code>, <b>do not open it</b>.
            </p>
            <p style="color: #CBD5E1; line-height: 1.6;">
                Real companies use their brand name (like <code>chase.com</code> or <code>amazon.com</code>). Raw numbers mean someone is hosting a webpage directly on a server without registering a proper domain name.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="app-card" style="height: 100%;">
            <h3 style="color: #38BDF8; margin-top: 0;">4. When in doubt, go directly</h3>
            <p style="color: #CBD5E1; line-height: 1.6;">
                The single safest rule on the internet:
            </p>
            <p style="color: #34D399; font-weight: 600; line-height: 1.6; background: rgba(16, 185, 129, 0.1); padding: 12px; border-radius: 8px;">
                Never click a login link sent in an unexpected email or SMS. Open your web browser yourself, go to the official website, and sign in there.
            </p>
            <p style="color: #CBD5E1; line-height: 1.6;">
                If there really is an issue with your account, you will see a notification in your official account dashboard.
            </p>
        </div>
        """, unsafe_allow_html=True)
