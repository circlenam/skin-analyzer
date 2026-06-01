import streamlit as st
import anthropic
import base64
import json
import re
import io
import math
import datetime
import numpy as np
from PIL import Image

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

st.set_page_config(
    page_title="손등 피부 텍스처 분석기 | 재능대 AI-바이오분석연구소",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

.stApp { background-color: #0a0e1a; }
[data-testid="stAppViewContainer"] { background-color: #0a0e1a; }
[data-testid="stHeader"] { background-color: #0f1628; border-bottom: 1px solid #1e3a5f; }
[data-testid="stSidebar"] { background-color: #0f1628 !important; border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }

h1, h2, h3 {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00d4ff !important;
    letter-spacing: 1px;
}
p, div, span, label { color: #c8d8e8; font-family: 'Noto Sans KR', sans-serif; }

[data-testid="stMetric"] {
    background: #0f1628;
    border: 1px solid #1e3a5f;
    padding: 1rem;
    border-radius: 4px;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00d4ff !important;
    font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] { color: #5a7a9a !important; font-size: 0.75rem !important; }

.stButton > button {
    background: transparent !important;
    border: 1px solid #1e3a5f !important;
    color: #00d4ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 1px;
    padding: 0.6rem 1.2rem !important;
    transition: all .2s;
    width: 100%;
}
.stButton > button:hover { border-color: #00d4ff !important; background: rgba(0,212,255,0.08) !important; }

[data-testid="stFileUploadDropzone"] {
    background: #0f1628 !important;
    border: 1px dashed #1e3a5f !important;
}

.stSelectbox > div > div { background: #0f1628 !important; color: #c8d8e8 !important; }
.stTextInput > div > div > input { background: #0f1628 !important; color: #c8d8e8 !important; border: 1px solid #1e3a5f !important; }
.stNumberInput > div > div > input { background: #0f1628 !important; color: #c8d8e8 !important; border: 1px solid #1e3a5f !important; }

hr { border-color: #1e3a5f !important; }

.result-card {
    background: #0f1628;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #00d4ff;
    border-radius: 4px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.opinion-box {
    background: #0f1628;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #7fff6e;
    padding: 1.25rem;
    font-size: 14px;
    line-height: 1.85;
    color: #c8d8e8;
    border-radius: 4px;
    margin: 1rem 0;
}
.mono { font-family: 'Share Tech Mono', monospace; color: #00d4ff; }
.badge-good { color: #7fff6e; border: 1px solid #7fff6e; padding: 2px 10px; font-family: 'Share Tech Mono', monospace; font-size: 12px; border-radius: 2px; }
.badge-warn { color: #ffb300; border: 1px solid #ffb300; padding: 2px 10px; font-family: 'Share Tech Mono', monospace; font-size: 12px; border-radius: 2px; }
.badge-bad  { color: #ff4757; border: 1px solid #ff4757; padding: 2px 10px; font-family: 'Share Tech Mono', monospace; font-size: 12px; border-radius: 2px; }
.notice { background: rgba(0,212,255,0.04); border: 1px solid #1e3a5f; padding: 10px 14px; font-family: 'Share Tech Mono', monospace; font-size: 12px; color: #5a7a9a; margin: 0.5rem 0; border-radius: 2px; }
.capture-guide {
    background: rgba(127,255,110,0.06);
    border: 2px dashed #7fff6e;
    border-radius: 4px;
    padding: 0.5rem 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #7fff6e;
    text-align: center;
    margin: 0.5rem 0;
}
.icc-box {
    background: #0f1628;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #ffb300;
    padding: 1rem 1.25rem;
    border-radius: 4px;
    margin: 1rem 0;
}
.data-table {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #c8d8e8;
}
</style>
""", unsafe_allow_html=True)


# ─── 유틸리티 함수 ───────────────────────────────────

def get_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except:
        import os
        return os.environ.get("ANTHROPIC_API_KEY", "")


def image_to_base64(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    max_side = 1200
    if max(image.size) > max_side:
        ratio = max_side / max(image.size)
        image = image.resize((int(image.size[0]*ratio), int(image.size[1]*ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"


def compute_icc(scores: list) -> float:
    """ICC(2,1) 계산 — 3회 측정값"""
    if len(scores) < 2:
        return None
    n = len(scores)
    grand_mean = sum(scores) / n
    ss_between = sum((s - grand_mean)**2 for s in scores) * n
    ss_within = sum((s - grand_mean)**2 for s in scores)
    if ss_between + ss_within == 0:
        return 1.0
    icc = (ss_between - ss_within) / (ss_between + ss_within * (n - 1))
    return max(0.0, min(1.0, round(icc, 3)))


def icc_grade(icc):
    if icc is None: return "—", "badge-warn"
    if icc >= 0.90: return "매우 우수", "badge-good"
    if icc >= 0.75: return "양호", "badge-good"
    if icc >= 0.50: return "보통", "badge-warn"
    return "불량", "badge-bad"


def analyze_image(image: Image.Image, api_key: str, subject_info: dict) -> dict:
    """Claude Vision으로 피부 텍스처 분석"""
    client = anthropic.Anthropic(api_key=api_key)
    b64, media_type = image_to_base64(image)

    system = """당신은 피부과학 전문 연구자입니다.
USB 디지털 현미경(실제 배율 약 35x, 시야 약 27×13mm)으로 촬영한
손등 중앙부 피부 표면 이미지를 분석합니다.
반드시 순수 JSON만 반환하고 마크다운은 절대 포함하지 마세요."""

    prompt = f"""피험자 정보: {subject_info['age_group']}대, {subject_info['gender']}성
이 손등 피부 표면 이미지를 분석하여 아래 JSON으로만 응답하세요:

{{
  "texture_uniformity": <피부결 균일도 0~100, 숫자>,
  "wrinkle_density": "<낮음|보통|높음>",
  "wrinkle_direction": "<규칙적|불규칙>",
  "polygon_pattern": "<명확|보통|불명확>",
  "polygon_size": "<소|중|대>",
  "brightness_uniformity": "<균일|보통|불균일>",
  "vellus_hair": "<있음|없음|불명확>",
  "aging_grade": <노화 등급 0~4, 0=매우젊음 4=심한노화, 숫자>,
  "skin_condition": "<건강|보통|건조|손상>",
  "texture_score": <전반적 피부 텍스처 점수 0~100, 숫자>,
  "confidence": "<높음|보통|낮음>",
  "image_quality": "<양호|보통|불량>",
  "opinion": "<3~4문장 한국어 전문 소견. 1)피부결·텍스처 관찰 2)주름·노화 평가 3)해당 연령대 비교 4)관리 제안. 마지막줄 반드시: 종합 소견: [한 문장]>",
  "limitations": "<이 분석의 한계점 1가지. 한국어>"
}}

이미지가 피부 표면이 아니면: {{"error": "피부 표면 이미지가 아닙니다."}}"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1200,
        system=system,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r'```json|```', '', raw).strip()
    return json.loads(raw)


def get_gsheet():
    """Google Sheets 연결"""
    try:
        if not GSPREAD_AVAILABLE:
            return None
        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        sheet_id = st.secrets.get("SHEET_ID", "")
        if not sheet_id:
            return None
        spreadsheet = client.open_by_key(sheet_id)
        return spreadsheet.worksheet("data")
    except Exception as e:
        return None


def save_to_storage(record: dict):
    """Google Sheets + 세션에 동시 저장"""
    # 세션 저장 (화면 표시용)
    try:
        existing = st.session_state.get('all_data', [])
        existing.append(record)
        st.session_state['all_data'] = existing
    except:
        pass

    # Google Sheets 저장 (영구 누적)
    try:
        ws = get_gsheet()
        if ws:
            row = [
                str(record.get('측정일시', '')),
                str(record.get('피험자ID', '')),
                str(record.get('연령대', '')),
                str(record.get('성별', '')),
                str(record.get('회차', '')),
                str(record.get('텍스처점수', '')),
                str(record.get('텍스처균일도', '')),
                str(record.get('노화등급', '')),
                str(record.get('주름밀도', '')),
                str(record.get('주름방향', '')),
                str(record.get('다각형패턴', '')),
                str(record.get('밝기균일도', '')),
                str(record.get('피부상태', '')),
                str(record.get('신뢰도', '')),
                str(record.get('이미지품질', '')),
                str(record.get('ICC', '')),
            ]
            ws.append_row(row)
    except Exception as e:
        pass


def get_summary_sheet():
    """summary 시트 연결"""
    try:
        if not GSPREAD_AVAILABLE:
            return None
        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        sheet_id = st.secrets.get("SHEET_ID", "")
        if not sheet_id:
            return None
        spreadsheet = client.open_by_key(sheet_id)
        return spreadsheet.worksheet("summary")
    except Exception as e:
        return None


def save_summary(results: list, subject_info: dict, icc_val):
    """summary 시트에 피험자 1명당 1행으로 평균값 저장"""
    try:
        ws = get_summary_sheet()
        if not ws:
            return

        scores = [r.get('texture_score', 0) for r in results]
        aging = [r.get('aging_grade', 0) for r in results]
        uniformity = [r.get('texture_uniformity', 0) for r in results]

        avg_score = round(sum(scores) / len(scores), 1)
        avg_aging = round(sum(aging) / len(aging), 1)
        avg_uni = round(sum(uniformity) / len(uniformity), 1)

        icc_label, _ = icc_grade(icc_val)
        last = results[-1]

        # 3회 점수 각각 + 평균 + ICC
        row = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            str(subject_info.get('id', '')),
            str(subject_info.get('age_group', '')),
            str(subject_info.get('gender', '')),
            str(scores[0]) if len(scores) > 0 else '',
            str(scores[1]) if len(scores) > 1 else '',
            str(scores[2]) if len(scores) > 2 else '',
            str(avg_score),
            str(avg_aging),
            str(avg_uni),
            str(round(icc_val, 3)) if icc_val is not None else '',
            str(icc_label),
            str(last.get('wrinkle_density', '')),
            str(last.get('skin_condition', '')),
            str(last.get('opinion', '')[:100]) if last.get('opinion') else '',
        ]
        ws.append_row(row)
    except Exception as e:
        pass


def load_all_data():
    """Google Sheets summary 시트에서 전체 데이터 로드"""
    try:
        ws = get_summary_sheet()
        if ws:
            records = ws.get_all_records()
            return records
    except:
        pass
    return st.session_state.get('all_data', [])


def badge_html(val, good, warn):
    if val == good: cls = "badge-good"
    elif val == warn: cls = "badge-warn"
    else: cls = "badge-bad"
    return f'<span class="{cls}">{val}</span>'


def score_badge(score):
    if score >= 75: return f'<span class="badge-good">{score}</span>'
    if score >= 50: return f'<span class="badge-warn">{score}</span>'
    return f'<span class="badge-bad">{score}</span>'


# ─── 세션 초기화 ───────────────────────────────────

if 'all_data' not in st.session_state:
    st.session_state['all_data'] = []
if 'results' not in st.session_state:
    st.session_state['results'] = []
if 'current_subject' not in st.session_state:
    st.session_state['current_subject'] = {}
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False


# ─── 접속 인증 ───────────────────────────────────────

def check_access_password(pw: str) -> bool:
    correct = st.secrets.get("ACCESS_PASSWORD", "jaeneung2025")
    return pw == correct


if not st.session_state['authenticated']:
    st.markdown("""
<div style="max-width:420px;margin:6rem auto;padding:2.5rem;background:#0f1628;
     border:1px solid #1e3a5f;border-top:3px solid #00d4ff;border-radius:4px;text-align:center">
  <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
       color:#5a7a9a;letter-spacing:3px;margin-bottom:1rem">
    재능대학교 AI-바이오분석특화연구소
  </div>
  <div style="font-size:22px;font-weight:700;color:#fff;margin-bottom:0.25rem">
    🔬 손등 피부 텍스처 분석기
  </div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
       color:#5a7a9a;margin-bottom:2rem">
    Research Edition · 승인된 연구 참여자 전용
  </div>
  <div style="font-size:13px;color:#c8d8e8;margin-bottom:1.5rem;line-height:1.7">
    본 프로그램은 연구 목적으로 운영됩니다.<br>
    참여 코드는 연구 담당자에게 문의하세요.
  </div>
</div>
""", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1,2,1])
    with col_c:
        pw_input = st.text_input(
            "참여 코드 입력",
            type="password",
            placeholder="연구 담당자에게 문의",
            label_visibility="collapsed"
        )
        if st.button("▶ 입장", use_container_width=True):
            if check_access_password(pw_input):
                st.session_state['authenticated'] = True
                st.rerun()
            else:
                st.error("참여 코드가 올바르지 않습니다. 연구 담당자에게 문의하세요.")
        st.markdown("""
<div style="text-align:center;margin-top:1.5rem;font-family:'Share Tech Mono',monospace;
     font-size:10px;color:#1e3a5f">
  본 연구는 재능대학교 바이오테크과 오픈랩 행사의 일환으로 진행됩니다<br>
  © 2025 Jay H. Nam · AI-바이오분석특화연구소
</div>
""", unsafe_allow_html=True)
    st.stop()


# ─── 헤더 ───────────────────────────────────────────

st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem">
  <span style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#5a7a9a;letter-spacing:3px">
    재능대학교 AI-바이오분석특화연구소
  </span>
</div>
""", unsafe_allow_html=True)
st.title("🔬 손등 피부 텍스처 분석기")
st.markdown('<div class="mono" style="font-size:12px;color:#5a7a9a">Dorsal Hand Skin Texture Analyzer · USB Microscopy + LLM Vision AI · Research Edition</div>', unsafe_allow_html=True)

# ─── 사이드바 ───────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ 설정")

    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    else:
        st.success("API 키 연결됨")

    st.markdown("---")
    mode = st.radio("모드 선택", ["📋 측정 모드", "🔒 연구팀 모드"], index=0)

    if mode == "🔒 연구팀 모드":
        pw = st.text_input("비밀번호", type="password")
        research_mode = (pw == st.secrets.get("RESEARCH_PASSWORD", "jaeneung2025"))
    else:
        research_mode = False

    st.markdown("---")
    st.markdown("""
<div style="font-size:11px;color:#5a7a9a;font-family:'Share Tech Mono',monospace;line-height:2">
측정 프로토콜:<br>
1 → 손등 중앙부 표시<br>
2 → USB 현미경 접촉<br>
3 → 촬영 → 갤러리 저장<br>
4 → 현미경 완전히 뗌<br>
5 → 30초 후 재촬영<br>
6 → 총 3회 반복<br>
7 → 이미지 3장 업로드
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    total = len(load_all_data())
    subjects = len(set(r.get('subject_id','') for r in load_all_data()))
    st.markdown(f'<div class="mono" style="font-size:11px;color:#5a7a9a">누적 측정: {total}건<br>피험자 수: {subjects}명</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div style="font-size:11px;color:#5a7a9a;font-family:'Share Tech Mono',monospace;line-height:1.9">
<span style="color:#00d4ff;letter-spacing:2px">// DEVELOPER</span><br><br>
<span style="color:#ffffff;font-weight:700;font-size:12px">남정훈 교수</span><br>
<span style="color:#c8d8e8;font-size:10px">Jay H. Nam, Ph.D.</span><br><br>
재능대학교 바이오테크과 학과장<br>
AI-바이오분석특화연구소 (산학협력단 부설)<br><br>
<span style="color:#7fff6e">연구분야</span><br>
생체유체역학 · 혈유변학<br>
랩온어칩 기반 시료전처리<br><br>
<a href="https://github.com/circlenam" target="_blank" style="color:#00d4ff;text-decoration:none">⬡ GitHub · circlenam</a><br>
<a href="https://linkedin.com/in/circlenam" target="_blank" style="color:#00d4ff;text-decoration:none">⬡ LinkedIn · circlenam</a><br>
<a href="mailto:namjh@jeiu.ac.kr" style="color:#00d4ff;text-decoration:none">⬡ namjh@jeiu.ac.kr</a><br><br>
<a href="https://circlenam.github.io/biogame/" target="_blank" style="color:#7fff6e;text-decoration:none">⬡ 바이오분석오락실</a><br>
<a href="https://bioanalysis.re.kr" target="_blank" style="color:#7fff6e;text-decoration:none">⬡ bioanalysis.re.kr</a><br><br>
<span style="color:#1e3a5f">© 2025 Jay H. Nam · 재능대학교</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# 측정 모드
# ═══════════════════════════════════════════════════

if not research_mode:

    st.markdown("---")
    st.markdown("### 01 · 피험자 정보 입력")

    col1, col2, col3 = st.columns(3)
    with col1:
        subject_id = st.text_input("피험자 ID", placeholder="P001", max_chars=10)
    with col2:
        age_group = st.selectbox("연령대", ["20", "30", "40", "50", "60"])
    with col3:
        gender = st.selectbox("성별", ["여", "남"])

    subject_info = {"id": subject_id, "age_group": age_group, "gender": gender}

    st.markdown("---")
    st.markdown("### 02 · 이미지 업로드 (3회 촬영)")
    st.markdown('<div class="notice">▶ 갤러리에서 동일 부위 3회 촬영 이미지를 순서대로 업로드하세요</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="mono" style="font-size:12px;margin-bottom:8px">1회차</div>', unsafe_allow_html=True)
        img1 = st.file_uploader("1회차", type=["jpg","jpeg","png"], key="img1", label_visibility="collapsed")
        if img1: st.image(Image.open(img1), use_container_width=True)

    with col_b:
        st.markdown('<div class="mono" style="font-size:12px;margin-bottom:8px">2회차</div>', unsafe_allow_html=True)
        img2 = st.file_uploader("2회차", type=["jpg","jpeg","png"], key="img2", label_visibility="collapsed")
        if img2: st.image(Image.open(img2), use_container_width=True)

    with col_c:
        st.markdown('<div class="mono" style="font-size:12px;margin-bottom:8px">3회차</div>', unsafe_allow_html=True)
        img3 = st.file_uploader("3회차", type=["jpg","jpeg","png"], key="img3", label_visibility="collapsed")
        if img3: st.image(Image.open(img3), use_container_width=True)

    uploaded = [f for f in [img1, img2, img3] if f is not None]
    n_uploaded = len(uploaded)

    st.markdown(f'<div class="notice">업로드된 이미지: {n_uploaded}/3장</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🔬 분석 시작", disabled=(n_uploaded == 0 or not api_key or not subject_id)):
        if not subject_id:
            st.error("피험자 ID를 입력하세요.")
        else:
            results = []
            progress = st.progress(0)
            status = st.empty()

            for i, f in enumerate(uploaded):
                status.markdown(f'<div class="notice">▶ {i+1}회차 분석 중...</div>', unsafe_allow_html=True)
                img = Image.open(f)
                try:
                    result = analyze_image(img, api_key, subject_info)
                    if "error" not in result:
                        result['round'] = i + 1
                        results.append(result)
                except Exception as e:
                    st.error(f"{i+1}회차 오류: {str(e)}")
                progress.progress((i+1) / n_uploaded)

            status.empty()
            progress.empty()

            if results:
                st.session_state['results'] = results
                st.session_state['current_subject'] = subject_info

                # 누적 저장
                scores = [r.get('texture_score', 0) for r in results]
                aging = [r.get('aging_grade', 0) for r in results]
                icc_val = compute_icc(scores) if len(scores) == 3 else None

                # data 시트: 3회 각각 원본 저장
                for i, r in enumerate(results):
                    record = {
                        "측정일시": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "피험자ID": subject_id,
                        "연령대": age_group,
                        "성별": gender,
                        "회차": i + 1,
                        "텍스처점수": r.get('texture_score'),
                        "텍스처균일도": r.get('texture_uniformity'),
                        "노화등급": r.get('aging_grade'),
                        "주름밀도": r.get('wrinkle_density'),
                        "주름방향": r.get('wrinkle_direction'),
                        "다각형패턴": r.get('polygon_pattern'),
                        "밝기균일도": r.get('brightness_uniformity'),
                        "피부상태": r.get('skin_condition'),
                        "신뢰도": r.get('confidence'),
                        "이미지품질": r.get('image_quality'),
                        "ICC": icc_val if i == 2 else None,
                    }
                    save_to_storage(record)

                # summary 시트: 피험자 1명당 1행 (평균값 + ICC)
                save_summary(results, subject_info, icc_val)

                st.success("✅ 분석 완료")
                st.rerun()

    # ─── 결과 표시 ─────────────────────────────────

    if st.session_state.get('results'):
        results = st.session_state['results']
        subj = st.session_state.get('current_subject', {})
        scores = [r.get('texture_score', 0) for r in results]
        aging_scores = [r.get('aging_grade', 0) for r in results]
        avg_score = round(sum(scores) / len(scores))
        avg_aging = round(sum(aging_scores) / len(aging_scores), 1)
        icc_val = compute_icc(scores) if len(scores) == 3 else None
        icc_label, icc_cls = icc_grade(icc_val)

        st.markdown("---")

        # 캡처 가이드
        st.markdown('<div class="capture-guide">📸 아래 결과 화면을 캡처하여 피험자에게 전송하세요</div>', unsafe_allow_html=True)

        # ── 결과 카드 (캡처용) ──
        st.markdown(f"""
<div class="result-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">
    <div>
      <div class="mono" style="font-size:11px;color:#5a7a9a;letter-spacing:2px">// SKIN TEXTURE ANALYSIS REPORT</div>
      <div style="font-size:18px;font-weight:700;color:#fff;margin-top:4px">손등 피부 텍스처 분석 결과</div>
    </div>
    <div style="text-align:right;font-family:'Share Tech Mono',monospace;font-size:11px;color:#5a7a9a">
      재능대학교<br>AI-바이오분석특화연구소<br>{datetime.datetime.now().strftime("%Y-%m-%d")}
    </div>
  </div>
  <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem">
    <div style="background:#0a0e1a;padding:8px 16px;border:1px solid #1e3a5f;border-radius:2px;font-family:'Share Tech Mono',monospace;font-size:12px">
      피험자 {subj.get('id','—')} · {subj.get('age_group','—')}대 · {subj.get('gender','—')}성
    </div>
    <div style="background:#0a0e1a;padding:8px 16px;border:1px solid #1e3a5f;border-radius:2px;font-family:'Share Tech Mono',monospace;font-size:12px">
      측정 {len(results)}회 완료
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # 핵심 수치
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("텍스처 점수 (평균)", f"{avg_score} / 100")
        c2.metric("노화 등급 (평균)", f"{avg_aging} / 4")
        c3.metric("측정 신뢰도 ICC", f"{icc_val:.2f}" if icc_val else "—")
        c4.metric("이미지 수", f"{len(results)} 회")

        # 회차별 상세
        st.markdown("#### 회차별 분석 결과")
        cols = st.columns(len(results))
        for i, (col, r) in enumerate(zip(cols, results)):
            with col:
                st.markdown(f'<div class="mono" style="font-size:11px;color:#5a7a9a;margin-bottom:8px">{i+1}회차</div>', unsafe_allow_html=True)
                rows = [
                    ("텍스처점수", f"{r.get('texture_score')}/100"),
                    ("노화등급", f"{r.get('aging_grade')}/4"),
                    ("주름밀도", r.get('wrinkle_density','—')),
                    ("피부결", r.get('polygon_pattern','—')),
                    ("피부상태", r.get('skin_condition','—')),
                    ("이미지품질", r.get('image_quality','—')),
                ]
                for k, v in rows:
                    st.markdown(f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e3a5f;font-size:12px"><span style="color:#5a7a9a">{k}</span><span style="font-family:Share Tech Mono,monospace;color:#c8d8e8">{v}</span></div>', unsafe_allow_html=True)

        # ICC 해석
        if icc_val is not None:
            st.markdown(f"""
<div class="icc-box">
  <div class="mono" style="font-size:11px;color:#ffb300;margin-bottom:8px">// 반복 측정 신뢰도 (ICC)</div>
  <div style="font-size:24px;font-weight:700;color:#ffb300;font-family:'Share Tech Mono',monospace">{icc_val:.3f}</div>
  <div style="margin-top:8px">
    <span class="{icc_cls}">{icc_label}</span>
    <span style="font-size:12px;color:#5a7a9a;margin-left:12px">
      {'ICC ≥ 0.90: 매우 우수' if icc_val >= 0.90 else 'ICC ≥ 0.75: 신뢰도 충분' if icc_val >= 0.75 else 'ICC < 0.75: 재측정 권장'}
    </span>
  </div>
  <div style="font-size:11px;color:#5a7a9a;margin-top:8px;font-family:'Share Tech Mono',monospace">
    텍스처 점수 3회: {scores[0]} · {scores[1]} · {scores[2]}
  </div>
</div>
""", unsafe_allow_html=True)

        # AI 소견 (마지막 회차 기준)
        last = results[-1]
        st.markdown("#### AI 판독 소견")
        conf = last.get('confidence','보통')
        conf_colors = {"높음": "#7fff6e", "보통": "#ffb300", "낮음": "#ff4757"}
        conf_color = conf_colors.get(conf, "#ffb300")
        st.markdown(f'<div class="notice">분석 신뢰도: <span style="color:{conf_color};font-family:Share Tech Mono,monospace">{conf}</span> · 이미지 품질: <span style="color:{conf_color};font-family:Share Tech Mono,monospace">{last.get("image_quality","—")}</span></div>', unsafe_allow_html=True)
        opinion = last.get('opinion','소견 없음').replace('\n','<br>')
        st.markdown(f'<div class="opinion-box">{opinion}</div>', unsafe_allow_html=True)

        # 한계점
        lim = last.get('limitations','')
        if lim:
            st.markdown(f'<div class="notice">⚠ 분석 한계: {lim}</div>', unsafe_allow_html=True)

        # 연구소 서명
        st.markdown(f"""
<div style="text-align:center;padding:1rem;border-top:1px solid #1e3a5f;margin-top:1rem">
  <div class="mono" style="font-size:11px;color:#5a7a9a">
    재능대학교 바이오테크과 · AI-바이오분석특화연구소<br>
    본 결과는 연구 참고용이며 임상 진단을 대체하지 않습니다
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="capture-guide">📸 위 결과를 화면 캡처 후 카카오톡으로 전송하세요</div>', unsafe_allow_html=True)

        st.markdown("---")

        # 초기화
        if st.button("↺ 새 피험자 측정"):
            st.session_state['results'] = []
            st.session_state['current_subject'] = {}
            st.rerun()


# ═══════════════════════════════════════════════════
# 연구팀 모드
# ═══════════════════════════════════════════════════

else:
    st.markdown("---")
    st.markdown("### 🔒 연구팀 대시보드")

    all_data = load_all_data()

    if not all_data:
        st.info("아직 측정 데이터가 없습니다.")
    else:
        import pandas as pd

        df = pd.DataFrame(all_data)

        # 컬럼명 안전하게 처리
        score_col = '평균텍스처점수' if '평균텍스처점수' in df.columns else ('텍스처점수' if '텍스처점수' in df.columns else None)
        aging_col = '평균노화등급' if '평균노화등급' in df.columns else ('노화등급' if '노화등급' in df.columns else None)
        id_col = '피험자ID' if '피험자ID' in df.columns else None
        age_col = '연령대' if '연령대' in df.columns else None
        icc_col = 'ICC' if 'ICC' in df.columns else None

        # 요약 통계
        st.markdown("#### 데이터 수집 현황")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 피험자 수", len(df))
        c2.metric("피험자 ID 수", df[id_col].nunique() if id_col else '—')

        if score_col:
            scores_num = pd.to_numeric(df[score_col], errors='coerce')
            c3.metric("평균 텍스처 점수", f"{scores_num.mean():.1f}" if not scores_num.isna().all() else '—')
        else:
            c3.metric("평균 텍스처 점수", "—")

        if aging_col:
            aging_num = pd.to_numeric(df[aging_col], errors='coerce')
            c4.metric("평균 노화 등급", f"{aging_num.mean():.2f}" if not aging_num.isna().all() else '—')
        else:
            c4.metric("평균 노화 등급", "—")

        st.markdown("---")

        # 연령별 통계
        if score_col and age_col:
            st.markdown("#### 연령대별 평균 텍스처 점수")
            df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
            age_stats = df.groupby(age_col)[score_col].agg(['mean','std','count']).round(2)
            age_stats.columns = ['평균점수', '표준편차', '피험자수']
            st.dataframe(age_stats, use_container_width=True)

        # ICC 현황
        if icc_col:
            df[icc_col] = pd.to_numeric(df[icc_col], errors='coerce')
            icc_data = df[df[icc_col].notna()]
            if not icc_data.empty:
                st.markdown("#### ICC 신뢰도 현황")
                c1, c2, c3 = st.columns(3)
                c1.metric("평균 ICC", f"{icc_data[icc_col].mean():.3f}")
                c2.metric("ICC ≥ 0.75 비율", f"{(icc_data[icc_col] >= 0.75).mean()*100:.0f}%")
                c3.metric("ICC ≥ 0.90 비율", f"{(icc_data[icc_col] >= 0.90).mean()*100:.0f}%")

        st.markdown("---")

        # 3회 점수 비교 (summary 시트)
        if all(c in df.columns for c in ['점수1회','점수2회','점수3회']):
            st.markdown("#### 회차별 점수 분포")
            for col in ['점수1회','점수2회','점수3회']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            round_stats = pd.DataFrame({
                '1회차': [df['점수1회'].mean().round(1), df['점수1회'].std().round(1)],
                '2회차': [df['점수2회'].mean().round(1), df['점수2회'].std().round(1)],
                '3회차': [df['점수3회'].mean().round(1), df['점수3회'].std().round(1)],
            }, index=['평균', '표준편차'])
            st.dataframe(round_stats, use_container_width=True)

        st.markdown("---")

        # 전체 데이터 테이블
        st.markdown("#### 전체 데이터 (summary)")
        st.dataframe(df, use_container_width=True)

        st.markdown("---")

        # CSV 다운로드
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        today = datetime.datetime.now().strftime("%Y%m%d")
        st.download_button(
            label="▼ CSV 다운로드 (논문용)",
            data=csv.encode('utf-8-sig'),
            file_name=f"skin_texture_summary_{today}.csv",
            mime="text/csv"
        )

        st.markdown(f'<div class="notice">※ summary 시트 기준 · 피험자 1명당 1행 · SPSS/R에서 바로 사용 가능</div>', unsafe_allow_html=True)

        # 데이터 초기화
        st.markdown("---")
        if st.button("⚠️ 세션 데이터 초기화"):
            st.session_state['all_data'] = []
            st.rerun()
