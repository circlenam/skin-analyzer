import streamlit as st
import anthropic
import base64
import json
import re
from PIL import Image
import io
import numpy as np

st.set_page_config(
    page_title="피부 각질세포 분석기 | 바이오분석오락실",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Noto+Sans+KR:wght@400;500;700&display=swap');

/* 전체 배경 */
.stApp { background-color: #0a0e1a; }
[data-testid="stAppViewContainer"] { background-color: #0a0e1a; }
[data-testid="stHeader"] { background-color: #0f1628; border-bottom: 1px solid #1e3a5f; }

/* 사이드바 */
[data-testid="stSidebar"] {
    background-color: #0f1628 !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }

/* 텍스트 */
h1, h2, h3 { font-family: 'Share Tech Mono', monospace !important; color: #00d4ff !important; }
p, div, span, label { color: #c8d8e8; font-family: 'Noto Sans KR', sans-serif; }

/* 메트릭 */
[data-testid="stMetric"] {
    background: #0f1628;
    border: 1px solid #1e3a5f;
    padding: 1rem;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00d4ff !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] { color: #5a7a9a !important; font-size: 0.75rem !important; }

/* 버튼 */
.stButton > button {
    background: transparent !important;
    border: 1px solid #1e3a5f !important;
    color: #00d4ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 1px;
    transition: all .2s;
}
.stButton > button:hover {
    border-color: #00d4ff !important;
    background: rgba(0,212,255,0.08) !important;
}

/* 업로드 */
[data-testid="stFileUploadDropzone"] {
    background: #0f1628 !important;
    border: 1px dashed #1e3a5f !important;
    color: #5a7a9a !important;
}

/* 구분선 */
hr { border-color: #1e3a5f; }

/* info/success/warning 박스 */
[data-testid="stAlert"] { background: #0f1628 !important; border: 1px solid #1e3a5f !important; }

/* 스캔라인 오버레이 */
.scanline {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px,
        rgba(0,212,255,0.012) 2px, rgba(0,212,255,0.012) 4px);
    pointer-events: none; z-index: 9999;
}
.mono { font-family: 'Share Tech Mono', monospace; }
.tag {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    padding: 2px 8px;
    border: 1px solid #1e3a5f;
    color: #5a7a9a;
    margin-right: 4px;
    margin-bottom: 4px;
}
.badge-good { color: #7fff6e; border: 1px solid #7fff6e; padding: 2px 10px;
              font-family: 'Share Tech Mono', monospace; font-size: 12px; }
.badge-warn { color: #ffb300; border: 1px solid #ffb300; padding: 2px 10px;
              font-family: 'Share Tech Mono', monospace; font-size: 12px; }
.badge-bad  { color: #ff4757; border: 1px solid #ff4757; padding: 2px 10px;
              font-family: 'Share Tech Mono', monospace; font-size: 12px; }
.opinion-box {
    background: #0f1628;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #00d4ff;
    padding: 1.25rem;
    font-size: 14px;
    line-height: 1.85;
    color: #c8d8e8;
    margin-bottom: 1rem;
}
.notice-box {
    background: rgba(0,212,255,0.04);
    border: 1px solid #1e3a5f;
    padding: 10px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #5a7a9a;
    letter-spacing: .5px;
    margin-bottom: 1rem;
}
.reliability-box {
    background: #0f1628;
    border: 1px solid #1e3a5f;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
}
</style>
<div class="scanline"></div>
""", unsafe_allow_html=True)


def get_api_key():
    """Streamlit secrets 또는 환경변수에서 API 키 로드"""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        import os
        return os.environ.get("ANTHROPIC_API_KEY", "")


def image_to_base64(image: Image.Image) -> tuple[str, str]:
    """PIL 이미지를 base64로 변환"""
    buf = io.BytesIO()
    fmt = "JPEG" if image.mode == "RGB" else "PNG"
    image.save(buf, format=fmt)
    data = base64.standard_b64encode(buf.getvalue()).decode()
    media_type = "image/jpeg" if fmt == "JPEG" else "image/png"
    return data, media_type


def pixel_coverage(image: Image.Image) -> float:
    """실제 픽셀 기반 피복률 계산 (밝은 영역 = 세포)"""
    gray = np.array(image.convert("L"), dtype=np.float32)
    thresh = 160
    bright = np.sum(gray > thresh)
    return round(float(bright) / gray.size * 100, 1)


def analyze_with_claude(image: Image.Image, api_key: str) -> dict:
    """Claude Vision으로 각질세포 이미지 분석"""
    client = anthropic.Anthropic(api_key=api_key)

    # 이미지 리사이즈 (API 비용 절감, 1200px 이하)
    max_side = 1200
    if max(image.size) > max_side:
        ratio = max_side / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    if image.mode != "RGB":
        image = image.convert("RGB")

    b64, media_type = image_to_base64(image)
    pixel_cov = pixel_coverage(image)

    system_prompt = """당신은 피부과학 전문 연구자이자 현미경 이미지 분석 전문가입니다.
테이프스트리핑으로 채취한 각질세포(corneocyte) 현미경 이미지를 분석합니다.
반드시 순수 JSON만 반환하고 마크다운 코드블록이나 다른 텍스트는 절대 포함하지 마세요."""

    user_prompt = f"""이 현미경 이미지를 분석해서 아래 JSON 형식으로만 응답하세요.

픽셀 분석으로 이미 측정된 값: 피복률 = {pixel_cov}%

나머지 파라미터를 이미지에서 추정해서 JSON으로 반환:

{{
  "area_um2": <평균 각질세포 면적 µm² 추정. 각질세포는 보통 200~500µm². 이미지 시야 크기와 세포 비율로 추정. 숫자>,
  "circularity": <원형도 0~1. 정상 다각형=0.6~0.8, 건조=0.8~0.9, 손상=0.9~1.0. 숫자>,
  "coverage_pct": {pixel_cov},
  "cv_pct": <세포 크기 변동계수%. 균일=10~20, 보통=20~35, 불균일=35+. 숫자>,
  "cell_count": <시야 내 추정 세포 수. 숫자>,
  "fragment_pct": <파편화된 세포 비율%. 숫자>,
  "barrier": "<양호|주의|저하>",
  "layer": "<두꺼움|정상|얇음>",
  "uniformity": "<균일|보통|불균일>",
  "moisture": "<충분|보통|부족>",
  "score": <종합 피부 점수 0~100. 숫자>,
  "confidence": "<높음|보통|낮음>",
  "image_quality": "<양호|보통|불량>",
  "opinion": "<4~6문장 한국어 전문 소견. 1)이미지 품질 및 세포 관찰 결과 2)각질세포 형태 해석(면적·원형도 의미) 3)피부 장벽 상태 평가 4)추정 원인 5)관리 방향. 마지막 줄 반드시: 종합 소견: [한 문장]>",
  "limitations": "<이 분석의 주요 한계점 1~2가지. 한국어>"
}}

이미지가 현미경 각질세포 이미지가 아니면:
{{"error": "각질세포 현미경 이미지가 아닌 것 같습니다. 테이프스트리핑 후 스마트폰 현미경으로 촬영한 이미지를 업로드해주세요."}}"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": user_prompt}
            ]
        }]
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r'```json|```', '', raw).strip()
    return json.loads(raw)


def badge(val, good, warn):
    if val == good:
        return f'<span class="badge-good">{val}</span>'
    elif val == warn:
        return f'<span class="badge-warn">{val}</span>'
    else:
        return f'<span class="badge-bad">{val}</span>'


def confidence_note(conf):
    notes = {
        "높음": "이미지 품질이 양호하고 세포 경계가 명확하여 분석 신뢰도가 높습니다.",
        "보통": "이미지에서 세포 경계가 일부 불명확합니다. 더 선명한 이미지로 재촬영하면 정확도가 높아집니다.",
        "낮음": "이미지 해상도나 초점이 분석에 충분하지 않습니다. 결과를 참고 수준으로만 활용하세요."
    }
    return notes.get(conf, "")


# ─── 메인 UI ──────────────────────────────────────────

# 헤더
col_back, col_title = st.columns([1, 6])
with col_back:
    st.markdown('<a href="https://circlenam.github.io/biogame/" style="font-family:Share Tech Mono,monospace;font-size:12px;color:#00d4ff;text-decoration:none;border:1px solid #1e3a5f;padding:4px 10px;letter-spacing:1px">← ARCADE</a>', unsafe_allow_html=True)
with col_title:
    st.markdown('<span style="font-family:Share Tech Mono,monospace;font-size:12px;color:#5a7a9a;letter-spacing:2px">STAGE 06 · SKIN CORNEOCYTE ANALYZER</span>', unsafe_allow_html=True)

st.markdown("---")

st.markdown('<div class="mono" style="color:#5a7a9a;font-size:11px;letter-spacing:3px">// STAGE 06 · CORNEOCYTE ANALYSIS</div>', unsafe_allow_html=True)
st.title("🔬 피부 각질세포 분석기")
st.markdown('<div class="mono" style="color:#5a7a9a;font-size:12px">Tape Stripping + Smartphone Microscopy → AI Skin Barrier Assessment</div>', unsafe_allow_html=True)

st.markdown('<div class="notice-box">▶ Claude Vision AI (claude-opus-4-5)가 이미지를 직접 분석합니다 · 피복률은 픽셀 실측 · 나머지는 AI 시각 추정</div>', unsafe_allow_html=True)

# ─── 사이드바 ───
with st.sidebar:
    st.markdown("### ⚙️ 설정")

    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="https://console.anthropic.com 에서 발급"
        )
    else:
        st.success("API 키 연결됨")

    st.markdown("---")
    st.markdown("### 📋 분석 단계")
    st.markdown("""
<div class="mono" style="font-size:11px;line-height:2">
1 → 테이프스트리핑 채취<br>
2 → 스마트폰 현미경 촬영<br>
3 → 이미지 업로드<br>
4 → AI 자동 판독<br>
5 → 리포트 확인
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚠️ 판독 한계")
    st.markdown("""
<div style="font-size:12px;line-height:1.7;color:#5a7a9a">
<b style="color:#ffb300">면적(µm²)</b><br>
캘리브레이션 없이 절대값 부정확.<br>
동일 배율에서 상대 비교 권장.<br><br>
<b style="color:#ffb300">원형도·CV%</b><br>
AI 시각 추정. ±15% 오차 가능.<br><br>
<b style="color:#7fff6e">피복률(%)</b><br>
픽셀 직접 측정. 가장 신뢰 가능.<br><br>
<b style="color:#7fff6e">정성 판독</b><br>
장벽 상태, 소견은 참고 수준.<br>
임상 진단 대체 불가.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div class="mono" style="font-size:10px;color:#1e3a5f">
바이오분석오락실 STAGE 06<br>
재능대학교 AI-바이오분석특화연구소<br>
© 2025 Jay H. Nam
</div>
""", unsafe_allow_html=True)

# ─── 업로드 ───
st.markdown("### 📂 이미지 업로드")
uploaded = st.file_uploader(
    "테이프스트리핑 현미경 이미지 (JPG / PNG / TIFF)",
    type=["jpg", "jpeg", "png", "tiff", "tif"],
    help="스마트폰 접사렌즈 또는 실험실 광학현미경 이미지를 업로드하세요"
)

if uploaded:
    image = Image.open(uploaded)

    col_img, col_info = st.columns([3, 2])
    with col_img:
        st.markdown("#### 업로드 이미지")
        st.image(image, use_container_width=True)
        st.markdown(f'<div class="mono" style="font-size:11px;color:#5a7a9a">크기: {image.size[0]}×{image.size[1]}px · 모드: {image.mode}</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown("#### 픽셀 실측값")
        pix_cov = pixel_coverage(image)
        st.metric("피복률 (픽셀 직접 측정)", f"{pix_cov}%", help="임계값(160) 이상 밝은 픽셀 비율. 가장 신뢰 가능한 수치.")

        cov_interp = "정상 범위" if 35 <= pix_cov <= 65 else ("각질 축적 의심" if pix_cov > 65 else "각질 부족")
        cov_badge = "badge-good" if 35 <= pix_cov <= 65 else "badge-warn"
        st.markdown(f'<span class="{cov_badge}">{cov_interp}</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
<div style="font-size:12px;color:#5a7a9a;line-height:1.8">
<b style="color:#c8d8e8">피복률 기준</b><br>
35~65% → 정상<br>
&gt;65% → 각질 축적<br>
&lt;35% → 각질 부족
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    if not api_key:
        st.warning("사이드바에 Anthropic API Key를 입력하면 AI 분석이 시작됩니다.")
    else:
        with st.spinner("Claude Vision AI 분석 중..."):
            try:
                result = analyze_with_claude(image, api_key)

                if "error" in result:
                    st.error(result["error"])
                else:
                    # ─── 수치 메트릭 ───
                    st.markdown("### 📊 형태학 파라미터")
                    st.markdown(f'<div class="notice-box">신뢰도: <b style="color:#00d4ff">{result.get("confidence","—")}</b> · 이미지 품질: <b style="color:#00d4ff">{result.get("image_quality","—")}</b> · 피복률은 픽셀 실측, 나머지는 AI 시각 추정</div>', unsafe_allow_html=True)

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("평균 면적 (µm²)", Math_round(result.get("area_um2", 0)),
                              help="AI 추정. 캘리브레이션 없이 절대값 부정확. 상대 비교 권장.")
                    c2.metric("원형도", f"{float(result.get('circularity',0)):.2f}",
                              help="0=선형, 1=완전원형. 정상 각질세포는 다각형(0.6~0.8).")
                    c3.metric("피복률 (%)", f"{result.get('coverage_pct',0):.1f}",
                              help="픽셀 직접 측정값. 가장 신뢰 가능.")
                    c4.metric("크기 CV (%)", f"{float(result.get('cv_pct',0)):.1f}",
                              help="세포 크기 균일성. 낮을수록 균일.")
                    c5.metric("추정 세포 수", Math_round(result.get("cell_count", 0)),
                              help="AI가 시야 내 세포를 시각적으로 계수한 추정값.")

                    st.markdown("---")

                    # ─── 상태 지표 + 파라미터 가이드 ───
                    col_status, col_guide = st.columns(2)

                    with col_status:
                        st.markdown("#### 피부 상태 지표")
                        rows = [
                            ("피부 장벽", result.get("barrier","—"), "양호", "주의"),
                            ("각질층 상태", result.get("layer","—"), "정상", "두꺼움"),
                            ("세포 균일성", result.get("uniformity","—"), "균일", "보통"),
                            ("수분 수준", result.get("moisture","—"), "충분", "보통"),
                        ]
                        for label, val, good, warn in rows:
                            b = badge(val, good, warn)
                            st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #1e3a5f;font-size:13px"><span>{label}</span>{b}</div>', unsafe_allow_html=True)

                        sc = int(result.get("score", 0))
                        sc_class = "badge-good" if sc >= 75 else ("badge-warn" if sc >= 50 else "badge-bad")
                        st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;font-size:13px"><span>종합 점수</span><span class="{sc_class}" style="font-family:Share Tech Mono,monospace;font-size:13px;padding:2px 10px;border:1px solid">{sc} / 100</span></div>', unsafe_allow_html=True)

                    with col_guide:
                        st.markdown("#### 파라미터 해석 기준")
                        st.markdown("""
<div style="font-size:12px;line-height:2;color:#c8d8e8">
<span class="badge-good">양호</span> 면적 &gt;350 µm² · 수분 충분<br>
<span class="badge-good">양호</span> 원형도 &lt;0.80 · 다각형 형태<br>
<span class="badge-good">양호</span> CV% &lt;25 · 크기 균일<br>
<span class="badge-warn">주의</span> 원형도 0.80~0.90 · 수축 경향<br>
<span class="badge-warn">주의</span> 피복률 &gt;65% · 각질 축적<br>
<span class="badge-bad">저하</span> 원형도 &gt;0.90 · 위축·손상<br>
<span class="badge-bad">저하</span> CV% &gt;35 · 크기 불균일<br>
<span class="badge-bad">저하</span> 파편 &gt;30% · 장벽 손상
</div>
""", unsafe_allow_html=True)

                    st.markdown("---")

                    # ─── AI 소견 ───
                    st.markdown("#### 🤖 AI 판독 소견")
                    conf = result.get("confidence", "보통")
                    conf_note = confidence_note(conf)
                    if conf_note:
                        st.markdown(f'<div class="notice-box">{conf_note}</div>', unsafe_allow_html=True)

                    opinion_html = result.get("opinion","소견 없음").replace("\n","<br>")
                    st.markdown(f'<div class="opinion-box">{opinion_html}</div>', unsafe_allow_html=True)

                    # ─── 한계점 명시 ───
                    limitations = result.get("limitations","")
                    if limitations:
                        st.markdown(f'<div class="reliability-box"><span class="mono" style="color:#ffb300;font-size:11px">⚠ 분석 한계</span><br><span style="font-size:12px;color:#5a7a9a">{limitations}</span></div>', unsafe_allow_html=True)

                    st.markdown("---")

                    # ─── 신뢰도 안내 ───
                    st.markdown("#### 📌 이 결과를 어떻게 활용하나요?")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("""
<div style="font-size:13px;line-height:1.8">
<span class="badge-good">권장 활용</span><br>
• 동일 배율 전후 비교 (보습제 효과 등)<br>
• 교육용 시연·오픈랩 데모<br>
• 연구 예비실험 스크리닝<br>
• 정성적 피부 상태 판단
</div>
""", unsafe_allow_html=True)
                    with col_b:
                        st.markdown("""
<div style="font-size:13px;line-height:1.8">
<span class="badge-bad">주의 필요</span><br>
• 절대 수치 기반 임상 진단 ✗<br>
• 논문 정량 데이터 단독 사용 ✗<br>
• 규제 제출용 분석 데이터 ✗<br>
• 배율 미기록 시 면적값 비교 ✗
</div>
""", unsafe_allow_html=True)

                    st.markdown("---")

                    # ─── 리포트 다운로드 ───
                    report = f"""피부 각질세포 AI 분석 리포트
생성: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
바이오분석오락실 STAGE 06 · circlenam.github.io/biogame
분석 모델: claude-opus-4-5 (Claude Vision)

[형태학 파라미터]
평균 면적 (AI 추정): {Math_round(result.get('area_um2',0))} µm²
원형도 (AI 추정): {float(result.get('circularity',0)):.2f}
피복률 (픽셀 실측): {result.get('coverage_pct',0):.1f}%
크기 CV% (AI 추정): {float(result.get('cv_pct',0)):.1f}%
추정 세포 수: {Math_round(result.get('cell_count',0))}개
파편화 비율 (AI 추정): {result.get('fragment_pct',0):.1f}%
이미지 품질: {result.get('image_quality','—')}
분석 신뢰도: {result.get('confidence','—')}

[피부 상태]
장벽: {result.get('barrier','—')}
각질층: {result.get('layer','—')}
균일성: {result.get('uniformity','—')}
수분: {result.get('moisture','—')}
종합 점수: {result.get('score',0)}점 / 100

[AI 판독 소견]
{result.get('opinion','')}

[분석 한계]
{result.get('limitations','')}

[활용 주의사항]
- 면적(µm²)은 캘리브레이션 없이 절대값 부정확. 동일 배율 상대 비교 권장.
- 피복률(%)은 픽셀 직접 측정으로 가장 신뢰 가능.
- 본 결과는 교육·연구 참고용이며 임상 진단을 대체하지 않습니다.
"""
                    st.download_button(
                        label="▼ 리포트 다운로드 (.txt)",
                        data=report.encode("utf-8"),
                        file_name=f"skin-report-{__import__('datetime').datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )

            except json.JSONDecodeError:
                st.error("AI 응답 파싱 오류. 다시 시도해주세요.")
            except anthropic.APIConnectionError:
                st.error("API 연결 오류. 인터넷 연결과 API 키를 확인해주세요.")
            except anthropic.AuthenticationError:
                st.error("API 키가 올바르지 않습니다. 사이드바에서 확인해주세요.")
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")

else:
    st.markdown("""
<div style="text-align:center;padding:3rem;border:1px dashed #1e3a5f;background:#0f1628">
<div class="mono" style="font-size:36px;color:#1e3a5f;margin-bottom:1rem">⬡</div>
<div style="font-size:14px;color:#5a7a9a">테이프스트리핑 후 스마트폰 현미경으로 촬영한 이미지를 업로드하세요</div>
<div class="mono" style="font-size:11px;color:#1e3a5f;margin-top:.5rem">JPG · PNG · TIFF</div>
</div>
""", unsafe_allow_html=True)
