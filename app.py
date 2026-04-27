import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import urllib.request
import xml.etree.ElementTree as ET
import requests
import re
import smtplib
import json
import firebase_admin
import google.generativeai as genai
import plotly.graph_objects as go
from firebase_admin import credentials, firestore
from email.mime.text import MIMEText

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- ✨ 모바일 최적화 & 진짜 앱처럼 보이게 만드는 마법의 CSS ---
st.set_page_config(page_title="나만의 AI 퀀트 비서", page_icon="🚀", layout="wide", initial_sidebar_state="auto")

st.markdown("""
<style>
	/* 1. 불필요한 상단 여백 줄이기 (모바일 화면 낭비 방지) */
	.block-container {
		padding-top: 2rem;
		padding-bottom: 2rem;
	}
	
	/* 2. 우측 상단 Streamlit 기본 메뉴 숨기기 (진짜 내 앱처럼!) */
	#MainMenu {visibility: hidden;}
	
	/* 3. 하단 워터마크 꼬리표 숨기기 (단, header는 사이드바 버튼을 위해 살려둡니다!) */
	footer {visibility: hidden;}
		
	/* 4. 숫자 요약(Metric) 박스를 예쁜 라운드 카드 형태로 묶어주기 */
	div[data-testid="metric-container"] {
		background-color: rgba(255, 255, 255, 0.05);
		border: 1px solid rgba(255, 255, 255, 0.1);
		padding: 15px;
		border-radius: 15px;
		box-shadow: 0 4px 6px rgba(0,0,0,0.1);
		transition: transform 0.2s;
	}
	/* 모바일에서 카드 누를 때 살짝 들어가는 터치 애니메이션 */
	div[data-testid="metric-container"]:active {
		transform: scale(0.98);
	}
</style>
""", unsafe_allow_html=True)

# --- ✨ 리팩토링 1: 앱 설정 및 AI API 글로벌 세팅 ---
st.set_page_config(layout="wide")
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_data(ttl=3600)
def load_data(ticker):
	data = yf.Ticker(ticker)
	df_history = data.history(period="2y")
	info_data = {}
	try:
		info_data = data.info
	except:
		pass
	return df_history, info_data

@st.cache_data(ttl=3600)
def get_finviz_data(ticker):
	try:
		url = f"https://finviz.com/quote.ashx?t={ticker}"
		headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
		res = requests.get(url, headers=headers, timeout=5)
		html = res.text

		def extract(label):
			try:
				match = re.search(rf'{label}</td>.*?<b>([^<]+)</b>', html, re.DOTALL)
				if match and match.group(1).strip() not in ['-', 'N/A']:
					return float(match.group(1).replace(',', ''))
			except:
				pass
			return 0

		pe = extract('P/E')
		pbr = extract('P/B')
		target = extract('Target Price')
		
		return pe, pbr, target
	except:
		return 0, 0, 0

@st.cache_data(ttl=3600)
def get_naver_finance_data(ticker):
	try:
		code = ticker.split('.')[0]
		url = f"https://finance.naver.com/item/main.naver?code={code}"
		headers = {'User-Agent': 'Mozilla/5.0'}
		res = requests.get(url, headers=headers, timeout=5)
		html = res.text

		pe_match = re.search(r'<em id="_per">([^<]+)</em>', html)
		pb_match = re.search(r'<em id="_pbr">([^<]+)</em>', html)

		pe = float(pe_match.group(1).replace(',', '')) if pe_match else 0
		pb = float(pb_match.group(1).replace(',', '')) if pb_match else 0
		
		return pe, pb
	except:
		return 0, 0

@st.cache_data(ttl=3600)
def get_exchange_rate():
	with st.spinner("🌍 실시간 원/달러 환율을 가져오는 중..."):
		try:
			rate_data = yf.Ticker("KRW=X")
			current_rate = float(rate_data.history(period="1d")['Close'].iloc[-1])
			return current_rate
		except:
			st.warning("환율을 불러오지 못해 임시 환율(1350원)을 적용합니다.")
			return 1350.0

@st.cache_data(ttl=3600)
def get_news_and_ai_summary(ticker):
	url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
	req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
	response = urllib.request.urlopen(req)
	root = ET.fromstring(response.read())

	items = root.findall('.//item')
	if not items:
		return None, "현재 이 종목에 대한 최신 뉴스가 없습니다.", 50

	news_titles = []
	news_md_list = []
	for item in items[:5]:
		title = item.find('title').text
		link = item.find('link').text
		publisher = item.find('source').text if item.find('source') is not None else "News"
		news_titles.append(title)
		news_md_list.append(f"- [{title}]({link}) ({publisher})")

	news_markdown = "\n".join(news_md_list)

	# API 세팅은 상단으로 옮겼으므로 모델만 바로 호출
	model = genai.GenerativeModel('gemini-3-flash-preview')
	prompt = f"""
	너는 월스트리트의 수석 주식 분석가야. 다음은 오늘 '{ticker}' 주식에 대한 최신 영문 뉴스 헤드라인 5개야.
	뉴스: {news_titles}

	이 뉴스들을 종합해서 한국어로 분석해 줘.
	1. 현재 이 주식의 상황을 아주 쉬운 한국어로 3줄로 요약해.
	2. 종합적으로 이 뉴스들이 '🟢 호재', '🔴 악재', '⚪ 중립' 중 어떤 것에 해당하는지 결론을 내려줘.
	3. 중요: 맨 마지막 줄에는 반드시 이 뉴스의 '감성 점수'를 0부터 100 사이의 숫자(정수)로만 딱 하나 적어줘. (0=최악의 악재, 50=중립, 100=최고의 호재. 예: 85)
	"""
	ai_response = model.generate_content(prompt)
	text = ai_response.text

	score = 50
	try:
		numbers = re.findall(r'\d+', text)
		if numbers:
			score = int(numbers[-1])
			score = max(0, min(100, score))
	except:
		pass

	return news_markdown, text, score

# --- Firebase 초기화 ---
try:
	firebase_admin.get_app()
except ValueError:
	key_dict = json.loads(st.secrets["FIREBASE_KEY"])
	cred = credentials.Certificate(key_dict)
	firebase_admin.initialize_app(cred)

db = firestore.client()
doc_ref = db.collection('user_data').document('my_portfolio')
account_ref = db.collection('user_data').document('my_account')

if 'account' not in st.session_state:
	acc_doc = account_ref.get()
	if acc_doc.exists:
		st.session_state['account'] = acc_doc.to_dict()
	else:
		default_account = {
			"cash": 10000.0,
			"holdings": {}
		}
		st.session_state['account'] = default_account
		account_ref.set(default_account)

if 'portfolio' not in st.session_state:
	doc = doc_ref.get()
	if doc.exists:
		port_data = doc.to_dict()
		for theme, items in port_data.items():
			if isinstance(items, list):
				port_data[theme] = {ticker: ticker for ticker in items}
		st.session_state['portfolio'] = port_data
		doc_ref.set(port_data)
	else:
		default_portfolio = {
			"💻 빅테크 & AI": {'AAPL': '애플', 'MSFT': '마이크로소프트', 'GOOGL': '구글'},
			"🏎️ F1 & 모터스포츠": {'FWONK': '포뮬러원', 'RACE': '페라리', 'F': '포드'},
			"🎮 PC 게임 & 하드웨어": {'NVDA': '엔비디아', 'AMD': 'AMD', 'EA': 'EA스포츠'},
			"🧪 화학 & 헬스케어": {'JNJ': '존슨앤존슨', 'PFE': '화이자', 'TMO': '써모피셔'},
			"📈 금융 & 데이터": {'JPM': 'JP모건', 'PLTR': '팔란티어', 'SNOW': '스노우플레이크'}
		}
		st.session_state['portfolio'] = default_portfolio
		doc_ref.set(default_portfolio)

# --- 사이드바 ---
st.sidebar.title("📁 내 포트폴리오 (DB 연동됨 ☁️)")

with st.sidebar.expander("➕ 새 종목/테마 검색해서 추가하기"):
	new_theme = st.text_input("테마 이름", "💻 빅테크 & AI")
	search_add_keyword = st.text_input("🔍 기업명 검색 (예: SK하이닉스, TSLA)")
	custom_name = st.text_input("🏷️ 리스트에 표시할 이름 (선택사항)", placeholder="예: 갓플, 킹비디아")

	target_ticker_to_add = None
	custom_name_to_save = None

	if search_add_keyword:
		search_url = "https://query2.finance.yahoo.com/v1/finance/search"
		headers = {'User-Agent': 'Mozilla/5.0'}
		params = {'q': search_add_keyword.strip(), 'lang': 'ko', 'region': 'KR', 'quotesCount': 5}
		try:
			res = requests.get(search_url, params=params, headers=headers, timeout=5)
			quotes = res.json().get('quotes', [])
			if quotes:
				options = [f"{q['symbol']} - {q.get('shortname', q.get('longname', '이름 없음'))}" for q in quotes if q.get('quoteType') in ['EQUITY', 'ETF']]
				if options:
					selected_add_option = st.selectbox("👇 추가할 종목 선택", options)
					target_ticker_to_add = selected_add_option.split(' ')[0]
					default_name = selected_add_option.split(' - ')[1] if ' - ' in selected_add_option else target_ticker_to_add
					custom_name_to_save = custom_name if custom_name else default_name
				else:
					st.warning("일치하는 주식/ETF가 없습니다.")
			else:
				st.warning("검색 결과가 없습니다.")
		except:
			st.warning("검색 중 오류가 발생했습니다.")

	if st.button("리스트에 추가"):
		if target_ticker_to_add:
			if new_theme not in st.session_state['portfolio']:
				st.session_state['portfolio'][new_theme] = {}
			if target_ticker_to_add not in st.session_state['portfolio'][new_theme]:
				st.session_state['portfolio'][new_theme][target_ticker_to_add] = custom_name_to_save
				doc_ref.set(st.session_state['portfolio'])
				st.success(f"'{new_theme}' 테마에 '{custom_name_to_save}' 추가 완료!")
				st.rerun()
			else:
				st.warning("이미 있는 종목입니다.")
		else:
			st.error("먼저 검색창에서 종목을 찾아 선택해주세요!")

with st.sidebar.expander("🗑️ 잘못 추가된 종목 삭제하기"):
	del_theme = st.selectbox("삭제할 테마 선택", list(st.session_state['portfolio'].keys()), key="del_theme")
	if st.session_state['portfolio'][del_theme]:
		del_ticker = st.selectbox(
			"삭제할 종목 선택", 
			options=list(st.session_state['portfolio'][del_theme].keys()),
			format_func=lambda x: f"{st.session_state['portfolio'][del_theme][x]} ({x})",
			key="del_ticker"
		)
		if st.button("❌ 이 종목 지우기"):
			del st.session_state['portfolio'][del_theme][del_ticker]
			if len(st.session_state['portfolio'][del_theme]) == 0:
				del st.session_state['portfolio'][del_theme]
			doc_ref.set(st.session_state['portfolio'])
			st.success("삭제 완료!")
			st.rerun()
	else:
		st.info("이 테마에는 삭제할 종목이 없습니다.")

st.sidebar.divider()
st.sidebar.write("👇 분석할 테마와 종목을 선택하세요")

theme_list = list(st.session_state['portfolio'].keys())
target_theme = "💻 빅테크 & AI"
default_idx = theme_list.index(target_theme) if target_theme in theme_list else 0

selected_theme = st.sidebar.selectbox("📂 관심 테마", theme_list, index=default_idx)
theme_dict = st.session_state['portfolio'][selected_theme]

selected_ticker = st.sidebar.radio(
	f"{selected_theme} 종목", 
	options=list(theme_dict.keys()), 
	format_func=lambda x: f"📌 {theme_dict[x]} ({x})",
	label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.subheader("🔔 매수 타이밍 알림 봇")

if st.sidebar.button("🚀 바겐세일 종목 스캔 & 알림 쏘기"):
	with st.spinner("전체 종목의 RSI를 분석 중입니다..."):
		buy_list = []
		for theme, tickers in st.session_state['portfolio'].items():
			for t in tickers:
				try:
					df_temp, _ = load_data(t)
					delta = df_temp['Close'].diff()
					up = delta.clip(lower=0)
					down = -1 * delta.clip(upper=0)
					ema_up = up.ewm(com=13, adjust=False).mean()
					ema_down = down.ewm(com=13, adjust=False).mean()
					rs = ema_up / ema_down
					df_temp['RSI'] = 100 - (100 / (1 + rs))
					latest_rsi = df_temp['RSI'].iloc[-1]
					if latest_rsi <= 30:
						buy_list.append(f"✅ **{t}** (현재 RSI: {latest_rsi:.1f}) - {theme}")
				except:
					pass

		if buy_list:
			# 알림 메시지 예쁘게 포맷팅
			alert_message = "🚨 **[주식 AI 봇] 바겐세일 매수 타이밍 포착!** 🚨\n\n" + "\n".join(buy_list) + "\n\n👉 *당장 앱에 접속해서 차트를 확인하세요!*"
			
			# 1. 🎮 디스코드 웹훅 전송
			try:
				if "discord" in st.secrets and "webhook_url" in st.secrets["discord"]:
					discord_url = st.secrets["discord"]["webhook_url"]
					payload = {"content": alert_message}
					res = requests.post(discord_url, json=payload)
					if res.status_code == 204:
						st.sidebar.success("🎮 디스코드로 매수 타이밍을 전송했습니다!")
					else:
						st.sidebar.warning(f"디스코드 전송 실패 (상태 코드: {res.status_code})")
			except Exception as e:
				st.sidebar.error(f"디스코드 연동 에러: {e}")

			# 2. 📧 이메일 전송 (기존 기능 유지)
			try:
				if "email" in st.secrets:
					sender = st.secrets["email"]["sender"]
					password = st.secrets["email"]["password"]
					receiver = st.secrets["email"]["receiver"]

					email_body = alert_message.replace("**", "") # 이메일에서는 마크다운 별표 제거
					msg = MIMEText(email_body)
					msg['Subject'] = "[주식 AI 봇] 강한 매수 찬스 알림!"
					msg['From'] = sender
					msg['To'] = receiver

					with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
						server.login(sender, password)
						server.send_message(msg)
					st.sidebar.success("📧 이메일로도 발송을 완료했습니다!")
			except Exception as e:
				st.sidebar.warning("이메일 전송은 실패했습니다. (Secrets 설정을 확인하세요)")
		else:
			st.sidebar.info("지금은 RSI 30 이하인 바겐세일 종목이 없습니다. (총알을 아끼세요!)")

# --- 메인 화면 ---
st.title("주식 AI 분석 앱")

with st.expander("💼 나의 모의투자 계좌 현황", expanded=True):
	my_cash = st.session_state['account']['cash']
	my_holdings = st.session_state['account']['holdings']

	total_stock_value_usd = 0.0
	total_expected_div_usd = 0.0 # ✨ 추가된 연간 배당금 합산 변수
	STARTING_BALANCE = 10000.0
	current_krw_rate = get_exchange_rate()

	if my_holdings:
		st.write("📦 **보유 주식 상세 내역:**")
		
		for ticker, info_dict in my_holdings.items():
			is_kr = ticker.endswith('.KS') or ticker.endswith('.KQ')
			sym = "₩" if is_kr else "$"
			decimals = 0 if is_kr else 2
			
			try:
				df_temp, info_temp = load_data(ticker)
				current_p = info_temp.get('currentPrice') or info_temp.get('regularMarketPrice')
				if not current_p: current_p = df_temp['Close'].iloc[-1]
				
				# ✨ 1주당 실제 배당금(달러/원) 확실하게 가져오기
				div_rate = info_temp.get('dividendRate', 0)
				div_yield = info_temp.get('dividendYield', 0)
				
				# (혹시 API가 % 단위로 뻥튀기해서 줄 때를 대비한 안전장치)
				if div_yield and div_yield > 1: 
					div_yield = div_yield / 100
			except:
				current_p = info_dict['avg_price']
				div_rate = 0
				div_yield = 0
			
			shares = info_dict['shares']
			avg_price = info_dict['avg_price']
			
			profit_pct = ((current_p - avg_price) / avg_price) * 100
			profit_amount = (current_p - avg_price) * shares
			current_total_value = current_p * shares
			
			# ✨ 1년 예상 배당금 계산 (가장 정확한 1주당 배당금(Rate) 우선 사용!)
			if div_rate:
				expected_annual_div = shares * div_rate
			else:
				expected_annual_div = current_total_value * div_yield
			
			if is_kr:
				total_stock_value_usd += (current_total_value / current_krw_rate)
				total_expected_div_usd += (expected_annual_div / current_krw_rate)
			else:
				total_stock_value_usd += current_total_value
				total_expected_div_usd += expected_annual_div
				
			arrow = "🔴" if profit_pct < 0 else "🟢"
			plus = "+" if profit_pct > 0 else ""
			
			st.markdown(f"- **{ticker}** | {shares}주 | 평단가: {sym}{avg_price:,.{decimals}f} ➔ **현재가: {sym}{current_p:,.{decimals}f}** |  {arrow} **{plus}{profit_pct:.2f}%** | 💸 연 배당금: {sym}{expected_annual_div:,.{decimals}f}")
			
	else:
		st.info("현재 보유 중인 주식이 없습니다. 맘에 드는 종목을 매수해 보세요!")

	# 2. 내 총 계좌 요약
	st.divider()
	total_account_value = my_cash + total_stock_value_usd
	total_profit_pct = ((total_account_value / STARTING_BALANCE) - 1) * 100
	
	st.write("💰 **총 계좌 요약:**")
	acc_c1, acc_c2, acc_c3 = st.columns(3)
	acc_c1.metric("보유 현금 (달러)", f"${my_cash:,.2f}")
	acc_c2.metric("총 계좌 자산", f"${total_account_value:,.2f}", f"{total_profit_pct:+.2f}% (원금 $10,000 대비)")
	acc_c3.metric("🎉 내 포트폴리오 연간 배당금", f"${total_expected_div_usd:,.2f}", "가만히 있어도 들어오는 꽁돈!")

	# ✨ 3. 내 실제 계좌 기반 배당 재투자(DRIP) 시뮬레이터!
	if total_expected_div_usd > 0:
		with st.expander("❄️ 내 계좌 배당 재투자(DRIP) 스노우볼 굴려보기", expanded=False):
			st.write("지금 세팅된 포트폴리오의 배당금을 빼 쓰지 않고 계속 재투자한다면?")
			drip_c1, drip_c2, drip_c3 = st.columns(3)
			
			port_cagr = drip_c1.number_input("포트폴리오 평균 예상 상승률 (%)", value=8.0, step=1.0)
			
			# 내 계좌의 실제 평균 배당률을 AI가 역산해서 딱 꽂아줌!
			port_div_yield = (total_expected_div_usd / total_stock_value_usd) * 100 if total_stock_value_usd > 0 else 0
			drip_c2.metric("내 계좌의 실제 평균 배당률", f"{port_div_yield:.2f}%")
			
			monthly_add = drip_c3.number_input("매월 추가 납입금 ($)", value=500, step=100)
			invest_years = st.slider("투자 유지 기간 (년)", 1, 40, 20, key='port_drip_years')

			if st.button("❄️ 내 포트폴리오 스노우볼 굴리기 (시작)", use_container_width=True):
				with st.spinner("내 계좌의 복리 마법을 계산 중입니다... ⏳"):
					months = invest_years * 12
					monthly_return = port_cagr / 100 / 12
					monthly_dividend = port_div_yield / 100 / 12
					
					data_records = []
					total_principal = total_account_value
					current_balance_no_drip = total_account_value
					current_balance_drip = total_account_value
					
					for m in range(1, months + 1):
						total_principal += monthly_add
						current_balance_no_drip = current_balance_no_drip * (1 + monthly_return) + monthly_add
						current_balance_drip = current_balance_drip * (1 + monthly_return + monthly_dividend) + monthly_add
						
						if m % 12 == 0:
							data_records.append({
								"Year": m // 12,
								"원금 (Principal)": total_principal,
								"주가 수익만 (No DRIP)": current_balance_no_drip,
								"배당 재투자 (DRIP)": current_balance_drip
							})
					
					df_drip = pd.DataFrame(data_records)
					
					st.success(f"시간이 무기입니다! {invest_years}년 후 내 계좌의 놀라운 결과입니다.")
					r_col1, r_col2, r_col3 = st.columns(3)
					r_col1.metric("총 투입 원금 (현재 자산 + 납입금)", f"${total_principal:,.0f}")
					r_col2.metric("배당금 안 합친 최종 자산", f"${current_balance_no_drip:,.0f}")
					drip_bonus = current_balance_drip - current_balance_no_drip
					r_col3.metric("배당 재투자 시 최종 자산", f"${current_balance_drip:,.0f}", f"+${drip_bonus:,.0f} (스노우볼 효과!)")
					
					fig_snow = go.Figure()
					fig_snow.add_trace(go.Scatter(x=df_drip['Year'], y=df_drip['원금 (Principal)'], fill='tozeroy', mode='none', name='내 순수 원금', fillcolor='rgba(128, 128, 128, 0.2)'))
					fig_snow.add_trace(go.Scatter(x=df_drip['Year'], y=df_drip['주가 수익만 (No DRIP)'], fill='tonexty', mode='none', name='단순 주가 상승분', fillcolor='rgba(0, 176, 246, 0.4)'))
					fig_snow.add_trace(go.Scatter(x=df_drip['Year'], y=df_drip['배당 재투자 (DRIP)'], fill='tonexty', mode='none', name='배당 재투자 (복리의 마법)', fillcolor='rgba(0, 255, 136, 0.6)'))
					fig_snow.update_layout(title=f"현재 내 포트폴리오 기반 자산 증식 시뮬레이션", xaxis_title="투자 기간 (년)", yaxis_title="자산 규모 (USD)", template="plotly_dark", hovermode="x unified", height=500)
					st.plotly_chart(fig_snow, use_container_width=True)

st.divider()

# --- ✨ 신규 기능: 노벨상 수상 알고리즘! 포트폴리오 최적화 ---
st.subheader("⚖️ AI 포트폴리오 비중 최적화 (마코위츠 모델)")
with st.expander("내 계좌의 주식들을 어떤 비율로 섞어야 가장 안전하고 수익이 높을까?", expanded=False):
	my_tickers = list(st.session_state['account']['holdings'].keys())
	if len(my_tickers) < 2:
		st.info("💡 최적화를 하려면 모의투자 계좌에 최소 2개 이상의 종목이 있어야 합니다! (계란을 한 바구니에 담지 마세요!)")
	else:
		if st.button("🧠 내 포트폴리오 황금 비율 찾기 (최적화 시작)", use_container_width=True):
			with st.spinner("수천 개의 비율 조합을 시뮬레이션하여 가장 효율적인 전선(Efficient Frontier)을 찾는 중입니다... ⏳"):
				try:
					# 1. 내 보유 종목들의 과거 2년치 데이터 가져오기
					data = yf.download(my_tickers, period="2y")['Close']
					
					# 2. 일일 수익률, 연평균 기대 수익률, 공분산(종목 간의 상관관계) 계산
					returns = data.pct_change().dropna()
					mean_returns = returns.mean() * 252
					cov_matrix = returns.cov() * 252

					# 3. 3000번의 평행우주(랜덤 비중) 생성
					num_portfolios = 3000
					results = np.zeros((3, num_portfolios))
					weights_record = []

					for i in range(num_portfolios):
						weights = np.random.random(len(my_tickers))
						weights /= np.sum(weights) # 비중의 합을 100%로 맞춤
						weights_record.append(weights)
						
						port_return = np.sum(mean_returns * weights)
						port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
						
						results[0,i] = port_std # 리스크 (변동성)
						results[1,i] = port_return # 기대 수익률
						# 샤프 지수 (위험 대비 수익률, 무위험 이자율 2% 가정)
						results[2,i] = (port_return - 0.02) / port_std if port_std > 0 else 0

					# 4. 가장 샤프 지수가 높은 '최적의 포트폴리오' 추출
					max_sharpe_idx = np.argmax(results[2])
					optimal_std = results[0, max_sharpe_idx]
					optimal_ret = results[1, max_sharpe_idx]
					optimal_weights = weights_record[max_sharpe_idx]

					st.success("찾았습니다! 리스크를 최소화하고 수익을 극대화하는 황금 비율입니다. 🏆")
					
					# 5. 멋진 차트로 결과 보여주기
					opt_c1, opt_c2 = st.columns([1, 1])
					
					with opt_c1:
						st.write("**🏆 AI 추천 종목 비중 (원형 차트)**")
						fig_pie = go.Figure(data=[go.Pie(labels=my_tickers, values=optimal_weights, hole=.4, textinfo='label+percent')])
						fig_pie.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=30, b=20))
						st.plotly_chart(fig_pie, use_container_width=True)
						
					with opt_c2:
						st.write("**📊 효율적 전선 (Efficient Frontier)**")
						fig_scatter = go.Figure()
						# 3000개의 랜덤 포트폴리오 구름
						fig_scatter.add_trace(go.Scatter(
							x=results[0], y=results[1], mode='markers',
							marker=dict(color=results[2], colorscale='Viridis', showscale=True, size=5, colorbar=dict(title="샤프지수")),
							name='시뮬레이션 포트폴리오'
						))
						# 최적의 포트폴리오 붉은 별표!
						fig_scatter.add_trace(go.Scatter(
							x=[optimal_std], y=[optimal_ret], mode='markers',
							marker=dict(color='red', size=18, symbol='star', line=dict(color='white', width=1)),
							name='🌟 최적의 황금 비율'
						))
						fig_scatter.update_layout(xaxis_title="위험 (리스크/변동성)", yaxis_title="기대 수익률", template="plotly_dark", height=350, margin=dict(l=20, r=20, t=30, b=20))
						st.plotly_chart(fig_scatter, use_container_width=True)

					st.info(f"💡 이 황금 비율대로 투자할 경우, 과거 2년 데이터 기준 **예상 연수익률은 {optimal_ret*100:.1f}%**, **예상 리스크(변동성)는 {optimal_std*100:.1f}%** 로 계산됩니다.")
				except Exception as e:
					st.error(f"최적화 계산 중 오류가 발생했습니다. (보유 종목의 상장 기간이 너무 짧거나 데이터가 부족할 수 있습니다.) 에러: {e}")

st.divider()

# --- ✨ 신규 기능: 나만의 퀀트 스크리너 ---
st.subheader("🔎 나만의 퀀트 종목 발굴기 (미니 스크리너)")
with st.expander("조건에 맞는 보석 같은 주식을 찾아보세요! (Top 우량주 & 내 관심종목 대상)", expanded=False):
	st.write("설정한 조건에 완벽하게 일치하는 종목만 필터링해서 보여줍니다.")
	
	sc_col1, sc_col2, sc_col3 = st.columns(3)
	max_per = sc_col1.number_input("최대 PER (이하)", value=20.0, step=1.0)
	min_roe = sc_col2.number_input("최소 ROE (%) (이상)", value=15.0, step=1.0)
	min_div = sc_col3.number_input("최소 배당수익률 (%) (이상)", value=1.0, step=0.5)

	if st.button("🚀 조건에 맞는 종목 발굴하기", use_container_width=True):
		with st.spinner("우량주와 내 관심종목 데이터를 맹렬히 스캔 중입니다... (약 10~20초 소요) ⏳"):
			# 1. 검사할 종목 리스트 (미국 대형주 기본 탑재)
			base_tickers = [
				'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'JNJ', 'V', 
				'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'NFLX', 'AMD', 'INTC', 'KO', 
				'PEP', 'CSCO', 'XOM', 'CVX', 'WMT', 'T', 'VZ', 'PFE', 'ABBV', 'MCD'
			]
			
			# 2. 내 포트폴리오에 있는 종목도 검사 대상에 추가
			for theme, tickers in st.session_state['portfolio'].items():
				base_tickers.extend(list(tickers.keys()))
			
			# 중복 제거
			search_list = list(set(base_tickers)) 
			passed_stocks = []
			
			# 진행률 바 표시 (시각적 재미!)
			progress_bar = st.progress(0)
			
			for idx, t in enumerate(search_list):
				try:
					# 빠른 스캔을 위해 info 데이터만 쏙 빼오기
					t_info = yf.Ticker(t).info
					
					t_per = t_info.get('trailingPE', 0)
					if t_per is None: t_per = 0
					
					t_roe = t_info.get('returnOnEquity', 0) * 100 if t_info.get('returnOnEquity') else 0
					t_div = t_info.get('dividendYield', 0) * 100 if t_info.get('dividendYield') else 0
					
					# 3. 퀀트 조건 검사! (PER이 0보다 크고 조건보다 낮으며, ROE와 배당이 조건 이상일 때)
					if (0 < t_per <= max_per) and (t_roe >= min_roe) and (t_div >= min_div):
						passed_stocks.append({
							"종목명": t_info.get('shortName', t),
							"티커": t,
							"PER": round(t_per, 2),
							"ROE (%)": round(t_roe, 2),
							"배당수익률 (%)": round(t_div, 2),
							"현재가 (USD)": t_info.get('currentPrice', 0)
						})
				except:
					pass
				
				# 진행률 애니메이션 업데이트
				progress_bar.progress((idx + 1) / len(search_list))
			
			# 4. 결과 출력
			if passed_stocks:
				st.success(f"🎉 퀀트 조건에 맞는 보석 같은 종목 {len(passed_stocks)}개를 발견했습니다!")
				df_screener = pd.DataFrame(passed_stocks)
				# 인덱스 숨기고 깔끔하게 표출
				st.dataframe(df_screener, use_container_width=True, hide_index=True)
			else:
				st.warning("조건에 맞는 종목이 없습니다. 조건을 조금 더 느슨하게(PER은 높게, 배당은 낮게) 조절해 보세요!")

search_keyword = st.text_input("🔍 기업명(예: 애플, 삼성전자, Tesla) 또는 티커를 입력하세요:", selected_ticker)
ticker_symbol = search_keyword.upper()

if search_keyword and search_keyword != selected_ticker:
	with st.spinner("글로벌 주식 DB에서 종목을 찾는 중입니다... 🌍"):
		search_url = "https://query2.finance.yahoo.com/v1/finance/search"
		headers = {'User-Agent': 'Mozilla/5.0'}
		try:
			res = requests.get(search_url, params={'q': search_keyword.strip()}, headers=headers, timeout=5)
			quotes = res.json().get('quotes', [])

			if quotes:
				options = [f"{q['symbol']} - {q.get('shortname', q.get('longname', '이름 없음'))} ({q.get('exchange', 'N/A')})"
					for q in quotes if q.get('quoteType') in ['EQUITY', 'ETF']]
				if options:
					selected_option = st.selectbox("👇 아래 검색 결과에서 정확한 종목을 선택하세요!", options)
					ticker_symbol = selected_option.split(' ')[0]
				else:
					st.warning("일치하는 주식/ETF를 찾을 수 없습니다. 영문이나 티커로 다시 검색해 보세요.")
					st.stop()
			else:
				st.warning("야후 파이낸스에서 해당 종목을 찾을 수 없습니다.")
				st.stop()
		except Exception as e:
			st.warning("검색 서버에 연결할 수 없습니다. 티커(예: AAPL)를 직접 입력해 주세요.")
			st.stop()

try:
	df, info = load_data(ticker_symbol)

	if df.empty:
		st.warning("⚠️ 주가 데이터를 찾을 수 없습니다. 아래 드롭다운에서 정확한 종목을 선택하거나, 올바른 티커(예: AAPL)를 입력해 주세요!")
		st.stop()

	current_price = info.get('currentPrice') or info.get('regularMarketPrice')
	if not current_price:
		current_price = round(df['Close'].iloc[-1], 2)
	
	high_52 = info.get('fiftyTwoWeekHigh')
	if not high_52:
		high_52 = round(df['High'].tail(252).max(), 2)
	
	per = info.get('trailingPE') or 0
	pbr = info.get('priceToBook') or 0
	target_price = info.get('targetMeanPrice') or 0
	
	is_kr = ticker_symbol.endswith('.KS') or ticker_symbol.endswith('.KQ')

	if is_kr:
		if per == 0 or pbr == 0:
			nv_pe, nv_pb = get_naver_finance_data(ticker_symbol)
			per = nv_pe if per == 0 else per
			pbr = nv_pb if pbr == 0 else pbr
	else:
		if per == 0 or pbr == 0 or target_price == 0:
			fv_pe, fv_pbr, fv_target = get_finviz_data(ticker_symbol)
			per = fv_pe if per == 0 else per
			pbr = fv_pbr if pbr == 0 else pbr
			target_price = fv_target if target_price == 0 else target_price
		
	KRW_RATE = get_exchange_rate()

	def fmt_price(val):
		if is_kr: return f"₩{val:,.0f}"
		return f"${val:,.2f}"

	short_name = info.get('shortName', ticker_symbol)
	st.subheader(f"🏢 {short_name} 요약 정보")

	# ✨ 배당률 정보 추가
	dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0.0
	dividend_rate = info.get('dividendRate', 0)

	col1, col2, col3, col4 = st.columns(4)
	col1.metric("현재 주가", fmt_price(current_price))
	col2.metric("PER (주가수익비율)", f"{per:.2f}" if per > 0 else "N/A")
	col3.metric("52주 최고가", fmt_price(high_52))
	
	if dividend_yield > 0:
		col4.metric("배당수익률", f"{dividend_yield:.2f}%", f"연 {fmt_price(dividend_rate)}")
	else:
		col4.metric("배당수익률", "배당 없음")

	st.write("### 모의투자 매수 / 매도")

	trade_col1, trade_col2, trade_col3 = st.columns([1, 1, 2])
	trade_shares = trade_col1.number_input("수량", min_value=1, value=1, step=1)
	total_trade_amount = trade_shares * current_price
	trade_amount_usd = total_trade_amount / KRW_RATE if is_kr else total_trade_amount

	trade_col3.info(f"💰 총 예상 금액: **{fmt_price(total_trade_amount)}**\n(계좌 차감액: **${trade_amount_usd:,.2f}**)\n\n(내 잔고: ${st.session_state['account']['cash']:,.2f})")

	if trade_col1.button("🟢 매수 (Buy)", use_container_width=True):
		if st.session_state['account']['cash'] >= trade_amount_usd:
			st.session_state['account']['cash'] -= trade_amount_usd
			holdings = st.session_state['account']['holdings']
			if ticker_symbol in holdings:
				old_shares = holdings[ticker_symbol]['shares']
				old_avg = holdings[ticker_symbol]['avg_price']
				new_shares = old_shares + trade_shares
				new_avg = ((old_shares * old_avg) + current_price) / new_shares
				holdings[ticker_symbol]['shares'] = new_shares
				holdings[ticker_symbol]['avg_price'] = new_avg
			else:
				holdings[ticker_symbol] = {'shares': trade_shares, 'avg_price': current_price}
			account_ref.set(st.session_state['account'])
			st.success(f"🎉 {ticker_symbol} {trade_shares}주 매수 완료! (DB 저장됨)")
			st.rerun()
		else:
			st.error("잔고가 부족합니다!😅 (원화/달러 환율을 확인해 보세요)")

	if trade_col2.button("🔴 매도 (Sell)", use_container_width=True):
		holdings = st.session_state['account']['holdings']
		if ticker_symbol in holdings and holdings[ticker_symbol]['shares'] >= trade_shares:
			st.session_state['account']['cash'] += trade_amount_usd
			holdings[ticker_symbol]['shares'] -= trade_shares
			if holdings[ticker_symbol]['shares'] == 0:
				del holdings[ticker_symbol]
			account_ref.set(st.session_state['account'])
			st.success(f"💸 {ticker_symbol} {trade_shares}주 매도 완료! (DB 저장됨)")
			st.rerun()
		else:
			st.error("보유한 주식 수량이 부족합니다! 🤔")

	st.divider()

	st.subheader("⚖️ 기업 가치 평가 (고평가 vs 저평가)")
	
	# ✨ 추가된 친절한 용어 설명서!
	with st.expander("📖 PER, PBR, 월스트리트 목표가가 정확히 무슨 뜻인가요?", expanded=False):
		st.info("""
		* 💰 **PER (주가수익비율):** '내가 투자한 돈을 이 회사가 현재 버는 이익으로 몇 년 만에 다 갚을 수 있는가?'를 뜻해요. PER이 10이라면 10년이 걸린다는 뜻이죠. 보통 이 숫자가 **낮을수록 저평가(싸다)**되었다고 봅니다.
		* 🏢 **PBR (주가순자산비율):** '이 회사가 당장 망해서 공장, 건물, 현금을 다 팔아치웠을 때 내 주식 가격보다 돈이 많이 남는가?'를 뜻해요. PBR이 1보다 낮으면 **회사의 진짜 재산보다 주식이 싸게 거래 중(바겐세일)**이라는 뜻입니다.
		* 🎯 **월스트리트 목표가:** 골드만삭스, JP모건 같은 글로벌 투자은행의 엘리트 애널리스트들이 이 회사의 미래 가치를 종합적으로 뜯어보고 내놓은 **'향후 12개월 예상 적정 주가'**입니다.
		""")

	v_col1, v_col2, v_col3 = st.columns(3)

	with v_col1:
		if per == 0:			
			st.info("PER 정보 없음")
		elif per <= 15:
			st.success(f"💰 PER: {per:.2f}\n\n**저평가 (수익 대비 쌈)**")
		elif per >= 25:
			st.error(f"🔥 PER: {per:.2f}\n\n**고평가 (수익 대비 비쌈)**")
		else:
			st.warning(f"⚖️ PER: {per:.2f}\n\n**적정 수준**")

	with v_col2:
		if pbr == 0:
			st.info("PBR 정보 없음")
		elif pbr <= 1.5:
			st.success(f"💰 PBR: {pbr:.2f}\n\n**저평가 (자산 대비 쌈)**")
		elif pbr >= 3.0:
			st.error(f"🔥 PBR: {pbr:.2f}\n\n**고평가 (자산 대비 비쌈)**")
		else:
			st.warning(f"⚖️ PBR: {pbr:.2f}\n\n**적정 수준**")

	with v_col3:
		if target_price > current_price and current_price > 0:
			up_potential = ((target_price - current_price) / current_price) * 100
			st.success(f"🎯 월스트리트 목표가: {fmt_price(target_price)}\n\n**+{up_potential:.1f}% 상승 여력**")
		elif target_price > 0 and current_price > 0:
			down_potential = ((current_price - target_price) / current_price) * 100
			st.error(f"🎯 월스트리트 목표가: {fmt_price(target_price)}\n\n**-{down_potential:.1f}% 하락 위험 (거품)**")
		else:
			st.info("목표가 정보 없음")
	
	st.divider()

	# ✨ ETF 스마트 판별기 (이 종목이 주식인지 ETF인지 확인)
	is_etf = info.get('quoteType') == 'ETF'

	if is_etf:
		st.subheader("🏢 ETF 심층 분석 안내")
		st.info("💡 **안내:** 검색하신 종목은 개별 기업이 아닌 **ETF(상장지수펀드)**입니다. ETF는 여러 주식을 모아놓은 바구니이므로 개별 기업의 재무제표(영업이익, R&D)나 잉여현금흐름(DCF) 데이터가 존재하지 않습니다.\n\n대신 **아래의 매수/매도 타이밍 차트, 딥러닝 예측, 몬테카를로 시뮬레이션** 등은 ETF의 과거 가격과 거래량을 바탕으로 완벽하게 작동하니 안심하고 투자에 활용해 보세요! 🚀")
		st.divider()
	else:
		st.subheader("🏢 기업 기초체력 (펀더멘털) 분석")
		with st.expander("📖 마진, ROE, 부채비율, 현금흐름이 무슨 뜻인가요?", expanded=False):
			st.info("""
			# ... (이 안의 설명 텍스트는 기존과 동일하게 둡니다) ...
			""")

		with st.expander("📊 재무제표 & 핵심 지표 열어보기 (진짜 돈 넣기 전 필수 확인!)"):
			with st.spinner("야후 파이낸스에서 기업의 재무 장부를 뒤지고 있습니다... ⏳"):
				
				# ✨ [핵심 해결책] 에러가 나더라도 앱이 죽지 않도록 '빈 바구니'를 미리 만들어 둡니다!
				financials = pd.DataFrame()
				balance_sheet = pd.DataFrame()
				cashflow = pd.DataFrame()
				fund_info = {}
				
				try:
					stock_obj = yf.Ticker(ticker_symbol)
					financials = stock_obj.financials
					balance_sheet = stock_obj.balance_sheet
					cashflow = stock_obj.cashflow
					fund_info = stock_obj.info # 정상 작동 시 빈 바구니에 데이터를 채움
					
					margin = fund_info.get('profitMargins', 0) * 100 if fund_info.get('profitMargins') else 0
					if margin == 0 and not financials.empty:
						try: margin = (financials.loc['Net Income'].iloc[0] / financials.loc['Total Revenue'].iloc[0]) * 100
						except: pass
						
					roe = fund_info.get('returnOnEquity', 0) * 100 if fund_info.get('returnOnEquity') else 0
					if roe == 0:
						try: roe = (financials.loc['Net Income'].iloc[0] / balance_sheet.loc['Stockholders Equity'].iloc[0]) * 100
						except: pass
						
					debt_to_eq = fund_info.get('debtToEquity', 0) if fund_info.get('debtToEquity') else 0
					if debt_to_eq == 0:
						try: debt_to_eq = (balance_sheet.loc['Total Debt'].iloc[0] / balance_sheet.loc['Stockholders Equity'].iloc[0]) * 100
						except: pass
						
					op_cashflow = fund_info.get('operatingCashflow', 0) if fund_info.get('operatingCashflow') else 0
					if op_cashflow == 0:
						try: op_cashflow = cashflow.loc['Operating Cash Flow'].iloc[0]
						except: pass

					st.write("💡 **핵심 펀더멘털 지표**")
					f_col1, f_col2, f_col3, f_col4 = st.columns(4)
					
					if margin >= 20:
						f_col1.success(f"순이익률\n\n**{margin:.1f}%** (마진 끝판왕 👑)")
					elif margin > 0:
						f_col1.metric("순이익률 (마진)", f"{margin:.1f}%")
					else:
						f_col1.error(f"순이익률\n\n**{margin:.1f}%** (적자 상태 🚨)")
						
					f_col2.metric("자기자본이익률 (ROE)", f"{roe:.1f}%" if roe != 0 else "N/A")
					f_col3.metric("부채비율 (빚)", f"{debt_to_eq:.1f}%" if debt_to_eq != 0 else "N/A")
					
					if op_cashflow != 0:
						if "KS" in ticker_symbol or "KQ" in ticker_symbol:
							cf_str = f"{op_cashflow / 100000000:,.0f}억 원"
						else:
							cf_str = f"${op_cashflow / 1000000:,.0f}M"
					else:
						cf_str = "N/A"
					f_col4.metric("영업활동 현금흐름", cf_str)

					st.divider()

					st.write("📈 **최근 4년 매출액 vs 당기순이익 성적표** (우상향하는 기업이 최고!)")
					st.caption("💡 주린이 꿀팁: 막대그래프 두 개(매출=회사의 덩치, 당기순이익=진짜 남긴 돈)가 매년 계단처럼 같이 **우상향**하고 있다면 장투하기 아주 좋은 우량주입니다!")
					if not financials.empty:
						fin_df = financials.T.head(4)[::-1]
						if 'Total Revenue' in fin_df.columns and 'Net Income' in fin_df.columns:
							chart_data = pd.DataFrame({
								'매출액 (Revenue)': fin_df['Total Revenue'],
								'당기순이익 (Net Income)': fin_df['Net Income']
							})
							st.bar_chart(chart_data)
						else:
							st.info("이 종목은 상세 매출/이익 차트를 제공하지 않습니다.")
					else:
						st.info("재무제표 데이터가 없습니다. (ETF나 상장 폐지 종목일 수 있습니다.)")

				except Exception as e:
					st.warning(f"재무 데이터를 불러오는 중 오류가 발생했습니다: {e}")

		st.divider()

		st.subheader("🔮 기업의 '진짜 가치' 찾기 (DCF 모델)")
		with st.expander("워렌 버핏처럼 기업의 적정 주가를 직접 계산해 보세요!", expanded=True):
			st.write("회사가 미래에 벌어들일 잉여현금흐름(FCF)을 추정하여 현재 가치로 할인하는 절대 가치 평가 모델입니다.")
			try:
				recent_fcf = 0
				if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
					recent_fcf = cashflow.loc['Free Cash Flow'].iloc[0]
				
				shares_out = fund_info.get('sharesOutstanding', 0)
				total_cash = fund_info.get('totalCash', 0)
				total_debt = fund_info.get('totalDebt', 0)
				if pd.isna(recent_fcf): recent_fcf = 0
				
				auto_growth_rate = 15.0
				try:
					if not financials.empty and 'Total Revenue' in financials.index:
						rev_data = financials.loc['Total Revenue'].dropna()
						if len(rev_data) >= 2:
							latest_rev = rev_data.iloc[0]
							oldest_rev = rev_data.iloc[-1]
							years = len(rev_data) - 1
							if oldest_rev > 0 and latest_rev > 0:
								cagr = ((latest_rev / oldest_rev) ** (1 / years) - 1) * 100
								auto_growth_rate = max(-10.0, min(cagr, 50.0))
				except:
					pass

				st.info(f"💡 과거 재무제표를 분석한 결과, 이 기업의 최근 연평균 매출 성장률은 **{auto_growth_rate:.1f}%**입니다. (슬라이더에 자동 세팅되었습니다!)")
				
				dcf_col1, dcf_col2 = st.columns(2)
				
				with dcf_col1:
					st.markdown("🏢 **[1단계] 현재 기초 체력 (자동 입력됨)**")
					input_fcf = st.number_input("최근 1년 잉여현금흐름 (FCF)", value=float(recent_fcf), step=1000000.0, format="%f")
					input_shares = st.number_input("발행 주식수", value=float(shares_out), step=1000000.0, format="%f")
					input_cash = st.number_input("보유 현금", value=float(total_cash), step=1000000.0, format="%f")
					input_debt = st.number_input("총 부채", value=float(total_debt), step=1000000.0, format="%f")
					
				with dcf_col2:
					st.markdown("📈 **[2단계] 미래 성장률 & 할인율 가정**")
					growth_1_5 = st.slider("향후 1~5년 예상 성장률 (%)", -10.0, 50.0, float(round(auto_growth_rate, 1)), 1.0)
					growth_6_10 = st.slider("향후 6~10년 예상 성장률 (%)", -10.0, 30.0, float(round(max(-10.0, auto_growth_rate * 0.5), 1)), 1.0)
					discount_rate = st.slider("할인율 (WACC, 요구수익률) (%)", 5.0, 20.0, 10.0, 0.5)
					terminal_growth = st.slider("영구 성장률 (10년 이후) (%)", 0.0, 5.0, 2.5, 0.1)
					
				if st.button("📊 이 조건으로 적정 주가 계산하기", type="primary", use_container_width=True):
					if input_fcf <= 0 or input_shares <= 0:
						st.warning("FCF(잉여현금흐름)와 발행 주식수는 0보다 커야 정상적인 계산이 가능합니다.")
					elif discount_rate <= terminal_growth:
						st.warning("수학적 오류! 할인율은 영구 성장률보다 커야 합니다.")
					else:
						with st.spinner("미래의 현금흐름을 현재 가치로 끌어오는 중... ⏳"):
							future_fcfs = []
							current_proj_fcf = input_fcf
							for year in range(1, 11):
								if year <= 5: current_proj_fcf *= (1 + (growth_1_5 / 100))
								else: current_proj_fcf *= (1 + (growth_6_10 / 100))
								discounted_fcf = current_proj_fcf / ((1 + (discount_rate / 100)) ** year)
								future_fcfs.append(discounted_fcf)
								
							sum_discounted_fcf = sum(future_fcfs)
							terminal_value = (current_proj_fcf * (1 + (terminal_growth / 100))) / ((discount_rate / 100) - (terminal_growth / 100))
							discounted_tv = terminal_value / ((1 + (discount_rate / 100)) ** 10)
							enterprise_value = sum_discounted_fcf + discounted_tv
							equity_value = enterprise_value + input_cash - input_debt
							intrinsic_value = equity_value / input_shares
							margin_of_safety = ((intrinsic_value - current_price) / current_price) * 100 if current_price > 0 else 0
								
							st.success("계산 완료! 시장이 평가하는 가격과 데이터가 말하는 진짜 가치를 비교해 보세요.")
							res_c1, res_c2, res_c3 = st.columns(3)
							res_c1.metric("현재 시장 주가", fmt_price(current_price))
							
							if intrinsic_value > 0:
								res_c2.metric("내가 계산한 적정 주가", fmt_price(intrinsic_value))
								if margin_of_safety > 0:
									res_c3.metric("안전 마진 (저평가율)", f"+{margin_of_safety:.1f}%", "저평가 (매수 찬스!)")
									st.info(f"💡 현재 주가보다 적정 가치가 **{margin_of_safety:.1f}%** 더 높습니다! 바겐세일 상태일 수 있습니다.")
								else:
									res_c3.metric("안전 마진 (고평가율)", f"{margin_of_safety:.1f}%", "고평가 (거품 주의)")
									st.warning(f"⚠️ 현재 주가가 적정 가치보다 비쌉니다. 거품이 끼어있을 수 있으니 주의하세요!")
							else:
								st.error("계산된 적정 주가가 마이너스입니다.")
			except Exception as e:
				st.error(f"DCF 데이터를 준비하는 중 오류가 발생했습니다: {e}")

		st.divider()

		st.subheader("🏭 산업(Sector) 맞춤형 심층 분석")
		with st.expander("📖 R&D, ROA, 마진율... 이 산업에서는 어떤 숫자가 좋은 건가요?", expanded=False):
			st.info("""
			* 🔬 **R&D (연구개발) 투자 비율:** 주로 빅테크나 제약/바이오에서 생명줄입니다. 번 돈의 **10~20% 이상 꾸준히 투자**한다면 미래가 밝은 기업 (🟢 **매수 긍정**).
			* 🏦 **총자산이익률 (ROA):** 주로 은행/금융주를 평가할 때 봅니다. 은행은 ROA가 **1~1.5%만 넘어도 돈을 기가 막히게 잘 굴리는 훌륭한 은행**입니다 (🟢 **매수 긍정**).
			* 🍔 **영업이익률 (Operating Margin):** 소비재, 제조업의 핵심이자 **삼성전자/애플 같은 하드웨어 테크 기업에게도 제일 중요한 지표**입니다! 재료비, 인건비 다 떼고 남긴 돈으로, **제조업은 10% 이상, 소프트웨어는 20~30% 이상**이면 훌륭합니다. (🔴 떨어지면 **매도 강력 주의**).
			* 💰 **매출 총이익률 (Gross Margin):** 순수 '원가'만 뺀 비율이에요. **50% 이상으로 아주 높다면**, 독점적 브랜드 파워를 가졌다는 뜻입니다 (🟢 **강력 매수**).
			* 📈 **매출 성장률 (YoY):** 작년 대비 회사의 덩치가 얼마나 커졌는지 보여줍니다. **꾸준히 두 자릿수(+10% 이상) 성장**해야 좋습니다.
			""")

		with st.expander(f"'{short_name}'이(가) 속한 산업의 핵심 지표 파헤치기", expanded=False):
			sector = fund_info.get('sector', '알 수 없음')
			industry = fund_info.get('industry', '알 수 없음')
			st.write(f"🏷️ **섹터:** {sector} | **세부 산업:** {industry}")
			
			try:
				gross_margins = fund_info.get('grossMargins', 0) * 100 if fund_info.get('grossMargins') else 0
				operating_margins = fund_info.get('operatingMargins', 0) * 100 if fund_info.get('operatingMargins') else 0
				revenue_growth = fund_info.get('revenueGrowth', 0) * 100 if fund_info.get('revenueGrowth') else 0
				
				rnd_expense = 0
				if not financials.empty and 'Research And Development' in financials.index:
					rnd_expense = financials.loc['Research And Development'].iloc[0]
					total_rev = financials.loc['Total Revenue'].iloc[0]
					rnd_ratio = (rnd_expense / total_rev) * 100 if total_rev > 0 else 0
				else:
					rnd_ratio = 0
					
				if sector == 'Technology' or sector == 'Healthcare':
					st.info("💡 **기술(Tech) 및 헬스케어 산업**은 미래를 위한 **'연구개발(R&D)'**과 당장의 **'영업이익률'**을 동시에 봐야 합니다!")
					s_col1, s_col2, s_col3, s_col4 = st.columns(4)
					s_col1.metric("R&D 투자 비율", f"{rnd_ratio:.1f}%" if rnd_ratio > 0 else "데이터 없음")
					s_col2.metric("영업이익률", f"{operating_margins:.1f}%")
					s_col3.metric("매출 총이익률", f"{gross_margins:.1f}%")
					s_col4.metric("매출 성장률 (YoY)", f"{revenue_growth:.1f}%")
				elif sector == 'Financial Services':
					st.info("💡 **금융 산업**은 PER보다 **'자산(ROA) 대비 수익성'**과 **'영업이익률'**이 중요합니다!")
					roa = fund_info.get('returnOnAssets', 0) * 100 if fund_info.get('returnOnAssets') else 0
					s_col1, s_col2, s_col3 = st.columns(3)
					s_col1.metric("총자산이익률 (ROA)", f"{roa:.2f}%")
					s_col2.metric("영업이익률", f"{operating_margins:.1f}%")
					s_col3.metric("매출 성장률 (YoY)", f"{revenue_growth:.1f}%")
				elif sector == 'Consumer Cyclical' or sector == 'Consumer Defensive' or sector == 'Industrials':
					st.info("💡 **소비재 및 산업재(제조업)**는 원가를 떼고 남기는 **'영업이익률'**과 흔들리지 않는 **'매출 성장'**이 핵심입니다!")
					s_col1, s_col2, s_col3 = st.columns(3)
					s_col1.metric("영업이익률", f"{operating_margins:.1f}%")
					s_col2.metric("매출 총이익률", f"{gross_margins:.1f}%")
					s_col3.metric("매출 성장률 (YoY)", f"{revenue_growth:.1f}%")
				else:
					st.info("💡 이 산업의 기본적인 수익성과 성장성을 확인해 보세요.")
					s_col1, s_col2, s_col3 = st.columns(3)
					s_col1.metric("영업이익률", f"{operating_margins:.1f}%")
					s_col2.metric("매출 총이익률", f"{gross_margins:.1f}%")
					s_col3.metric("매출 성장률 (YoY)", f"{revenue_growth:.1f}%")
			except Exception as e:
				st.warning(f"산업 세부 지표를 불러오는 데 실패했습니다: {e}")

# --- ✨ 신규 기능: 내부자 거래 & 기관 수급 추적 ---
	st.divider()
	st.subheader("🕵️‍♂️ 내부자 거래 & 기관 수급 추적 (돈의 흐름)")
	with st.expander(f"'{short_name}'의 회장님과 거대 자본은 이 주식을 사고 있을까?", expanded=False):
		st.info("""
		* 👔 **내부자 지분율:** CEO나 임원진이 회사 주식을 얼마나 들고 있는지 보여줍니다. 비율이 높을수록 회사 미래에 대한 자신감이 크다는 뜻입니다. (특히 폭락장에서의 내부자 매수는 강력한 호재입니다!)
		* 🏦 **기관 지분율:** 블랙록, 뱅가드 같은 글로벌 큰손들이 얼마나 투자했는지 보여줍니다. 이 비율이 높으면 주가가 쉽게 무너지지 않는 든든한 방어막 역할을 합니다.
		""")
		
		try:
			stock_obj = yf.Ticker(ticker_symbol)
			
			# 야후 파이낸스 info에서 지분율 데이터 가져오기
			insider_pct = info.get('heldPercentInsiders', 0)
			inst_pct = info.get('heldPercentInstitutions', 0)
			
			# 데이터가 소수점(0.05)으로 들어올 때와 퍼센트(5)로 들어올 때를 방어하는 로직
			if insider_pct is not None and insider_pct < 1: insider_pct *= 100
			if inst_pct is not None and inst_pct < 1: inst_pct *= 100
			
			i_col1, i_col2 = st.columns(2)
			i_col1.metric("👔 내부자 (경영진) 지분율", f"{insider_pct:.2f}%" if insider_pct else "데이터 없음")
			i_col2.metric("🏦 글로벌 기관 지분율", f"{inst_pct:.2f}%" if inst_pct else "데이터 없음")
			
			st.write("📝 **최근 내부자 거래 동향 (Top 5)**")
			insider_trades = stock_obj.insider_transactions
			if insider_trades is not None and len(insider_trades) > 0:
				st.dataframe(insider_trades.head(5), use_container_width=True)
			else:
				st.write("최근 보고된 경영진의 주식 매수/매도 내역이 없습니다.")
				
		except Exception as e:
			st.warning("수급 데이터를 불러오는 데 실패했습니다.")

	# --- 공통 지표 연산 (앱 전체에서 사용) ---
	df['20일_이동평균'] = df['Close'].rolling(window=20).mean()
	df['60일_이동평균'] = df['Close'].rolling(window=60).mean()
	
	# ✨ 딥러닝을 위한 볼린저 밴드 지표 추가
	df['BB_상단'] = df['20일_이동평균'] + 2 * df['Close'].rolling(window=20).std()
	df['BB_하단'] = df['20일_이동평균'] - 2 * df['Close'].rolling(window=20).std()
	
	delta = df['Close'].diff()
	up = delta.clip(lower=0)
	down = -1 * delta.clip(upper=0)
	ema_up = up.ewm(com=13, adjust=False).mean()
	ema_down = down.ewm(com=13, adjust=False).mean()
	rs = ema_up / ema_down
	df['RSI'] = 100 - (100 / (1 + rs))
	exp1 = df['Close'].ewm(span=12, adjust=False).mean()
	exp2 = df['Close'].ewm(span=26, adjust=False).mean()
	df['MACD'] = exp1 - exp2
	df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
	df['수익률'] = df['Close'].pct_change()
	df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

	st.subheader("🎯 매수/매도 타이밍 분석 (기술적 지표)")

	with st.expander("📖 RSI, MACD, 정배열... 차트 지표가 무슨 뜻인가요?", expanded=False):
		st.info("""
		* 📊 **RSI (상대강도지수):** 주식이 최근 얼마나 미친 듯이 오르고 내렸는지를 0~100으로 나타낸 '온도계'입니다. **30 밑이면 사람들이 패닉셀을 해서 너무 싸진 상태(바겐세일 매수 찬스)**, **70 위면 광기가 껴서 너무 비싸진 상태(거품 붕괴 주의, 매도)**로 봅니다.
		* 📈 **MACD (추세 방향):** 주가의 단기 흐름과 장기 흐름이 만나는 지점을 분석합니다. MACD 선이 Signal 선을 **뚫고 위로 올라가면 '이제부터 오름세 시작(매수)'**, **아래로 꺾여서 내려가면 '이제부터 내림세 시작(매도)'**을 의미합니다.
		* 🌟 **이동평균선 (정배열/역배열):** 최근 20일간의 평균 주가가 과거 60일간의 평균 주가보다 위에 있는 것을 **'정배열'**이라고 합니다. 이는 최근 주가 흐름이 더 좋아져서 **안정적으로 우상향(상승장)**하고 있다는 뜻이죠. 반대로 20일 선이 아래로 처박히면 **'역배열(하락장)'**입니다.
		""")

	latest_rsi = df['RSI'].iloc[-1]
	latest_macd = df['MACD'].iloc[-1]
	latest_signal = df['Signal_Line'].iloc[-1]
	ma_20 = df['20일_이동평균'].iloc[-1]
	ma_60 = df['60일_이동평균'].iloc[-1]

	# ✨ AI의 똑똑한 종합 판단을 위한 점수 매기기 로직
	score = 0
	
	c1, c2, c3 = st.columns(3)
	with c1:
		if latest_rsi <= 30: 
			st.success(f"📊 RSI 온도계: {latest_rsi:.1f}\n\n🔥 **강한 매수 찬스 (과매도)**\n\n(사람들이 던져서 헐값입니다!)")
			score += 1
		elif latest_rsi >= 70: 
			st.error(f"📊 RSI 온도계: {latest_rsi:.1f}\n\n⚠️ **매도 주의 (과매수)**\n\n(광기가 껴서 비싼 상태입니다!)")
			score -= 1
		else: 
			st.info(f"📊 RSI 온도계: {latest_rsi:.1f}\n\n➖ **보통 (관망)**\n\n(과열되지 않은 평범한 상태)")
			
	with c2:
		if latest_macd > latest_signal: 
			st.success(f"📈 MACD 추세선\n\n**상승 추세 (매수 시그널)**\n\n(MACD가 시그널을 돌파했습니다!)")
			score += 1
		else: 
			st.error(f"📉 MACD 추세선\n\n**하락 추세 (조심!)**\n\n(MACD가 시그널 아래로 꺾였습니다.)")
			score -= 1
			
	with c3:
		if ma_20 > ma_60: 
			st.success(f"🌟 이동평균선\n\n**정배열 (안정적 상승세)**\n\n(단기 흐름이 장기 흐름을 이겼습니다!)")
			score += 1
		else: 
			st.error(f"🌧️ 이동평균선\n\n**역배열 (하락세 지속)**\n\n(단기 흐름이 꺾여서 흘러내리는 중입니다.)")
			score -= 1

	# ✨ 점수를 바탕으로 최종 결론 내려주기
	st.write("### 🤖 차트 지표 종합 평가 의견")
	if score >= 2:
		st.success("🟢 **[종합 의견: 적극 매수 찬스!]**\n차트 흐름이 매우 좋습니다. 바닥을 치고 올라오는 바겐세일 구간이거나, 안정적인 우상향 상승장을 탔습니다. 매수를 적극적으로 고려해 볼 만한 훌륭한 타이밍입니다.")
	elif score == 1:
		st.info("🟡 **[종합 의견: 분할 매수 / 지켜보기]**\n전반적인 흐름은 나쁘지 않지만 확실한 대세 상승장은 아닙니다. 섣불리 한 번에 다 사기보다는 조금씩 나누어 사거나(분할매수), 며칠 더 지켜보는 것을 추천합니다.")
	elif score == -1:
		st.warning("🟠 **[종합 의견: 비중 축소 / 주의]**\n차트가 조금씩 무너지고 있습니다. 새로 사는 것은 추천하지 않으며, 이미 주식을 들고 있다면 수익을 조금 챙겨두는(부분 매도) 방어적인 전략이 필요합니다.")
	else: # score <= -2
		st.error("🔴 **[종합 의견: 매수 금지 / 도망치세요!]**\n차트가 완전히 꺾인 하락장이거나 거품이 심하게 낀 상태입니다. '떨어지는 칼날'을 맨손으로 잡지 마세요! 지금은 매수 버튼에서 손을 떼고 도망쳐야 할 때입니다.")

	st.divider()

	st.subheader("🧪 현실 고증 퀀트 시뮬레이션 (세금/수수료 반영)")
	st.write("💡 **조건:** 지난 2년간 RSI 30 이하에서 전량 매수, 70 이상에서 전량 매도할 때의 찐수익은?")
	with st.expander("⚙️ 현실 세계 마찰력 설정 (슬리피지, 수수료, 세금)", expanded=True):
		sim_col1, sim_col2, sim_col3 = st.columns(3)
		capital = sim_col1.number_input("초기 투자 금액 ($)", min_value=1000, value=10000, step=1000)
		commission = sim_col2.number_input("증권사 거래 수수료 (%)", min_value=0.0, max_value=1.0, value=0.07, step=0.01) 
		slippage = sim_col3.number_input("슬리피지 오차 (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05) 
		st.caption("※ 해외주식 매매차익은 연 250만 원(약 $1,850) 공제 후 22% 양도소득세 부과")

	cash = capital
	shares = 0
	total_fees_paid = 0 

	for i in range(len(df)):
		price = df['Close'].iloc[i]
		rsi = df['RSI'].iloc[i]
		if rsi <= 30 and cash > 0: 
			buy_price = price * (1 + (slippage / 100))
			fee = cash * (commission / 100)
			total_fees_paid += fee
			shares = (cash - fee) / buy_price
			cash = 0
		elif rsi >= 70 and shares > 0: 
			sell_price = price * (1 - (slippage / 100))
			gross_proceeds = shares * sell_price
			fee = gross_proceeds * (commission / 100)
			total_fees_paid += fee
			cash = gross_proceeds - fee
			shares = 0

	if shares > 0:
		final_sell_price = df['Close'].iloc[-1] * (1 - (slippage / 100))
		gross_val = shares * final_sell_price
		final_value = gross_val - (gross_val * (commission / 100))
	else:
		final_value = cash

	total_profit = final_value - capital
	tax_amount = 0
	taxable_threshold_usd = 1850 
	if total_profit > taxable_threshold_usd:
		tax_amount = (total_profit - taxable_threshold_usd) * 0.22 

	net_final_value = final_value - tax_amount
	net_profit_pct = ((net_final_value - capital) / capital) * 100

	b_col1, b_col2, b_col3 = st.columns(3)
	b_col1.metric("초기 투자 금액", f"${capital:,.2f}")
	b_col2.metric("세후 최종 통장 잔고", f"${net_final_value:,.2f}", f"{net_profit_pct:+.2f}% (찐수익)")
	b_col3.metric("💸 뜯긴 돈 (수수료+세금)", f"-${(total_fees_paid + tax_amount):,.2f}")

	if tax_amount > 0:
		st.error(f"🚨 수익이 250만 원을 초과하여 양도소득세 **${tax_amount:,.2f}**가 부과되었습니다.")
	else:
		st.success("✅ 비과세 구간입니다! (수익이 250만 원 이하이거나 손실 중입니다.)")

	# --- ✨ 리팩토링 2: 딥러닝 모델 고도화 ---
	st.subheader("🤖 AI 딥러닝(LSTM) 내일 주가 예측")
	st.write("과거 10일 치의 **거래량, 변동성(볼린저 밴드), 주가 패턴**을 종합 분석해 내일의 방향성을 예측합니다.")

	if st.button("🧠 딥러닝 모델 고도화 학습 및 예측 실행", use_container_width=True):
		with st.spinner("AI가 노이즈를 제거하고 핵심 패턴을 맹렬히 학습 중입니다... (약 10~15초 소요) ⏳"):
			try:
				df_clean = df.dropna()
				
				# ✨ 1. 데이터(Feature) 확장: 거래량과 볼린저 밴드 추가
				features = ['Close', 'Volume', '수익률', 'RSI', 'MACD', 'BB_상단', 'BB_하단']
				scaler = MinMaxScaler()
				scaled_data = scaler.fit_transform(df_clean[features])

				time_step = 10
				X_lstm, y_lstm = [], []
				for i in range(len(scaled_data) - time_step):
					X_lstm.append(scaled_data[i:(i + time_step)])
					y_lstm.append(df_clean['Target'].iloc[i + time_step])

				X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

				# ✨ 2. 모델 고도화: 층을 깊게 쌓고 과적합(Overfitting) 방지
				model = Sequential()
				model.add(LSTM(50, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
				model.add(Dropout(0.2)) # 외우기 방지 레이어
				model.add(LSTM(50, return_sequences=False))
				model.add(Dropout(0.2)) # 외우기 방지 레이어
				model.add(Dense(1, activation='sigmoid'))
				
				model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
				# 에포크를 약간 늘려서 더 꼼꼼히 학습
				model.fit(X_lstm, y_lstm, epochs=10, batch_size=16, verbose=0)

				last_10_days = scaled_data[-time_step:]
				last_10_days = np.expand_dims(last_10_days, axis=0)
				prediction_prob = model.predict(last_10_days, verbose=0)[0][0]

				# ✨ 3. 전문가용 게이지 차트(Gauge Chart) 시각화
				fig_gauge = go.Figure(go.Indicator(
					mode = "gauge+number",
					value = prediction_prob * 100,
					number = {'suffix': "%", 'font': {'size': 40, 'color': 'white'}},
					domain = {'x': [0, 1], 'y': [0, 1]},
					title = {'text': "내일 주가 상승 확률", 'font': {'size': 20, 'color': 'lightgray'}},
					gauge = {
						'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
						'bar': {'color': "white", 'thickness': 0.2},
						'bgcolor': "rgba(0,0,0,0)",
						'steps': [
							{'range': [0, 45], 'color': "rgba(255, 75, 75, 0.6)"}, # 하락 유력 (빨강)
							{'range': [45, 55], 'color': "rgba(128, 128, 128, 0.3)"}, # 중립 (회색)
							{'range': [55, 100], 'color': "rgba(0, 255, 136, 0.6)"} # 상승 유력 (초록)
						],
						'threshold': {
							'line': {'color': "white", 'width': 3},
							'thickness': 0.75,
							'value': prediction_prob * 100
						}
					}
				))
				fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), template="plotly_dark")
				st.plotly_chart(fig_gauge, use_container_width=True)

				# 평가 코멘트
				if prediction_prob >= 0.55:
					st.success("🟢 **강력한 상승 시그널:** AI가 긍정적인 패턴을 포착했습니다.")
				elif prediction_prob <= 0.45:
					st.error("🔴 **강력한 하락 시그널:** AI가 부정적인 패턴(저항선 도달, 거래량 감소 등)을 포착했습니다. 주의하세요.")
				else:
					st.warning("🟡 **방향성 탐색 중:** 뚜렷한 상승/하락 패턴이 보이지 않습니다. (동전 던지기와 비슷한 확률입니다.)")
					
				st.info("💡 **알림:** 본 딥러닝 모델은 과거 패턴의 통계적 요약일 뿐, 내일 아침에 터질 뉴스나 파월 의장의 발언 같은 '외부 변수'는 모릅니다. 절대 맹신하지 마세요!")

			except Exception as e:
				st.warning(f"데이터가 부족하여 딥러닝 모델을 학습할 수 없습니다. (에러: {e})")

	st.divider()

	# ====================================================================
	# ✨ 신규 기능: AI 이벤트 추적 인터랙티브 차트 (가짜 데이터 원천 차단!)
	# ====================================================================
	st.subheader("🚀 주가 폭등/폭락 원인 추적 (AI 뉴스 요약 차트)")
	with st.expander("별표(⭐) 위에 마우스를 올려서 실제 주가 변동 이유를 확인하세요!", expanded=True):
		try:
			import time # API 과부하 방지용

			df_event = df.tail(252).copy() 
			df_event['Change'] = df_event['수익률'] * 100
			
			# 5% 이상 크게 움직인 날짜들 먼저 고르기
			significant_events = df_event[abs(df_event['Change']) > 5].copy()
			
			# 🚨 가짜 데이터 다 지우고 빈칸으로 시작!
			significant_events['Event_Text'] = ""

			if not significant_events.empty:
				with st.spinner(f"🕵️‍♂️ AI가 {ticker_symbol}의 과거 실제 뉴스를 검색 중입니다... (API 과부하 방지를 위해 5~10초 소요) ⏳"):
					# 가장 심하게 폭등/폭락한 상위 5개 날짜만 뽑기
					top_5_dates = significant_events['Change'].abs().nlargest(5).index
					
					for date in top_5_dates:
						date_str = date.strftime('%Y-%m-%d')
						change_val = significant_events.loc[date, 'Change']
						
						# 추측을 배제하고 팩트만 요구하는 프롬프트
						prompt = f"미국 주식 {ticker_symbol}이(가) {date_str}에 주가가 약 {change_val:.1f}% 변동했습니다. 이 날짜 근처의 실제 경제 뉴스, 실적발표 등 명확한 사실(Fact)에 기반한 변동 원인을 한국어로 딱 1문장(50자 이내)으로 요약하세요. 추측성 내용은 절대 포함하지 마세요."
						
						try:
							# 상단에서 정의한 글로벌 'model' 변수를 그대로 사용
							response = model.generate_content(prompt)
							real_news = response.text.replace('\n', ' ').strip()
							
							color_tag = "#00FF88" if change_val > 0 else "#FF4B4B"
							sign = "폭등" if change_val > 0 else "폭락"
							significant_events.loc[date, 'Event_Text'] = f"<b style='color:{color_tag};'>{sign}! ({change_val:.1f}%)</b><br>📰 <b>팩트 체크:</b> {real_news}"
							
							# API 연속 호출로 인한 Rate Limit(차단) 방지를 위해 1.5초 대기
							time.sleep(1.5) 
						except Exception as e:
							# 에러가 나면 숨기지 않고 실패 사유를 그대로 노출
							significant_events.loc[date, 'Event_Text'] = f"<b>({change_val:.1f}%)</b><br>⚠️ 실제 뉴스 로드 실패 (API 에러)"

			# Event_Text가 성공적으로 채워진(AI가 진짜 뉴스를 가져온) 데이터만 남기기
			valid_events = significant_events[significant_events['Event_Text'] != ""]

			# 차트 그리기
			fig_event = go.Figure()
			fig_event.add_trace(go.Scatter(
				x=df_event.index, y=df_event['Close'], 
				mode='lines', name='주가 흐름', line=dict(color='rgba(255, 255, 255, 0.4)', width=2)
			))
			
			if not valid_events.empty:
				fig_event.add_trace(go.Scatter(
					x=valid_events.index, 
					y=valid_events['Close'],
					mode='markers',
					marker=dict(
						color=['#00FF88' if c > 0 else '#FF4B4B' for c in valid_events['Change']], 
						size=18, 
						symbol='star',
						line=dict(color='white', width=1)
					),
					name='주요 이벤트 (마우스 오버)',
					text=valid_events['Event_Text'], 
					hoverinfo='text' 
				))
			
			fig_event.update_layout(
				template="plotly_dark", height=400, 
				margin=dict(l=20, r=20, t=30, b=20),
				hovermode="closest",
				yaxis_title="주가 (USD)"
			)
			st.plotly_chart(fig_event, use_container_width=True)
			st.caption("💡 팩트 체크가 완료된 Top 5 핵심 변동일만 별표(⭐)로 표시됩니다.")
			
		except Exception as e:
			st.error(f"이벤트 차트를 그리는 중 오류가 발생했습니다: {e}")

	st.divider()
	# ====================================================================

	st.subheader(f"📊 {ticker_symbol} 전문가용 캔들 차트 (최근 1년)")
	df_chart = df.tail(252)
	fig = go.Figure(data=[go.Candlestick(
		x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
		low=df_chart['Low'], close=df_chart['Close'], name='주가 캔들'
	)])
	fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['20일_이동평균'], mode='lines', name='20일 이동평균', line=dict(color='orange', width=1.5)))
	fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['60일_이동평균'], mode='lines', name='60일 이동평균', line=dict(color='blue', width=1.5)))
	fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', height=500, yaxis_title='주가 (USD)')
	st.plotly_chart(fig, use_container_width=True)

	st.subheader("RSI (상대강도지수) 차트 - 30 밑이면 매수, 70 위면 매도")
	st.line_chart(df['RSI'].tail(252))

	st.subheader("📰 최신 뉴스 & Gemini AI 3줄 요약")
	if st.button("🚀 Gemini AI 뉴스 3줄 요약 실행하기"):
		with st.spinner("AI가 월스트리트 뉴스를 싹 다 읽고 있습니다... ⏳"):
			try:
				news_md, ai_summary, sentiment_score = get_news_and_ai_summary(ticker_symbol)
				if news_md:
						st.markdown(news_md)
						st.divider()
						st.subheader("🤖 AI가 평가한 오늘의 뉴스 감성 점수")
						score_col1, score_col2 = st.columns([1, 4])
						with score_col1: st.metric("감성 점수 (0~100)", f"{sentiment_score}점")
						with score_col2:
							st.write("")
							st.progress(sentiment_score / 100.0)
							if sentiment_score >= 70: st.success("시장 분위기가 아주 좋습니다! 강력한 호재가 예상됩니다. 🚀")
							elif sentiment_score <= 30: st.error("시장 분위기가 얼어붙었습니다. 리스크 관리에 주의하세요! 🥶")
							else: st.warning("시장 분위기가 미지근합니다. 특별한 호재도 악재도 없네요. 😐")
						st.divider()
						st.write("🧠 **Gemini AI 상세 분석 리포트**")
						st.info(ai_summary)
				else:
						st.write(ai_summary)
			except Exception as e:
				st.warning("⏳ 구글 API 한도를 초과했거나 에러가 발생했습니다. 잠시 후 다시 시도해 주세요!")
	else:
		st.info("👆 위 버튼을 누르면 인공지능이 최신 뉴스를 읽고 분석해 줍니다! (할당량 절약 모드)")

	st.divider()

	# --- ✨ 거래량 심층 분석 업그레이드 ---
	st.subheader("🔥 거래량(Volume) 심층 분석: 시장의 진짜 돈의 흐름")
	
	with st.expander("📖 주가는 속여도 거래량은 못 속인다? (거래량 보는 법)", expanded=False):
		st.info("""
		* 📈 **주가 상승 + 거래량 폭발:** 세력(기관/외국인)이 진짜 돈을 싸들고 들어왔다는 뜻입니다. 가장 강력한 **진짜 상승 신호(찐반등)**입니다.
		* 📉 **주가 하락 + 거래량 폭발:** 누군가 엄청난 물량을 시장에 집어 던지고 도망갔다는 뜻입니다. **강력한 하락 신호(패닉셀)**입니다.
		* 🥱 **거래량 가뭄:** 시장의 관심이 식었음을 의미합니다. 이때 주가가 찔끔찔끔 오르는 건 적은 돈으로 쉽게 조작될 수 있는 '가짜 상승(속임수)'일 확률이 높습니다.
		* 💡 **차트 해석 꿀팁:** 빨간색/초록색 막대가 **주황색 점선(20일 평균 거래량)**을 시원하게 뚫고 솟아오른 날을 주목하세요! 그날이 바로 변곡점입니다.
		""")

	# 거래량 차트 예쁘게 그리기 (업/다운 색상 구분 + 20일 평균선)
	df_vol = df.tail(120).copy() # 너무 길면 안 보이니까 최근 6개월(120일)만 돋보기로 확대!
	df_vol['Volume_MA20'] = df_vol['Volume'].rolling(window=20).mean()
	
	# 양봉(오른 날)은 초록색, 음봉(내린 날)은 빨간색으로 칠하기
	df_vol['Color'] = np.where(df_vol['Close'] >= df_vol['Open'], 'rgba(0, 255, 136, 0.7)', 'rgba(255, 75, 75, 0.7)') 

	fig_vol = go.Figure()
	# 1. 일일 거래량 막대그래프
	fig_vol.add_trace(go.Bar(
		x=df_vol.index, 
		y=df_vol['Volume'], 
		marker_color=df_vol['Color'], 
		name='일일 거래량'
	))
	# 2. 20일 평균 거래량 선 (이 선을 넘으면 '폭발'한 것!)
	fig_vol.add_trace(go.Scatter(
		x=df_vol.index, 
		y=df_vol['Volume_MA20'], 
		mode='lines', 
		line=dict(color='orange', width=2, dash='dot'), 
		name='20일 평균선'
	))
	fig_vol.update_layout(
		template="plotly_dark", 
		height=300, 
		margin=dict(l=20, r=20, t=30, b=20),
		showlegend=False,
		hovermode="x unified"
	)
	st.plotly_chart(fig_vol, use_container_width=True)

	st.divider()

	st.subheader("🎲 닥터 스트레인지의 평행우주 (몬테카를로 시뮬레이션)")
	with st.expander(f"1년 뒤 '{short_name}' 주식에 일어날 수 있는 수많은 미래 엿보기", expanded=False):
		st.write("과거의 주가 변동성(위험)과 평균 수익률을 바탕으로 수학적 난수(주사위)를 발생시켜, 미래 주가의 확률적 분포를 시뮬레이션합니다.")
		try:
			daily_returns = df['Close'].pct_change().dropna()
			mu = daily_returns.mean()
			sigma = daily_returns.std()
			
			# ✨ 사용자가 이해하기 쉽게 '연평균(Annual)' 수치로 변환! (1년 = 252 거래일)
			annual_mu = mu * 252
			annual_sigma = sigma * np.sqrt(252)

			st.info(f"💡 과거 2년치 데이터를 분석하여 이 종목의 **연평균 수익률({annual_mu*100:.1f}%)**과 **연간 변동성({annual_sigma*100:.1f}%)**을 자동으로 세팅했습니다. 원하는 수치로 조작해 보세요!")

			mc_c1, mc_c2, mc_c3 = st.columns(3)
			sim_days = mc_c1.slider("시뮬레이션 기간 (거래일)", 30, 252, 252) 
			sim_paths = mc_c2.slider("생성할 평행우주(시나리오) 개수", 10, 500, 100)
			
			# ✨ 입력창을 연평균 기준으로 변경
			user_annual_mu = mc_c3.number_input("연평균 기대 수익률 (%)", value=float(annual_mu * 100), step=1.0)
			user_annual_sigma = mc_c3.number_input("연간 변동성 (리스크) (%)", value=float(annual_sigma * 100), step=1.0)

			if st.button("🎲 수백 개의 미래 엿보기 (시뮬레이션 실행)", type="primary", use_container_width=True):
				with st.spinner("수백 개의 평행우주를 겹쳐서 그리는 중... 🌀"):
					last_price = df['Close'].iloc[-1]
					
					# ✨ 시뮬레이션을 돌리기 위해 사용자가 입력한 연평균 값을 다시 일일 수치로 쪼개기
					daily_mu_user = (user_annual_mu / 100) / 252
					daily_sigma_user = (user_annual_sigma / 100) / np.sqrt(252)
					
					simulation_df = pd.DataFrame()
					for x in range(sim_paths):
						shock = np.random.normal(loc=daily_mu_user - (0.5 * daily_sigma_user**2), scale=daily_sigma_user, size=sim_days)
						price_path = last_price * np.exp(np.cumsum(shock))
						simulation_df[x] = price_path
					
					fig_mc = go.Figure()
					for x in range(sim_paths):
						fig_mc.add_trace(go.Scatter(x=list(range(sim_days)), y=simulation_df[x], mode='lines', line=dict(color='rgba(0, 176, 246, 0.05)'), showlegend=False, hoverinfo='skip'))
						
					median_path = simulation_df.median(axis=1)
					top_5_path = simulation_df.quantile(0.95, axis=1)
					bottom_5_path = simulation_df.quantile(0.05, axis=1)
					
					fig_mc.add_trace(go.Scatter(x=list(range(sim_days)), y=median_path, mode='lines', name='가장 흔한 미래 (중앙값)', line=dict(color='yellow', width=3)))
					fig_mc.add_trace(go.Scatter(x=list(range(sim_days)), y=top_5_path, mode='lines', name='상위 5% 초대박 우주', line=dict(color='lime', width=2, dash='dash')))
					fig_mc.add_trace(go.Scatter(x=list(range(sim_days)), y=bottom_5_path, mode='lines', name='하위 5% 최악의 우주', line=dict(color='red', width=2, dash='dash')))
					
					fig_mc.update_layout(title=f"'{ticker_symbol}' 향후 {sim_days}일 주가 시나리오", template="plotly_dark", height=500, hovermode="x unified")
					st.plotly_chart(fig_mc, use_container_width=True)
					
					final_median = median_path.iloc[-1]
					final_top = top_5_path.iloc[-1]
					final_bottom = bottom_5_path.iloc[-1]
					
					st.success("시뮬레이션 완료! 난수가 만들어낸 미래의 확률 분포입니다.")
					m_col1, m_col2, m_col3 = st.columns(3)
					m_col1.metric("가장 현실적인 미래 (중앙값)", fmt_price(final_median), f"{((final_median/last_price)-1)*100:.1f}%")
					m_col2.metric("상위 5% 대박 시나리오", fmt_price(final_top), f"{((final_top/last_price)-1)*100:.1f}%")
					m_col3.metric("하위 5% 폭망 시나리오 (최악)", fmt_price(final_bottom), f"{((final_bottom/last_price)-1)*100:.1f}%")
		except Exception as e:
			st.error(f"시뮬레이션 중 오류가 발생했습니다: {e}")

	st.divider()

	st.subheader("📄 원클릭 AI 종합 리포트 생성기")
	st.write("지금까지 분석한 모든 팩트 데이터와 AI의 인사이트를 한 장의 깔끔한 보고서로 요약해 줍니다.")

	if st.button("📝 AI 리포트 작성 시작 (약 10초 소요)", type="primary", use_container_width=True):
		with st.spinner("AI가 월스트리트 수준의 리포트를 작성 중입니다... ⏳"):
			try:
				report_model = genai.GenerativeModel('gemini-3-flash-preview')
				report_prompt = f"""
				너는 월스트리트 최고의 퀀트 애널리스트이자 깐깐한 가치 투자자야. 다음 데이터를 바탕으로 '{short_name} ({ticker_symbol})'에 대한 투자 리포트를 작성해.

				[현재 데이터 팩트]
				- 주가: {current_price} (52주 최고가: {high_52})
				- 가치지표: PER {per:.2f}, PBR {pbr:.2f}
				- 기술적지표: RSI {latest_rsi:.1f}, MACD {latest_macd:.2f}

				[요구사항]
				반드시 HTML 태그를 사용해서 예쁘게 꾸며줘. (<h1>, <h2>, <ul>, <b> 등 사용. 마크다운 기호 사용 금지)
				1. 🎯 한 줄 요약 (이 주식을 지금 사야 하는가?)
				2. 📊 기술적/가치 분석 요약 (단기 매수 타이밍 및 고평가/저평가 여부)
				3. ⏳ 장기 투자 가치 평가 (가장 중요! 이 기업을 10년 이상 장기 보유할 가치가 있는지 명확히 평가해.)
				4. 🧠 장기 투자 판단 기준 (위 3번에서 장기 투자가 적합/부적합하다고 판단한 구체적인 '핵심 기준' 3가지를 재무, 비즈니스 모델, 해자(Moat) 관점에서 논리적으로 설명해 줘.)
				5. ⚖️ 최종 투자 의견 (Strong Buy / Buy / Hold / Sell) 및 액션 플랜
				"""
				report_response = report_model.generate_content(report_prompt)

				html_report = f"""
				<!DOCTYPE html>
				<html lang="ko">
				<head>
					<meta charset="utf-8">
					<title>{ticker_symbol} AI 투자 리포트</title>
					<style>
						body {{ font-family: 'Malgun Gothic', sans-serif; line-height: 1.6; padding: 40px; color: #333; background-color: #f4f7f6; }}
						.container {{ max-width: 800px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
						h1 {{ color: #1E3A8A; border-bottom: 3px solid #1E3A8A; padding-bottom: 10px; }}
						h2 {{ color: #2563EB; margin-top: 30px; border-left: 4px solid #2563EB; padding-left: 10px; }}
						.date-badge {{ display: inline-block; background: #eee; padding: 5px 10px; border-radius: 5px; font-size: 14px; margin-bottom: 20px; }}
						.footer {{ text-align: center; margin-top: 50px; font-size: 12px; color: #888; border-top: 1px solid #eee; padding-top: 20px; }}
					</style>
				</head>
				<body>
					<div class="container">
						<h1>📈 {short_name} ({ticker_symbol}) 종합 분석 리포트</h1>
						<div class="date-badge">📅 발행일: {pd.Timestamp.now().strftime('%Y년 %m월 %d일')}</div>
						{report_response.text}
						<div class="footer">
							<p>※ 본 리포트는 AI 비서가 데이터를 기반으로 자동 생성한 것이며, 투자 결과에 대한 법적 책임을 지지 않습니다.</p>
							<p>Generated by My Quant HTS System</p>
						</div>
					</div>
				</body>
				</html>
				"""

				st.success("🎉 리포트 생성이 완료되었습니다!")
				st.info("💡 **PDF 저장 꿀팁:** 다운로드한 HTML 파일을 크롬 브라우저로 열고, **`Ctrl + P` (인쇄)를 누른 뒤 'PDF로 저장'**을 선택하면 완벽한 PDF 리포트가 완성됩니다.")
				st.download_button(label="📥 AI 리포트 다운로드 (클릭!)", data=html_report, file_name=f"{ticker_symbol}_AI_Report.html", mime="text/html")
				with st.expander("👀 리포트 미리보기", expanded=True):
					st.components.v1.html(html_report, height=600, scrolling=True)
			except Exception as e:
				st.error(f"리포트 생성 중 오류가 발생했습니다: {e}")

	st.divider()

	st.subheader("💬 내 주식 전담 AI 비서")
	st.write(f"**{ticker_symbol}** 종목이나 투자 전략에 대해 무엇이든 물어보세요!")

	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []

	for msg in st.session_state.chat_history:
		with st.chat_message(msg["role"]):
			st.markdown(msg["content"])

	if chat_input := st.chat_input("이 주식의 향후 전망을 분석해 줘!"):
		st.session_state.chat_history.append({"role": "user", "content": chat_input})
		with st.chat_message("user"):
			st.markdown(chat_input)

		with st.chat_message("assistant"):
			with st.spinner("AI 비서가 차트와 뉴스를 분석하며 답변을 작성 중입니다... ✍️"):
				try:
					chat_model = genai.GenerativeModel('gemini-3-flash-preview')
					system_prompt = f"너는 월스트리트 수석 퀀트 분석가이자 친절한 주식 멘토야. 현재 사용자는 '{ticker_symbol}' 주식 데이터를 보고 있어. 질문에 전문적이고 친절하게 대답해 줘. 질문: {chat_input}"
					response = chat_model.generate_content(system_prompt)
					
					st.markdown(response.text)
					st.session_state.chat_history.append({"role": "assistant", "content": response.text})
				except Exception as chat_e:
					st.error(f"앗, AI 응답에 문제가 생겼습니다! API 한도나 네트워크를 확인해 주세요. (에러: {chat_e})")

except Exception as e:
	st.error(f"데이터를 불러오는 데 실패했습니다. (에러: {e})")