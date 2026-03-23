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
from tensorflow.keras.layers import LSTM, Dense

# 설정은 맨 위에
st.set_page_config(layout="wide")

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

	genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
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
	format_func=lambda x: f"📌 {theme_dict[x]} ({x})", # 화면에는 '📌 갓플 (AAPL)' 처럼 예쁘게 출력!
	label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.subheader("🔔 매수 타이밍 알림 봇")

if st.sidebar.button("🚀 포트폴리오 전체 스캔 및 메일 전송"):
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
						buy_list.append(f"✅ {t} (현재 RSI: {latest_rsi:.1f}) - {theme}")
				except:
					pass

		if buy_list:
			try:
				sender = st.secrets["email"]["sender"]
				password = st.secrets["email"]["password"]
				receiver = st.secrets["email"]["receiver"]

				email_body = "다음 종목들의 RSI가 30 이하로 떨어졌습니다. 바겐세일 매수 타이밍을 확인하세요!\n\n" + "\n".join(buy_list)
				msg = MIMEText(email_body)
				msg['Subject'] = "[주식 AI 봇] 강한 매수 찬스 알림!"
				msg['From'] = sender
				msg['To'] = receiver

				with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
					server.login(sender, password)
					server.send_message(msg)

				st.sidebar.success("매수 추천 종목을 이메일로 성공적으로 발송했습니다!")
			except Exception as e:
				st.sidebar.error("❌ 이메일 전송 실패! 스트림릿 Secrets 설정을 확인해 주세요.")
		else:
			st.sidebar.info("지금은 RSI 30 이하인 바겐세일 종목이 없습니다.")


st.title("주식 AI 분석 앱")

with st.expander("💼 나의 모의투자 계좌 현황", expanded=True):
	my_cash = st.session_state['account']['cash']
	my_holdings = st.session_state['account']['holdings']

	st.write(f"💵 **보유 현금:** ${my_cash:,.2f}")

	if my_holdings:
		st.write("📦 **보유 주식:**")
		for ticker, info in my_holdings.items():
				st.write(f"- **{ticker}**: {info['shares']}주 (평단가: ${info['avg_price']:,.2f})")
	else:
		st.info("현재 보유 중인 주식이 없습니다. 맘에 드는 종목을 매수해 보세요!")

st.divider()

st.subheader("🔎 내 포트폴리오 조건검색기 (스캐너)")
with st.expander("🤖 포트폴리오 전체 종목 스캔하기 (클릭!)"):
	st.write("내 관심 종목 중 현재 살만한 타이밍인 주식이 있는지 한 번에 찾아보세요!")
	
	filter_col1, filter_col2 = st.columns(2)
	rsi_condition = filter_col1.selectbox("📊 RSI (매수/매도 타이밍)", ["상관없음", "🟢 RSI 30 이하 (과매도/바겐세일 찬스!)", "🔴 RSI 70 이상 (과매수/거품 주의!)"])
	macd_condition = filter_col2.selectbox("📈 MACD (추세 방향)", ["상관없음", "🟢 MACD > Signal (상승 추세 시작)", "🔴 MACD < Signal (하락 추세 시작)"])

	if st.button("🔍 위 조건으로 전체 스캔 실행", type="primary"):
		with st.spinner("월스트리트 데이터를 긁어와 전체 종목을 수학적으로 분석 중입니다... ⏳"):
			results = []
			
			for theme, tickers_dict in st.session_state['portfolio'].items():
				for t_ticker, t_name in tickers_dict.items():
					try:
						df_temp, _ = load_data(t_ticker)
						if df_temp.empty: continue

						delta = df_temp['Close'].diff()
						up = delta.clip(lower=0)
						down = -1 * delta.clip(upper=0)
						ema_up = up.ewm(com=13, adjust=False).mean()
						ema_down = down.ewm(com=13, adjust=False).mean()
						rs = ema_up / ema_down
						df_temp['RSI'] = 100 - (100 / (1 + rs))

						exp1 = df_temp['Close'].ewm(span=12, adjust=False).mean()
						exp2 = df_temp['Close'].ewm(span=26, adjust=False).mean()
						macd = exp1 - exp2
						signal = macd.ewm(span=9, adjust=False).mean()

						latest_price = df_temp['Close'].iloc[-1]
						latest_rsi = df_temp['RSI'].iloc[-1]
						latest_macd = macd.iloc[-1]
						latest_signal = signal.iloc[-1]

						pass_rsi = True
						if "RSI 30 이하" in rsi_condition and latest_rsi > 30: pass_rsi = False
						if "RSI 70 이상" in rsi_condition and latest_rsi < 70: pass_rsi = False

						pass_macd = True
						if "MACD > Signal" in macd_condition and latest_macd <= latest_signal: pass_macd = False
						if "MACD < Signal" in macd_condition and latest_macd >= latest_signal: pass_macd = False

						if pass_rsi and pass_macd:
							results.append({
								"테마": theme,
								"종목명": t_name,
								"티커": t_ticker,
								"현재가": f"${latest_price:.2f}",
								"RSI": round(latest_rsi, 2),
								"MACD": "상승세 🟢" if latest_macd > latest_signal else "하락세 🔴"
							})
					except:
						pass # 에러나는 종목은 조용히 패스

			if results:
				st.success(f"🎉 삐빅! 조건에 딱 맞는 종목 {len(results)}개를 찾았습니다!")
				st.dataframe(pd.DataFrame(results), use_container_width=True)
			else:
				st.warning("🥲 현재 설정한 조건에 맞는 종목이 없습니다. 조건을 '상관없음'으로 조금 완화해 보세요!")
st.divider()

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
	
	if per == 0 or pbr == 0 or target_price == 0:
		fv_pe, fv_pbr, fv_target = get_finviz_data(ticker_symbol)
		per = fv_pe if per == 0 else per
		pbr = fv_pbr if pbr == 0 else pbr
		target_price = fv_target if target_price == 0 else target_price
		
	short_name = info.get('shortName', ticker_symbol)
	st.subheader(f"🏢 {short_name} 요약 정보")

	col1, col2, col3 = st.columns(3)
	col1.metric("현재 주가", f"${current_price}")
	col2.metric("PER (주가수익비율)", f"{per:.2f}" if per > 0 else "N/A")
	col3.metric("52주 최고가", f"${high_52:.2f}")

	st.write("### 모의투자 매수 / 매도")

	trade_col1, trade_col2, trade_col3 = st.columns([1, 1, 2])

	trade_shares = trade_col1.number_input("수량", min_value=1, value=1, step=1)
	total_trade_amount = trade_shares * current_price
	trade_col3.info(f"💰 총 예상 금액: **${total_trade_amount:,.2f}**\n\n(내 잔고: ${st.session_state['account']['cash']:,.2f})")

	if trade_col1.button("🟢 매수 (Buy)", use_container_width=True):
		if st.session_state['account']['cash'] >= total_trade_amount:
			st.session_state['account']['cash'] -= total_trade_amount

			holdings = st.session_state['account']['holdings']
			if ticker_symbol in holdings:
				old_shares = holdings[ticker_symbol]['shares']
				old_avg = holdings[ticker_symbol]['avg_price']
				new_shares = old_shares + trade_shares
				new_avg = ((old_shares * old_avg) + total_trade_amount) / new_shares
				holdings[ticker_symbol]['shares'] = new_shares
				holdings[ticker_symbol]['avg_price'] = new_avg
			else:
				holdings[ticker_symbol] = {'shares': trade_shares, 'avg_price': current_price}

			account_ref.set(st.session_state['account'])
			st.success(f"🎉 {ticker_symbol} {trade_shares}주 매수 완료! (DB 저장됨)")
			st.rerun()
		else:
			st.error("잔고가 부족합니다!😅")

	if trade_col2.button("🔴 매도 (Sell)", use_container_width=True):
		holdings = st.session_state['account']['holdings']
		if ticker_symbol in holdings and holdings[ticker_symbol]['shares'] >= trade_shares:
			st.session_state['account']['cash'] += total_trade_amount

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
		if target_price > current_price and current_price >0:
			up_potential = ((target_price - current_price) / current_price) * 100
			st.success(f"🎯 월스트리트 목표가: ${target_price}\n\n**+{up_potential:.1f}% 상승 여력**")
		elif target_price > 0 and current_price > 0:
			down_potential = ((current_price - target_price) / current_price) * 100
			st.error(f"🎯 월스트리트 목표가: ${target_price}\n\n**-{down_potential:.1f}% 하락 위험 (거품)**")
		else:
			st.info("목표가 정보 없음")
	
	st.divider()

	st.subheader("🏢 기업 기초체력 (펀더멘털) 분석")
	with st.expander("📊 재무제표 & 핵심 지표 열어보기 (진짜 돈 넣기 전 필수 확인!)"):
		with st.spinner("야후 파이낸스에서 기업의 재무 장부를 뒤지고 있습니다... ⏳"):
			try:
				stock_obj = yf.Ticker(ticker_symbol)
				financials = stock_obj.financials
				
				st.write("💡 **핵심 펀더멘털 지표**")
				f_col1, f_col2, f_col3, f_col4 = st.columns(4)
				
				margin = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
				roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
				debt_to_eq = info.get('debtToEquity', 0) if info.get('debtToEquity') else 0
				op_cashflow = info.get('operatingCashflow', 0) if info.get('operatingCashflow') else 0
				
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
				if not financials.empty:
					fin_df = financials.T.head(4)[::-1] # 최근 4년치 추출 후 과거->현재 순으로 시간순 배열
					
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

	df['20일_이동평균'] = df['Close'].rolling(window=20).mean()
	df['60일_이동평균'] = df['Close'].rolling(window=60).mean()

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

	st.subheader("🎯 매수/매도 타이밍 분석")
	latest_rsi = df['RSI'].iloc[-1]
	latest_macd = df['MACD'].iloc[-1]
	latest_signal = df['Signal_Line'].iloc[-1]

	c1, c2, c3 = st.columns(3)

	with c1:
		if latest_rsi <= 30:
			st.success(f"📊 RSI: {latest_rsi:.1f}\n\n🔥 **강한 매수 찬스 (과매도)**")
		elif latest_rsi >= 70:
			st.error(f"📊 RSI: {latest_rsi:.1f}\n\n⚠️ **매도 주의 (과매수)**")
		else:
			st.info(f"📊 RSI: {latest_rsi:.1f}\n\n➖ **보통 (관망)**")

	with c2:
		if latest_macd > latest_signal:
			st.success(f"📈 MACD 흐름\n\n**상승 추세 진입 (매수 시그널)**")
		else:
			st.error(f"📉 MACD 흐름\n\n**하락 추세 (조심!)**")

	with c3:
		if df['20일_이동평균'].iloc[-1] > df['60일_이동평균'].iloc[-1]:
			st.success("🌟 이동평균선\n\n**정배열 (안정적 상승세)**")
		else:
			st.error("🌧️ 이동평균선\n\n**역배열 (하락세)**")

	st.divider()

	st.subheader("🕸️ 포트폴리오 종목 상관관계 (분산투자 리스크 점검)")
	st.write("내 포트폴리오의 주식들이 서로 얼마나 비슷하게 움직이는지 확인해 보세요.")

	if st.button("📊 상관관계 히트맵 그리기"):
		with st.spinner("포트폴리오 전체 데이터를 수학적으로 분석 중입니다... ⏳"):
			try:
				all_tickers = []
				for theme, tickers in st.session_state['portfolio'].items():
					all_tickers.extend(tickers)
				all_tickers = list(set(all_tickers))

				if len(all_tickers) > 1:
					close_prices = pd.DataFrame()
					for t in all_tickers:
						df_temp, _ = load_data(t)
						if not df_temp.empty:
							close_prices[t] = df_temp['Close']

					returns = close_prices.pct_change().dropna()
					corr_matrix = returns.corr()

					fig_corr = go.Figure(data=go.Heatmap(
						z=corr_matrix.values,
						x=corr_matrix.columns,
						y=corr_matrix.index,
						colorscale='RdBu_r',
						zmin=-1, zmax=1,
						text=np.round(corr_matrix.values, 2),
						texttemplate="%{text}",
						hoverinfo="text"
					))

					fig_corr.update_layout(
						template="plotly_dark",
						height=500,
						margin=dict(l=20, r=20, t=20, b=20)
					)
					
					st.plotly_chart(fig_corr, use_container_width=True)

					st.info("💡 **데이터 해석 꿀팁:**\n* **빨간색(1.0)에 가까울수록:** 두 주식이 완전히 똑같이 움직인다는 뜻! (위험 분산 안 됨)\n* **파란색(-1.0)에 가까울수록:** 두 주식이 반대로 움직인다는 뜻! (시장 폭락 시 방어력 좋음)\n* **흰색(0)에 가까울수록:** 서로 전혀 상관없이 움직인다는 뜻입니다.")
				else:
					st.warning("상관관계를 분석하려면 포트폴리오에 최소 2개 이상의 종목이 있어야 합니다!")
			except Exception as e:
				st.error(f"데이터 분석 중 에러가 발생했습니다: {e}")

	st.divider()

	st.subheader("퀀트 투자 시뮬레이션 (백테스팅)")
	st.write("💡 **조건** 지난 2년간 RSI가 30 이하일 때 1,000달러를 전량 매수하고, 70 이상일 때 전량 매도했다면?")

	capital = 1000 # 시작 금액 1,000달러
	cash = capital
	shares = 0

	for i in range(len(df)):
		price = df['Close'].iloc[i]
		rsi = df['RSI'].iloc[i]

		if rsi <= 30 and cash > 0: # 매수
			shares = cash / price
			cash = 0
		elif rsi >= 70 and shares > 0: #매도
			cash = shares * price
			shares = 0

	# 오늘까지 안 팔고 들고 있다면 현재 주가로 가치 계산
	final_value = cash if cash > 0 else shares * df['Close'].iloc[-1]
	profit_pct = ((final_value - capital) / capital) * 100

	b_col1, b_col2 = st.columns(2)
	b_col1.metric("초기 투자 금액", f"${capital:,.2f}")

	if profit_pct > 0:
		b_col2.metric("현재 내 통장 잔고 (수익률)", f"${final_value:,.2f}", f"+{profit_pct:.2f}%")
	else:
		b_col2.metric("현재 내 통장 잔고 (수익률)", f"${final_value:,.2f}", f"{profit_pct:.2f}%")

	if shares > 0:
		st.info("현재 상태: **주식 보유 중** (아직 팔 타이밍(RSI 70)이 오지 않았습니다)")
	else:
		st.info("현재 상태: **현금 보유 중** (아직 살 타이밍(RSI 30)이 오지 않았습니다)")

	st.divider()

	st.subheader("⏳ 과거로 가는 타임머신 (적립식 투자 시뮬레이터)")
	with st.expander(f"💸 만약 내가 매달 '{short_name}' 주식을 꾸준히 샀다면?"):
		st.write(f"과거로 돌아가서 매월 일정한 금액으로 **{ticker_symbol}** 주식을 모아갔다면 지금 얼마가 되었을지 확인해 보세요!")
		
		dca_col1, dca_col2 = st.columns(2)
		monthly_inv = dca_col1.number_input("매월 투자할 금액 (달러)", min_value=10, value=100, step=10)
		invest_months = dca_col2.slider("투자 기간 (개월)", min_value=3, max_value=24, value=12) # 최대 2년(24개월)
		
		if st.button("🚀 타임머신 출발!", type="primary", use_container_width=True):
			with st.spinner("과거 데이터를 타고 시간을 거슬러 올라가는 중... 🌀"):
				try:
					df_dca = df.copy()
					
					df_monthly = df_dca.resample('M').last()
					
					df_monthly = df_monthly.tail(invest_months)
					
					if len(df_monthly) < invest_months:
						st.warning(f"💡 이 종목은 상장된 지 얼마 안 되어서 {len(df_monthly)}개월치 데이터만 시뮬레이션합니다.")
					
					total_shares = 0
					total_invested = 0
					history_invested = []
					history_value = []
					dates = []
					
					for date, row in df_monthly.iterrows():
						price = row['Close']
						if pd.isna(price): continue
						
						shares_bought = monthly_inv / price # 쪼개기 매수(소수점 주식) 가정
						total_shares += shares_bought
						total_invested += monthly_inv
						
						dates.append(date)
						history_invested.append(total_invested)
						history_value.append(total_shares * price)
					
					final_value = total_shares * current_price
					profit_money = final_value - total_invested
					profit_pct = (profit_money / total_invested) * 100 if total_invested > 0 else 0
					
					st.success(f"시뮬레이션 완료! 총 {len(dates)}개월 동안 꾸준히 모은 결과입니다.")
					
					res_col1, res_col2, res_col3 = st.columns(3)
					res_col1.metric("내 통장에서 빠져나간 원금", f"${total_invested:,.2f}")
					res_col2.metric("현재 평가 금액 (수익률)", f"${final_value:,.2f}", f"{profit_pct:.2f}%")
					res_col3.metric("누적 모은 주식 수", f"{total_shares:.2f}주")
					
					fig_dca = go.Figure()
					
					fig_dca.add_trace(go.Scatter(x=dates, y=history_invested, mode='lines', name='총 투자 원금', line=dict(color='gray', width=2, dash='dash')))
					
					line_color = '#00ff88' if profit_money >= 0 else '#ff4b4b'
					fig_dca.add_trace(go.Scatter(x=dates, y=history_value, mode='lines+markers', name='실제 평가 금액', line=dict(color=line_color, width=3)))
					
					fig_dca.update_layout(
						title=f"매월 ${monthly_inv}씩 {ticker_symbol}에 투자했을 때의 자산 변화",
						template="plotly_dark",
						xaxis_title="투자 기간",
						yaxis_title="금액 (USD)",
						hovermode="x unified",
						height=400,
						margin=dict(l=20, r=20, t=40, b=20)
					)
					st.plotly_chart(fig_dca, use_container_width=True)
					
					if profit_pct > 0:
						st.balloons()
						st.info("💡 역시 우상향하는 주식에 적립식으로 꾸준히 장기 투자하는 게 정답이네요! 시간과 복리의 마법입니다. 🧙‍♂️")
					else:
						st.warning("🥲 하락장에서는 적립식 투자도 손실을 피할 순 없네요. 하지만 비쌀 때나 쌀 때나 꾸준히 사서 '평균 단가'를 낮추는 방어 효과는 확실했습니다!")
						
				except Exception as e:
					st.error(f"시뮬레이션 중 오류가 발생했습니다: {e}")
	
	st.subheader("AI 딥러닝(LSTM) 내일 주가 예측")
	st.write("딥러닝 모델이 과거 10일치 패턴을 학습 중...⏳")

	df_clean = df.dropna()
	features = ['Close', '20일_이동평균', '60일_이동평균', '수익률', 'RSI', 'MACD']

	scaler = MinMaxScaler()
	scaled_data = scaler.fit_transform(df_clean[features])

	time_step = 10
	X_lstm, y_lstm = [], []
	for i in range(len(scaled_data) - time_step):
		X_lstm.append(scaled_data[i:(i + time_step)])
		y_lstm.append(df_clean['Target'].iloc[i + time_step])

	X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

	model = Sequential()
	model.add(LSTM(50, return_sequences=False, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	model.fit(X_lstm, y_lstm, epochs=5, batch_size=16, verbose=0)

	last_10_days = scaled_data[-time_step:]
	last_10_days = np.expand_dims(last_10_days, axis=0)

	prediction_prob = model.predict(last_10_days, verbose=0)[0][0]

	if prediction_prob > 0.5:
		st.success(f"딥러닝 예측: 내일은 **상승**할 확률이 높습니다! (상승 확률: {prediction_prob*100:.1f}%)")
	else:
		st.error(f"딥러닝 예측: 내일은 **하락**할 확률이 높습니다. (하락 확률: {(1-prediction_prob)*100:.1f}%)")

	st.divider()

	st.subheader(f"📊 {ticker_symbol} 전문가용 캔들 차트 (최근 1년)")

	df_chart = df.tail(252)

	fig = go.Figure(data=[go.Candlestick(
		x=df_chart.index,
		open=df_chart['Open'],
		high=df_chart['High'],
		low=df_chart['Low'],
		close=df_chart['Close'],
		name='주가 캔들'
	)])

	fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['20일_이동평균'], mode='lines', name='20일 이동평균', line=dict(color='orange', width=1.5)))
	fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['60일_이동평균'], mode='lines', name='60일 이동평균', line=dict(color='blue', width=1.5)))

	fig.update_layout(
		xaxis_rangeslider_visible=False,
		template='plotly_dark',
		margin=dict(l=20, r=20, t=40, b=20),
		height=500,
		yaxis_title='주가 (USD)'
	)

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

						with score_col1:
							st.metric("감성 점수 (0~100)", f"{sentiment_score}점")

						with score_col2:
							st.write("")
							st.progress(sentiment_score / 100.0)

							if sentiment_score >= 70:
								st.success("시장 분위기가 아주 좋습니다! 강력한 호재가 예상됩니다. 🚀")
							elif sentiment_score <= 30:
								st.error("시장 분위기가 얼어붙었습니다. 리스크 관리에 주의하세요! 🥶")
							else:
								st.warning("시장 분위기가 미지근합니다. 특별한 호재도 악재도 없네요. 😐")
						st.divider()
						st.write("🧠 **Gemini AI 상세 분석 리포트**")
						st.info(ai_summary)
				else:
						st.write(ai_summary)
			except Exception as e:
				if "429" in str(e) or "quota" in str(e).lower():
					st.warning("⏳ 구글 API 한도를 초과했습니다. 잠시 후 다시 버튼을 눌러주세요!")
				else:
					st.warning(f"에러가 발생했습니다: {e}")
	else:
		st.info("👆 위 버튼을 누르면 인공지능이 최신 뉴스를 읽고 분석해 줍니다! (할당량 절약 모드)")

	st.subheader("거래량 (Volume)")
	st.bar_chart(df['Volume'])

	st.write("최근 데이터 확인하기:")
	st.dataframe(df.tail())

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