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

if 'portfolio' not in st.session_state:
	st.session_state['portfolio'] = {
		"💻 빅테크 & AI": ['AAPL', 'MSFT', 'GOOGL'],
		"🏎️ F1 & 모터스포츠": ['FWONK', 'RACE', 'F'],
		"🎮 PC 게임 & 하드웨어": ['NVDA', 'AMD', 'EA'],
		"🧪 화학 & 헬스케어": ['JNJ', 'PFE', 'TMO'],
		"📈 금융 & 데이터": ['JPM', 'PLTR', 'SNOW']
	}

st.sidebar.title("📁 내 포트폴리오")

with st.sidebar.expander("➕ 새 종목/테마 추가하기"):
	new_theme = st.text_input("테마 이름 (기존 테마 입력 시 추가됨)", "💻 빅테크 & AI")
	new_ticker = st.text_input("추가할 티커 (예: AMZN)").upper()

	if st.button("리스트에 추가"):
		if new_ticker:
			if new_theme not in st.session_state['portfolio']:
				st.session_state['portfolio'][new_theme] = []

			if new_ticker not in st.session_state['portfolio'][new_theme]:
				st.session_state['portfolio'][new_theme].append(new_ticker)
				st.success(f"'{new_theme}' 테마에 {new_ticker} 추가 완료!")
			else:
				st.warning("이미 있는 종목입니다.")

st.sidebar.divider()
st.sidebar.write("👇 분석할 테마와 종목을 선택하세요")

selected_theme = st.sidebar.selectbox("📂 관심 테마", list(st.session_state['portfolio'].keys()))
selected_ticker = st.sidebar.radio(f"{selected_theme} 종목", st.session_state['portfolio'][selected_theme], label_visibility="collapsed")

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
ticker_symbol = st.text_input("직접 검색하거나 왼쪽 리스트에서 선택하세요:", selected_ticker).upper()

try:
	df, info = load_data(ticker_symbol)

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

	# 기술적 지표 계산
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

	st.subheader(f"{ticker_symbol} 주가 및 이동평균선 흐름")
	st.line_chart(df[['Close', '20일_이동평균', '60일_이동평균']].tail(252))

	st.subheader("RSI (상대강도지수) 차트 - 30 밑이면 매수, 70 위면 매도")
	st.line_chart(df['RSI'].tail(252))

	st.subheader("최신 뉴스 & AI 감성 분석")

	try:
		url = f"https://news.google.com/rss/search?q={ticker_symbol}+stock&hl=en-US&gl=US&ceid=US:en"
		req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
		response = urllib.request.urlopen(req)
		root = ET.fromstring(response.read())
	
		analyzer = SentimentIntensityAnalyzer()
		items = root.findall('.//item')
	
		if items:
			for item in items[:5]:
				title = item.find('title').text
				link = item.find('link').text
				publisher = item.find('source').text if item.find('source') is not None else "News"

				score = analyzer.polarity_scores(title)['compound']
	
				if score >= 0.05:
					sentiment = "🟢 호재 (긍정적)"
				elif score <= -0.05:
					sentiment = "🔴 악재 (부정적)"
				else:
					sentiment = "⚪ 중립"

				st.markdown(f"- **[{title}]({link})** ({publisher}) ➔ **분석: {sentiment}** (점수: {score:.2f})")
		else:
			st.write("현재 이 종목에 대한 최신 뉴스가 없습니다.")

	except Exception as e:
		st.warning(f"뉴스를 불러오는 중 문제가 발생했습니다: {e}")

	st.subheader("거래량 (Volume)")
	st.bar_chart(df['Volume'])

	st.write("최근 데이터 확인하기:")
	st.dataframe(df.tail())

except Exception as e:
	st.error(f"데이터를 불러오는 데 실패했습니다. (에러: {e})")