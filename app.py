import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import urllib.request
import xml.etree.ElementTree as ET

# 설정은 맨 위에
st.set_page_config(layout="wide")

if 'watchlist' not in st.session_state:
	st.session_state['watchlist'] = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'JPM']

st.sidebar.title("⭐ 나의 관심 종목")
new_ticker = st.sidebar.text_input("새 종목 티커 추가 (예: AMZN, GOOGL)")

if st.sidebar.button("리스트에 추가"):
	new_ticker = new_ticker.upper()
	if new_ticker and new_ticker not in st.session_state['watchlist']:
		st.session_state['watchlist'].append(new_ticker)
		st.sidebar.success(f"{new_ticker} 추가 완료!")
	elif new_ticker in st.session_state['watchlist']:
		st.sidebar.warning("이미 리스트에 있는 종목입니다.")

st.sidebar.divider()
st.sidebar.write("👇 분석할 종목을 클릭하세요")
selected_ticker = st.sidebar.radio("Watchlist", st.session_state['watchlist'], label_visibility="collapsed")

st.title("주식 AI 분석 앱")
ticker_symbol = st.text_input("직접 검색하거나 왼쪽 리스트에서 선택하세요:", selected_ticker).upper()

try:
	ticker_data = yf.Ticker(ticker_symbol)
	df = ticker_data.history(period = "2y")
	info = ticker_data.info

	current_price = info.get('currentPrice')
	if current_price is None:
		current_price = info.get('regularMarketPrice', 0)
	
	st.subheader(f"{info.get('shortName', ticker_symbol)} 요약 정보")
	col1, col2, col3 = st.columns(3)
	col1.metric("현재 주가", f"${info.get('currentPrice', 0)}")
	col2.metric("PER (주가수익비율)", info.get('trailingPE', 'N/A'))
	col3.metric("52주 최고가", f"${info.get('fiftyTwoWeekHigh', 0)}")

	st.divider()

	st.subheader("⚖️ 기업 가치 평가 (고평가 vs 저평가)")
	per = info.get('trailingPE', 0)
	pbr = info.get('priceToBook', 0)
	target_price = info.get('targetMeanPrice', 0)

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
	
	# AI 예측
	df_clean = df.dropna()
	# AI가 RSI랑 MACD 보고 판단
	features = ['Close', '20일_이동평균', '60일_이동평균', '수익률']
	x = df_clean[features]
	y = df_clean['Target']

	model = RandomForestClassifier(n_estimators = 100, random_state = 42)
	model.fit(x, y)

	latest_data = x.iloc[-1:]
	prediction = model.predict(latest_data)
	probability = model.predict_proba(latest_data) [0]

	st.subheader("AI의 내일 주가 예측 결과")
	if prediction[0] == 1:
		st. success(f"내일은 **상승**할 것으로 예측됩니다! (상승 확률: {probability[1]*100:.1f}%_")
	else:
		st.error(f"내일은 **하락**할 것으로 예측됩니다. (하락 확률: {probability[0]*100:.1f}%)")

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