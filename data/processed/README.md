# ETF 데이터 전처리

- 실행 위치 : 프로젝트 최상위 폴더
- Python : 3.9 이상
- 패키지 설치 : `python3 -m pip install pandas==2.3.3 numpy==2.0.2`
- 실행 명령 : `python3 data/processed/preprocess.py`

- 입력 파일 : `data/raw/etf_prices.csv`
- 출력 파일 : `data/processed/etf_features.csv`
- 출력 기간 : 2023-01-03 ~ 2025-12-31
- 데이터 크기 : 752행 × 61컬럼
- 행 기준 : 미국 거래일별 1행
- 가격 기준 : 원본 `Close`
- ETF 순서 : SPY, QQQ, SMH, SCHD, XLE, EFA, SHY, IEF, TLT, TIP, LQD, GLD, DBB, DBA, VNQ
- 컬럼 순서 : `Date` → ETF별 `CLOSE`, `LOG_RETURN`, `RSI`, `MACD`

- `{TICKER}_CLOSE` : 종가, USD
- `{TICKER}_LOG_RETURN` : ln(오늘 종가 / 전 거래일 종가), 소수 단위
- `{TICKER}_RSI` : 14거래일 RSI, Wilder 방식, 0~100
- `{TICKER}_MACD` : 12거래일 EMA − 26거래일 EMA, 가격 단위

- 계산 범위 : 2022년 7월부터 종목별 계산 후 2023~2025년 추출
- RSI 초기값 : 최초 14개 상승폭·하락폭 각각의 단순평균
- RSI 갱신 : Wilder 평균, α=1/14
- RSI 중립값 : 평균 상승폭·하락폭 모두 0인 경우 50
- EMA 초기값 : 첫 종가
- MACD 저장값 : MACD선
- 보간 : 미적용
- 오류 처리 : 결측·무한대·0 이하 종가 확인 시 중단
- 추가 가공 : 배당 추가 반영·이상치 삭제·정규화·반올림 미적용
- 분할 보정 : 원본에 반영된 값 유지
- 지표 사용 시점 : 해당 거래일 종가 확정 이후
