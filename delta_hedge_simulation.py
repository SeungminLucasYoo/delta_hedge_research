"""
Report에 적은 수식을 토대로 dynamic delta hedge simulation

q_1: XBTUSD Perpetual 계약수
q_2: ETHUSD Perpetual 계약수
q_3: ETHXBT Futures 계약수

Simulation이 가능한 날짜: ETHUSD 출시날 (2018-08-02)

"""
# custom module
from bitmex_api import BitmexRestAPI
# built-in module
import logging
import pandas as pd
import matplotlib.pyplot as plt


class DeltaHedgeSimulator(BitmexRestAPI):

    _BITMEX_TAKER_FEE = 0.00075
    _BITMEX_MAKER_FEE = -0.00025
    _COIN_TRADE_FEE = 0.002

    def __init__(self, start_dt, end_dt, init_xbt, init_quanto, freq):
        """
        Simulation Setting
        :param start_dt: Simulation Start time. ex) 2019-01-01 11:00
        :param end_dt: Simulation End time. ex) 2019-01-01 11:00
        :param init_xbt: 계정이 처음에 들고 있는 XBT 갯수.
        :param init_quanto: ETHUSD Perpetual에 notional principal을 얼마나 가져갈지(XBT 단위). ex) 100: 100XBT만큼의 exposure
        :param freq: rebalancing 주기. ex) '1m', '1h', '1d', .....
        """
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.init_xbt = init_xbt
        self.init_quanto = init_quanto
        self.freq = freq

    def get_eth_xbt_futures(self):
        """
        근월물 데이터만 모으기 위한 작업
        :return: pd.DataFrame
        """
        settlement_data = pd.Timestamp(self.start_dt) - pd.Timedelta(days=93)   # 조회하려는 ETHXBT의 

    def run(self):
        
        # data loading
        eth_usd_df = DeltaHedgeSimulator.get_candle_data_by_chart('ETHUSD', self.freq, self.start_dt, self.end_dt)
        eth_xbt_df = DeltaHedgeSimulator.get_candle_data_by_chart('ETHXBT', self.freq, self.start_dt, self.end_dt)
        xbt_btc_df = DeltaHedgeSimulator.get_candle_data_by_chart('ETH:')
        
        
        q_2 = self.init_quanto