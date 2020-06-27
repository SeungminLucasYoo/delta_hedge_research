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

    _QUANTO_MULTIPLIER = 10**(-6)

    def __init__(self, start_dt, end_dt, init_xbt, init_quanto, freq, fee):
        """
        Simulation Setting
        :param start_dt: Simulation Start time. ex) 2019-01-01 11:00
        :param end_dt: Simulation End time. ex) 2019-01-01 11:00
        :param init_xbt: 계정이 처음에 들고 있는 XBT 갯수.
        :param init_quanto: ETHUSD Perpetual에 notional principal을 얼마나 가져갈지(XBT 단위). ex) 100: 100XBT만큼의 exposure
        :param freq: rebalancing 주기. ex) '1m', '1h', '1d', .....
        :param fee: TAKER or MAKER
        """
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.init_total_xbt = init_xbt
        self.init_quanto = init_quanto
        self.freq = freq

        if fee == 'TAKER':
            self.fee = DeltaHedgeSimulator._BITMEX_TAKER_FEE
        else:
            self.fee = DeltaHedgeSimulator._BITMEX_MAKER_FEE

    def get_eth_xbt_futures_data(self):
        """
        근월물 데이터만 모으기 위한 작업(만기 시간에는 청산되는 선물과 근월물 데이터가 같이 있음(주의))
        :return: pd.DataFrame
        """
        settlement_start_dt = pd.Timestamp(self.start_dt) - pd.Timedelta(days=93)   # 조회하려는 ETHXBT의 과거 settlement history

        settlement_df = self.get_historical_settlement('ETH', settlement_start_dt, 500)

        nearest_settlement_df = settlement_df[settlement_df['timestamp'] > pd.Timestamp(self.start_dt, tz='utc')]   # 실제 거래에 쓰이는 근월물 데이터

        final_df = pd.DataFrame()
        i = self.start_dt
        for _, row in nearest_settlement_df.iterrows():
            settle_date = row['timestamp'].strftime('%Y-%m-%d %H:%M')

            df = self.get_candle_data_by_chart(row['symbol'], freq, i, settle_date)

            final_df = pd.concat([final_df, df])

            if final_df['t'].max() >= pd.Timestamp(self.end_dt, tz='utc'):
                break

            i = settle_date

        if settle_date < self.end_dt:
            last_symbol = final_df['symbol'].iloc[-1]
            if last_symbol[3] == 'H':
                last_symbol = last_symbol[:3] + 'M' + last_symbol[-2:]
            elif last_symbol[3] == 'M':
                last_symbol = last_symbol[:3] + 'U' + last_symbol[-2:]
            elif last_symbol[3] == 'U':
                last_symbol = last_symbol[:3] + 'Z' + last_symbol[-2:]
            elif last_symbol[3] == 'Z':
                last_symbol = last_symbol[:3] + 'M' + str(int(last_symbol[-2:])+1)

            df = self.get_candle_data_by_chart(last_symbol, freq, settle_date, self.end_dt)
            final_df = pd.concat([final_df, df])

        final_df = final_df.merge(nearest_settlement_df[['symbol', 'settledPrice', 'timestamp']], how='left')

        return final_df

    @staticmethod
    def calc_average_price(symbol, q_1, p_1, q_2, p_2):
        """
        inverse contract라 평단가 계산이 XBT로 되어야
        :param symbol: XBTUSD, ETHXBT
        :param q_1: 계약 수
        :param p_1: 가격
        :param q_2: 계약 수
        :param p_2: 가격
        :return:
        """
        if symbol == 'XBTUSD':
            return (q_1 + q_2) / (q_1/p_1 + q_2/p_2)
        elif symbol == 'ETHUSD' or symbol == 'ETHXBT':
            return (q_1*p_1 + q_2*p_2)/(q_1 + q_2)

    @staticmethod
    def calc_pnl(symbol, p_1, p_2, q):
        if symbol =='XBTUSD':
            return (1/p_1 - 1/p_2)*q
        elif symbol == 'ETHXBT':
            return (p_2-p_1)*q
        elif symbol == 'ETHUSD':
            return (p_2 - p_1)*DeltaHedgeSimulator._QUANTO_MULTIPLIER*q

    @staticmethod
    def rebalancing_result(symbol, avg_p, q, p, delta_q):
        """

        :param symbol: ETHXBT, XBTUSD
        :param avg_p: 기존 평단가
        :param q: 기존 포지션 계약 수
        :param p: 추가 포지션 진입가격
        :param delta_q: 추가 포지션 계약 수
        :return: (xbt_real_pnl, xbt_unreal_pnl,new_avg_p)
        """
        new_q = q + delta_q # 총 계약수

        if q * delta_q > 0: # 추가 포지션 진입. 평단가 갱신, realised pnl = 0
            new_avg_p = DeltaHedgeSimulator.calc_average_price(symbol, q, avg_p, delta_q, p)    # 추가 진입에 따른 평단가 변화
            xbt_unreal_pnl = DeltaHedgeSimulator.calc_pnl(symbol, new_avg_p, p, new_q)  # 평단가 변화에 따른 unrealised pnl
            xbt_real_pnl = 0

        elif q * delta_q < 0: # 일부청산 or 전부 청산 or 반대방향으로 포지션 돌림. xbt기준 손익 계산,

            if new_q * q > 0:   # 포지션 일부 청산. 평단가 그대로, pnl은 축소시킨 양 만큼
                new_avg_p = avg_p   # 평단가 그대로
                xbt_real_pnl = DeltaHedgeSimulator.calc_pnl(symbol, avg_p, p, -delta_q)   # 포지션 축소만큼 realised
                xbt_unreal_pnl = DeltaHedgeSimulator.calc_pnl(symbol, new_avg_p, p, new_q)

            elif new_q * q == 0:    # 포지션 전부 청산 (roll over 여기 해당)
                new_avg_p = 0   # 전부 청산이라 평단가는 0
                xbt_real_pnl = DeltaHedgeSimulator.calc_pnl(symbol, avg_p, p, -delta_q) # 포지션 줄인만큼 realised
                xbt_unreal_pnl = 0

            else:   # 포지션 정리 후 반대방향으로.
                new_avg_p = p
                xbt_real_pnl = DeltaHedgeSimulator.calc_pnl(symbol, avg_p, p, q)
                xbt_unreal_pnl = 0

        else: # 신규 포지션 진입(delta_q !=0) or 변화 없음(q != 0)

            if q == 0 and delta_q != 0:  # 신규 포지션 진입
                new_avg_p = p
                xbt_real_pnl = 0
                xbt_unreal_pnl = 0

            elif q ==0 and delta_q == 0:   # 원래 포지션도 없고 포지션 변화 없음
                new_avg_p = 0
                xbt_real_pnl = 0
                xbt_unreal_pnl = 0
            else:   # 원래 포지션 그대로 유지.
                new_avg_p = avg_p
                xbt_real_pnl = 0
                xbt_unreal_pnl = DeltaHedgeSimulator.calc_pnl(symbol, avg_p, p, new_q)
        return xbt_real_pnl, xbt_unreal_pnl, new_avg_p

    def run(self):

        # data loading
        eth_usd_df = DeltaHedgeSimulator.get_candle_data_by_chart('ETHUSD', self.freq, self.start_dt, self.end_dt).set_index('t')
        xbt_usd_df = DeltaHedgeSimulator.get_candle_data_by_chart('XBTUSD', self.freq, self.start_dt, self.end_dt).set_index('t')
        eth_xbt_df = DeltaHedgeSimulator.get_eth_xbt_futures_data(self).set_index('t') # 만기때는 settle되는 선물과 근월물 2개의 데이터가 있음

        # data check
        if self.freq == '1m':
            correct_length = len(pd.date_range(self.start_dt, self.end_dt, freq='T'))
        elif self.freq == '1h':
            correct_length = len(pd.date_range(self.start_dt, self.end_dt, freq='H'))
        elif self.freq == '1d':
            correct_length = len(pd.date_range(self.start_dt, self.end_dt, freq='D'))

        if correct_length**3 != len(eth_usd_df.index)*len(xbt_usd_df.index)*len(eth_xbt_df.index.drop_duplicates()):
            logging.critical('DATA LENGTH MISMATCH')
            raise ValueError
        else:
            logging.warning('DATA LENGTH CORRECT')

        settle_info_dict = eth_xbt_df[['settledPrice', 'timestamp']].drop_duplicates().set_index('timestamp')['settledPrice'].to_dict()

        eth_usd_funding_df = DeltaHedgeSimulator.get_funding_rate('ETHUSD', self.start_dt, self.end_dt)
        xbt_usd_funding_df = DeltaHedgeSimulator.get_funding_rate('XBTUSD', self.start_dt, self.end_dt)

        enter_eth_usd_price = eth_usd_df.loc[pd.Timestamp(self.start_dt), 'o']  # 초기 ETHUSD Perp 진입가격
        enter_xbt_usd_price = xbt_usd_df.loc[pd.Timestamp(self.start_dt), 'o']  # 초기 XBTUSD Perp 진입가격

        q_2 = round(- self.init_quanto / DeltaHedgeSimulator._QUANTO_MULTIPLIER / enter_eth_usd_price)   # ETHUSD Perp 진입수량

        prev_q_1 = 0
        prev_q_3 = 0

        # state
        avg_xbt_usd_price = 0   # 유지 계약 평단가
        avg_eth_xbt_price = 0   # 유지 계약 평단가
        avg_eth_usd_price = 0   # 고정된 계약을 들어가서 이전 가격

        # rebalancing마다 발생하는 누적손익 저장 list
        pnl_xbt_usd_list = []
        pnl_eth_usd_list = []
        pnl_eth_xbt_list = []

        # trading fee, funding fee, roll over pnl은 누적이 아닌 spot
        fee_xbt_usd_list = []
        fee_eth_xbt_list = []
        eth_usd_funding_fee_list = []
        xbt_usd_funding_fee_list = []
        pnl_roll_over_list = []

        for t, row in eth_usd_df.iterrows():    # simulation start

            eth_usd_price = eth_usd_df.loc[t, 'o']
            xbt_usd_price = xbt_usd_df.loc[t, 'o']

            # 리밸런싱 수량 계산
            q_3 = round(- q_2 * DeltaHedgeSimulator._QUANTO_MULTIPLIER * xbt_usd_price)
            q_1 = round(-self.init_total_xbt * xbt_usd_price - DeltaHedgeSimulator._QUANTO_MULTIPLIER * q_2 * enter_xbt_usd_price * (eth_usd_price - enter_eth_usd_price + enter_eth_usd_price * xbt_usd_price / enter_xbt_usd_price))

            delta_q_3 = q_3 - prev_q_3
            delta_q_1 = q_1 - prev_q_1
            # 리밸런싱 진행. XBTUSD 손익 계산
            xbt_usd_real_pnl, xbt_usd_unreal_pnl, avg_xbt_usd_price = DeltaHedgeSimulator.rebalancing_result('XBTUSD', avg_xbt_usd_price, prev_q_1, xbt_usd_price, delta_q_1)
            xbt_usd_fee = abs(delta_q_1)/xbt_usd_price * self.fee

            total_xbt_usd_pnl = xbt_usd_real_pnl + xbt_usd_unreal_pnl
            # 리밸런싱 진행. ETHXBT 손익 계산
            if t in list(settle_info_dict.keys()): # 만기 청산 고려
                settle_price = settle_info_dict[t]
                roll_over_pnl, _, _ = DeltaHedgeSimulator.rebalancing_result('ETHXBT', avg_eth_xbt_price, prev_q_3, settle_price, -prev_q_3)
                eth_xbt_price = eth_xbt_df.loc[t, 'o'].iloc[-1] # 신규진입 근월물 가격
                eth_xbt_real_pnl, eth_xbt_unreal_pnl, avg_eth_xbt_price = DeltaHedgeSimulator.rebalancing_result('ETHXBT', 0 , 0, eth_xbt_price, q_3)
                # settlement fee는 0, 신규진입 포지션 수수료 고려
                eth_xbt_fee = abs(q_3) * eth_xbt_price * self.fee

            else:   # 만기 아님.
                eth_xbt_price = eth_xbt_df.loc[t, 'o']
                eth_xbt_real_pnl, eth_xbt_unreal_pnl, avg_eth_xbt_price = DeltaHedgeSimulator.rebalancing_result('ETHXBT', avg_eth_xbt_price, prev_q_3, eth_xbt_price, delta_q_3)
                eth_xbt_fee = abs(delta_q_3)*eth_xbt_price * self.fee
                roll_over_pnl = 0

            total_eth_xbt_pnl = roll_over_pnl + eth_xbt_real_pnl + eth_xbt_unreal_pnl

            # ETHUSD quanto의 손익 계산 (수수료는 초기 진입에 이미 포함)
            total_eth_usd_pnl = DeltaHedgeSimulator.calc_pnl('ETHUSD', enter_eth_usd_price, eth_usd_price, q_2)
            if t == pd.Timestamp(self.start_dt, tz='utc'):
                total_eth_usd_pnl += - self.init_quanto*self.fee    # 고정수량만큼 quanto를 매도하기 때문에 초기에 비용 나감.

            # Funding 생각하기.
            if t in eth_usd_funding_df['timestamp'].to_list():

                eth_usd_funding_rate = eth_usd_funding_df.loc[eth_usd_funding_df['timestamp'] == t, 'fundingRate'].iloc[0]
                xbt_usd_funding_rate = xbt_usd_funding_df.loc[xbt_usd_funding_df['timestamp'] == t, 'fundingRate'].iloc[0]

                eth_usd_funding_fee = q_2*eth_usd_price*DeltaHedgeSimulator._QUANTO_MULTIPLIER * eth_usd_funding_rate # q_2: short이라 음수.
                xbt_usd_funding_fee = q_1 / xbt_usd_price * xbt_usd_funding_rate    # q_1:

            else:
                eth_usd_funding_fee = 0
                xbt_usd_funding_fee = 0

            # current status update
            prev_q_1 = q_1
            prev_q_3 = q_3

            # delta hedge pnl appending
            pnl_xbt_usd_list.append(total_xbt_usd_pnl)
            pnl_eth_usd_list.append(total_eth_usd_pnl)
            pnl_eth_xbt_list.append(total_eth_xbt_pnl)

            # funding appending
            eth_usd_funding_fee_list.append(eth_usd_funding_fee)
            xbt_usd_funding_fee_list.append(xbt_usd_funding_fee)

            # fee related appending
            pnl_roll_over_list.append(roll_over_pnl)
            fee_xbt_usd_list.append(xbt_usd_fee)
            fee_eth_xbt_list.append(eth_xbt_fee)

        final_df = pd.DataFrame(index=eth_usd_df.index)
        final_df['pnl_xbt_usd'] = pnl_xbt_usd_list
        final_df['pnl_eth_usd'] = pnl_eth_usd_list
        final_df['pnl_eth_xbt'] = pnl_eth_xbt_list
        final_df['funding_fee_eth_usd'] = eth_usd_funding_fee_list
        final_df['funding_fee_xbt_usd'] = xbt_usd_funding_fee_list
        final_df['pnl_roll_over'] = pnl_roll_over_list
        final_df['fee_xbt_usd'] = fee_xbt_usd_list
        final_df['fee_eth_xbt'] = fee_eth_xbt_list

        # funding fee, roll over, trading fee를 누적으로 수정.
        final_df['funding_fee_xbt_usd'] = final_df['funding_fee_xbt_usd'].cumsum()
        final_df['funding_fee_eth_usd'] = final_df['funding_fee_eth_usd'].cumsum()
        final_df['pnl_roll_over'] = final_df['pnl_roll_over'].cumsum()
        final_df['fee_xbt_usd'] = final_df['fee_xbt_usd'].cumsum()
        final_df['fee_eth_xbt'] = final_df['fee_eth_xbt'].cumsum()

        final_df['total_xbt'] = final_df['pnl_xbt_usd'] + final_df['pnl_eth_usd'] + final_df['pnl_eth_xbt'] + final_df['pnl_roll_over'] - final_df['fee_xbt_usd'] - final_df['fee_eth_xbt'] - final_df['funding_fee_eth_usd'] - final_df['funding_fee_xbt_usd'] + self.init_total_xbt

        btbx_index = DeltaHedgeSimulator.get_candle_data_by_chart('.BXBT', self.freq, self.start_dt, self.end_dt).set_index('t')

        btbx_index.rename(columns={'o': 'bxbt_price'}, inplace=True)

        final_df = final_df.join(btbx_index['bxbt_price'])

        final_df.to_csv(f'simulation_result/{pd.Timestamp(self.start_dt).strftime("%Y%m%d%H%M%S")}_{pd.Timestamp(self.end_dt).strftime("%Y%m%d%H%M%S")}_{self.init_total_xbt}_{self.init_quanto}_{self.fee}_{self.freq}.csv')


    # def analysis(self):















if __name__ == '__main__':

    start_dt = '2018-08-04 00:00'
    end_dt = '2020-05-31 00:00'
    init_xbt = 1000
    init_quanto = 500
    freq = '1m'
    c = DeltaHedgeSimulator(start_dt, end_dt, init_xbt, init_quanto, freq, 'TAKER')
    c.run()
    # c.get_eth_xbt_futures_data()