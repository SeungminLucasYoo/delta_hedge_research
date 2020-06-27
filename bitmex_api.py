"""
Bitmex REST API Library

Timestamp 입력시 Format: "2014-12-26 11:00"

"""
import requests
import datetime
import time
import pandas as pd
import logging


class BitmexRestAPI(object):


    @staticmethod
    def get_candle_data_by_api_count(symbol, intv, start_dt, cnt):
        """
        bitmex API를 통한 캔들. timestamp가 해당 candle의 endtime이다......
        :param symbol: XBT:nearest 가능
        :param intv: 1m, 5m, 1h, 1d
        :param start_dt: '2010-01-01 11:00'
        :param cnt: up to 1000
        :return:
        """
        request_url = 'https://www.bitmex.com/api/v1/trade/bucketed'

        query_params = {
            'symbol': symbol,
            'binSize': intv,
            'startTime': start_dt,
            'count': cnt
        }
        headers = {
            'Accept': 'application/json'
        }

        result = BitmexRestAPI._request(request_url, 'GET', query_params, headers)

        return result

    @staticmethod
    def get_candle_data_by_api(symbol, intv, start_dt, end_dt):
        """

        :param symbol: XBTUSD, .....
        :param intv: 1m, 5m, 1h, 1d
        :param start_dt: '2019-01-01 00:00'
        :param end_dt:
        :return: pd.DataFrame
        """
        final_result = []

        if intv == '1m':
            time_intv = pd.Timedelta(minutes=1)
        elif intv == '5m':
            time_intv = pd.Timedelta(minutes=5)
        elif intv == '1h':
            time_intv = pd.Timedelta(hours=1)
        else:
            time_intv = pd.Timedelta(days=1)

        i = start_dt
        while True:

            result = BitmexRestAPI.get_candle_data_by_api_count(symbol, intv, i, 1000)
            final_result += result

            last_dt = datetime.datetime.strptime(final_result[-1]['timestamp'], '%Y-%m-%dT%H:%M:%S.%f%z')

            if last_dt > datetime.datetime.strptime(end_dt, '%Y-%m-%d %H:%M').astimezone(datetime.timezone.utc):

                # timestamp가 캔들의 끝나는 시간 기준이라 수정 필요
                df = pd.DataFrame(final_result)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['timestamp'] = df['timestamp'].shift(1)

                final_df = df[(df['timestamp'] >= pd.Timestamp(start_dt, tz='utc')) & (df['timestamp'] <= pd.Timestamp(end_dt, tz='utc'))].reset_index(drop=True)

                return final_df

            else:

                i = (pd.Timestamp(last_dt) + time_intv).strftime('%Y-%m-%d %H:%M')

    @staticmethod
    def get_candle_data_by_chart(symbol, intv, start_dt, end_dt):
        """
        Chart 크롤링을 통한 데이터
        :param symbol: .BXBT, .BETH, ETHUSD, ....
        :param intv: 1m, 1h
        :param start_dt: '2010-01-01 11:00'
        :param end_dt: '2010-01-31 15:00'
        :return:
        """
        if intv =='1m':
            intv = 1
        elif intv == '1h':
            intv = 60
        elif intv == '1d':
            intv = 1440

        final_df = pd.DataFrame()

        i = start_dt
        while pd.Timestamp(i) < pd.Timestamp(end_dt):
            df = BitmexRestAPI.get_candle_data_by_chart_count(symbol, intv, i, (pd.Timestamp(i)+pd.Timedelta(minutes=10079*intv)).strftime('%Y-%m-%d %H:%S'))   # up to 10080 candles available

            final_df = pd.concat([final_df, df])
            if final_df['t'].max() >= pd.Timestamp(end_dt, tz='utc'):
                final_df = final_df[(final_df['t']>=pd.Timestamp(start_dt, tz='utc'))&(final_df['t']<=pd.Timestamp(end_dt, tz='utc'))].reset_index(drop=True)
                return final_df
            else:
                i = (final_df['t'].max() + pd.Timedelta(minutes=intv)).strftime('%Y-%m-%d %H:%M')


    @staticmethod
    def get_candle_data_by_chart_count(symbol, intv, start_dt, end_dt):
        """
        Chart 크롤링을 통한 데이터
        :param symbol: .BXBT, .BETH, ETHUSD, ....
        :param intv: 1m, 1h
        :param start_dt: '2010-01-01 11:00'
        :param end_dt: '2010-01-31 15:00'
        :return:
        """
        request_url = "https://www.bitmex.com/api/udf/history"
        unix_start_dt = pd.Timestamp(start_dt).timestamp()

        unix_end_dt = pd.Timestamp(end_dt).timestamp()
        if intv == '1m':
            intv = 1
        elif intv == '1h':
            intv = 60
        elif intv == '1d':
            intv = 1440

        query_params = {
            'symbol': symbol,
            'resolution': intv,
            'from': unix_start_dt,
            'to': unix_end_dt
        }

        result = BitmexRestAPI._request(request_url, 'GET', query_params=query_params)

        final_df = pd.DataFrame(result)
        final_df['t'] = pd.to_datetime(final_df['t'], utc=True, unit='s')
        final_df = final_df[(final_df['t']>=pd.Timestamp(start_dt, tz='utc'))&(final_df['t']<=pd.Timestamp(end_dt, tz='utc'))].reset_index(drop=True)
        final_df.loc[:, 'symbol'] = symbol

        return final_df


    @staticmethod
    def get_funding_rate_by_count(symbol, start_dt, cnt):
        request_url = 'https://www.bitmex.com/api/v1/funding'

        query_params = {
            'symbol': symbol,
            'startTime': start_dt,
            'count': cnt
        }
        headers = {
            'Accept': 'application/json'
        }

        result = BitmexRestAPI._request(request_url, 'GET', query_params, headers)

        return result

    @staticmethod
    def get_funding_rate(symbol, start_dt, end_dt):

        final_result = []


        i = start_dt
        while True:

            result = BitmexRestAPI.get_funding_rate_by_count(symbol, i, 500)
            final_result += result

            last_dt = datetime.datetime.strptime(final_result[-1]['timestamp'], '%Y-%m-%dT%H:%M:%S.%f%z')

            if last_dt >= datetime.datetime.strptime(end_dt, '%Y-%m-%d %H:%M').astimezone(datetime.timezone.utc):

                df = pd.DataFrame(final_result)
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                final_df = df[(df['timestamp'] >= pd.Timestamp(start_dt, tz='utc')) & (df['timestamp'] <= pd.Timestamp(end_dt, tz='utc'))]

                return final_df

            else:

                i = (last_dt + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')

    @staticmethod
    def get_historical_settlement(symbol, start_dt, cnt):
        """
        historical settlement date, settlement price 관련 정보
        :param symbol: 원래 입력하면 잘 나와야하는데 이상하게 ETH만 입력시 결과값이 없음
        :param start_dt: '2010-01-01 11:00'
        :param end_dt:'2010-01-31 11:00'
        :param cnt: up to 500. 500개만해도
        :return: pd.DataFrame
        """
        request_url = "https://www.bitmex.com/api/v1/settlement"

        query_params = {
            # 'symbol': symbol,
            'startTime': start_dt,
            'count': cnt
        }
        headers = {
            'Accept': 'application/json'
        }

        result = BitmexRestAPI._request(request_url, 'GET', query_params, headers)

        df = pd.DataFrame(result)

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        final_df = df[df['symbol'].str.contains(symbol)]

        return final_df

    @staticmethod
    def _request(request_url, action, query_params=None, headers=None):

        while True:

            response = getattr(requests, action.lower())(request_url, headers=headers, params=query_params, verify=True)

            if response.ok:
                return BitmexRestAPI._handle_response(response)
            if response.status_code == 429:
                logging.error('Rate LIMIT. sleep 1 sec')
                time.sleep(1)



    @staticmethod
    def _handle_response(response):

        try:
            result = response.json()
            return result
        except ValueError:
            raise ValueError(response.text)


if __name__ == '__main__':

    c = BitmexRestAPI()
    # c.get_funding_rate('XBTUSD', '2020-01-01 00:00', '2020-06-20 00:00')

    # c.get_candle_data_by_api_count('XBTUSD', '1m', '2019-01-01 00:00', 1000)
    # c.get_candle_data_by_api('XBTUSD', '1m', '2019-01-01 00:00', '2019-01-07 00:00')
    # c.get_funding
    # c.get_historical_settlement('ETH', '2018-01-01 00:00', 500)
    # c.get_candle_data_by_api('ETH:quarterly', '1d', '2018-01-01 00:00', '2020-06-24 00:00')

    c.get_candle_data_by_chart('ETHUSD', '1h', '2018-08-24 00:00', '2020-05-31 00:00')